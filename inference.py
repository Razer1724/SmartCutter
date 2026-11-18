import os
import sys
import torch
import torchaudio
import glob
import math
import gc

from model_v3 import DSCA_ResUNet_v3
from smartcutter.config import CutConfig
from smartcutter.cuts import plan_cuts, render_audio, runs, _inward_snap
from smartcutter.frontend import legacy_features

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

#  INFER AND PROCESSING CONFIG
MODEL_VERSION = "v3"
FORCE_CPU = False                                       #   By default runs with GPU Acceleration ( CUDA )
MASK_MODE = "Soft"                                      #   Available:  "Soft", "Hard", "PowerMean" and "Hybrid"
DEBUG_MASK_PRED = True                                 #   Set to True if you need to debug / predict the model's prediction on your samples.
SAVE_EXTENSION = "wave_16"                              #   Available:  "flac", "wave_16" and "wave_32float"

#  SMART CUTTER CONFIG ( Safe defaults. )
SILENCE_TARGET_DURATION = 0.100                         #   The target duration for silence gaps (e.g. 500ms gap / silence --> 100ms)
MIN_SEGMENT_DURATION_MS = 100                           #   Minimum length for detected spots to count as viable for cutting ( 100ms, safe default. )

#  PREDICTION STABILIZATION
STABILITY_NOISE = False                                 #   Injects subtle noise into pure silence to stabilize the model
STABILITY_DB_LEVEL = -75.0                              #   The dB level of the injected noise  ( -80 is minimum;  Model's limitation. )
STABILITY_FADE_MS = 1                                   #   Fade duration (ms) for the injected noise edges to be softer

#  PATHS
IN_DIR = "infer_input"
OUT_DIR = "infer_output"
CKPT_DIR = "ckpts"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



# Estabilished safe params, do not tweak these unless necessary and you know what you're doing.
SEARCH_WINDOW_MS = 25
FADE_DURATION_MS = 10
CUTTING_PROBABILITY = 0.5
SAFETY_BUFFER_MS = 5
SEGMENT_LEN = 8.0



def get_cosine_fade(length, device):
    # Generates a raised cosine curve (half-Hanning).
    # It's mathematically smoother than a linear fade, protecting spectral integrity.
    t = torch.linspace(0, math.pi, length, device=device)
    fade_curve = 0.5 * (1 - torch.cos(t))
    return fade_curve

def apply_fade(waveform, fade_samples, mode="both"):
    # Applies the cosine fade
    if waveform.shape[1] < fade_samples * 2:
        return waveform

    fade_curve = get_cosine_fade(fade_samples, waveform.device)

    if mode == "in" or mode == "both":
        # Fade In: curve goes 0 -> 1
        waveform[:, :fade_samples] *= fade_curve

    if mode == "out" or mode == "both":
        # Fade Out: flip the curve so it goes 1 -> 0
        waveform[:, -fade_samples:] *= fade_curve.flip(0)

    return waveform

def inject_stability_noise(wav, sr, device):
    """
    Injects steady, neutral colored noise
    """
    noise_amp = 10 ** (STABILITY_DB_LEVEL / 20.0)

    silence_mask = (wav.squeeze(0) == 0.0).float()
    diff = torch.diff(silence_mask, prepend=torch.tensor([0.0], device=device), append=torch.tensor([0.0], device=device))
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    if len(starts) == 0:
        return wav

    raw_noise = torch.randn_like(wav)

    alpha = 0.85
    neutral_noise = torchaudio.functional.lfilter(
        raw_noise, 
        torch.tensor([1.0, 0.0], device=device), 
        torch.tensor([1.0, -alpha], device=device)
    )
    neutral_noise *= noise_amp
    fade_samples = int(sr * (STABILITY_FADE_MS / 1000.0))

    for start, end in zip(starts, ends):
        length = end - start
        if length <= 0: continue

        noise_chunk = neutral_noise[:, start:end].clone()

        if length > fade_samples * 2:
            noise_chunk = apply_fade(noise_chunk, fade_samples, mode="both")
        else:
            window = torch.hann_window(length, device=device)
            noise_chunk *= window

        wav[:, start:end] = noise_chunk

    return wav

def find_cuts(mask, waveform, sr):
    """Legacy sample-mask API. Snapping may only shrink the approved interval."""
    mask = mask.detach().cpu().reshape(-1)
    if mask.numel() != waveform.shape[-1]:
        raise ValueError("Sample mask and audio lengths differ")
    starts, ends = [], []
    margin = round(sr * SAFETY_BUFFER_MS / 1000)
    minimum = round(sr * MIN_SEGMENT_DURATION_MS / 1000)
    for first, last in runs((mask > CUTTING_PROBABILITY).numpy()):
        if last - first < minimum or last - first <= 2 * margin:
            continue
        start, end = _inward_snap(waveform.detach().cpu().numpy(), first + margin, last - margin,
                                  round(sr * SEARCH_WINDOW_MS / 1000))
        if start < end:
            starts.append(start)
            ends.append(end)
    return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long)

def SmartCutter(waveform, mask, sr=48000):
    """Compatibility API backed by the conservative planner and native renderer."""
    waveform = waveform.cpu()
    probabilities = mask.detach().cpu().reshape(-1)
    hop = sr // 100
    policy = CutConfig(enter_probability=CUTTING_PROBABILITY, exit_probability=CUTTING_PROBABILITY,
                       min_confident_ms=0, min_gap_ms=MIN_SEGMENT_DURATION_MS,
                       target_gap_ms=SILENCE_TARGET_DURATION * 1000, padding_ms=SAFETY_BUFFER_MS,
                       fade_ms=FADE_DURATION_MS, snap_ms=SEARCH_WINDOW_MS, use_boundaries=False)
    plan = plan_cuts(probabilities, sr, waveform.shape[-1], waveform.shape[0], hop / sr,
                     policy, waveform=waveform.numpy())
    full_mask = probabilities.repeat_interleave(hop)[:waveform.shape[-1]]
    return render_audio(waveform, plan), full_mask

def process_grid_aligned(model, transform, waveform, sr, hop_length, device, static_input_buffer):
    """Blend probabilities on an explicit grid, including short files and tails."""
    total_frames = math.ceil(waveform.shape[-1] / hop_length)
    chunk_frames = max(2, round(SEGMENT_LEN * sr / hop_length))
    stride_frames = max(1, chunk_frames // 2)
    accumulator = torch.zeros(1, total_frames)
    weights = torch.zeros(1, total_frames)
    window = torch.hann_window(chunk_frames, periodic=False).clamp_min(0.05)
    with torch.inference_mode():
        for first in range(0, total_frames, stride_frames):
            last = min(total_frames, first + chunk_frames)
            chunk = waveform[:, first * hop_length:min(waveform.shape[-1], last * hop_length)].to(device)
            chunk = torch.nn.functional.pad(chunk, (0, chunk_frames * hop_length - chunk.shape[-1]))
            prediction = _run_inference(model, transform, chunk, device, static_input_buffer).reshape(1, -1)
            count = last - first
            accumulator[:, first:last] += prediction[:, :count].cpu() * window[:count]
            weights[:, first:last] += window[:count]
    return accumulator / weights.clamp_min(1e-8)

def _run_inference(model, mel_transform, wav_chunk, device, input_buffer):
    features = legacy_features(wav_chunk, mel_transform)
    channel_masks = []
    for channel in features:
        frames = channel.shape[-1]
        if frames > input_buffer.shape[-1]:
            raise ValueError("Legacy input buffer is shorter than the feature grid")
        input_buffer[0, :, :, :frames].copy_(channel)
        # v3 emits logits. Every aggregation mode must operate on probabilities.
        probabilities = model(input_buffer[:, :, :, :frames]).sigmoid()
        if MASK_MODE == "Soft":
            prediction = probabilities.mean(dim=2)
        elif MASK_MODE == "Hybrid":
            prediction = 0.7 * probabilities.mean(dim=2) + 0.3 * probabilities.amax(dim=2)
        elif MASK_MODE == "PowerMean":
            prediction = probabilities.square().mean(dim=2).sqrt()
        elif MASK_MODE == "Hard":
            prediction = probabilities.amax(dim=2)
        else:
            raise ValueError(f"Unsupported MASK_MODE: {MASK_MODE}")
        channel_masks.append(prediction)
    # Preserve the original channels and let any protected channel veto removal.
    return torch.stack(channel_masks).amin(dim=0)

def processing():
    # Device setup
    if FORCE_CPU:
        device = torch.device("cpu")
        print("FORCE_CPU is True. Using CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False # For consistency we disable it.
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(IN_DIR, "*.wav")) + glob.glob(os.path.join(IN_DIR, "*.flac"))
    print(f"Found {len(files)} files.")

    loaded_models = {}

    # Loop
    for f_path in files:
        try:
            fname = os.path.basename(f_path)
            print(f"Processing: {fname}...")

            # Audio loading
            wav, sr = torchaudio.load(f_path)

            # Analysis retains every channel; preprocessing is shared with training.
            wav_for_inference = wav.clone()

            if STABILITY_NOISE:
                wav_for_inference = inject_stability_noise(wav_for_inference, sr, wav.device)

            # Dynamic model loading based on Sample Rate.
            current_hop = sr // 100
            if sr not in loaded_models:

                # Release previous model if switching SR
                if len(loaded_models) > 0:
                    print("Unloading previous model to free VRAM...")
                    loaded_models.clear()
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                if MODEL_VERSION == "v6":
                    model_path = os.path.join(CKPT_DIR, "v6_omi.pth")
                else:
                    model_path = os.path.join(CKPT_DIR, f"{MODEL_VERSION}_model_{sr}.pth")

                if MODEL_VERSION == "v3" and not os.path.exists(model_path):
                    candidates = [os.path.join(CKPT_DIR, f"v3_best_model_{sr}.pth"),
                                  os.path.join(CKPT_DIR, f"v3_latest_model_{sr}.pth")]
                    model_path = next((p for p in candidates if os.path.exists(p)), model_path)

                if not os.path.exists(model_path):
                    print(f"Skipping {fname}: No {MODEL_VERSION} model for {sr}Hz")
                    continue

                print(f"Loading {sr}Hz {MODEL_VERSION} model ...")

                if MODEL_VERSION == "v3":
                    model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(device)   # v3
                elif MODEL_VERSION == "v5":
                    from model_v5 import CGA_ResUNet
                    model = CGA_ResUNet(n_channels=2, n_classes=1).to(device)       # v5
                elif MODEL_VERSION == "v6":
                    from model_v6 import CGTA_ResUNet
                    model = CGTA_ResUNet(n_channels=2, n_classes=1).to(device)      # v6
                else:
                    print(f"'{MODEL_VERSION}' is not a valid model version choice. Exiting.")
                    sys.exit(1)

                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                if "sample_rate" in checkpoint and checkpoint["sample_rate"] != sr:
                    raise ValueError("Checkpoint sample rate differs from the audio")
                model.load_state_dict(checkpoint.get("state_dict", checkpoint))
                model.eval()

                # mel transform config
                if sr in [48000, 40000]:
                    N_FFT = 2048
                    N_MELS = 160
                else:
                    N_FFT = 1024 # for 32khz model variant.
                    N_MELS = 128

                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=current_hop, pad_mode="constant"
                ).to(device)

                # we're pre-allocating a static buffer for the model input
                # Shape: [Batch, Channels, Mel_Bins, Frames_per_60s_chunk]
                # Channels = 2 (Mel + Delta)
                dummy_frames = int(math.ceil((SEGMENT_LEN * sr) / current_hop)) + 5
                static_buffer = torch.zeros((1, 2, N_MELS, dummy_frames), device=device)

                loaded_models[sr] = (model, mel_transform, static_buffer)

            curr_model, curr_mel_transform, curr_buffer = loaded_models[sr]

            # Inference (GPU-accelerated, CPU accumulation)
            mel_mask = process_grid_aligned(curr_model, curr_mel_transform, wav_for_inference, sr, current_hop, device, curr_buffer)

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # SmartCutter on CPU.
            cleaned, binary_mask = SmartCutter(wav, mel_mask, sr=sr)

            # Debug section
            if DEBUG_MASK_PRED:
                if binary_mask.dim() > 2: binary_mask = binary_mask.squeeze(0)

                binary_mask_interpolated = torch.nn.functional.interpolate(
                    binary_mask.view(1,1,-1), size=wav.shape[1], mode='nearest'
                ).squeeze()

                debug_noise = torch.randn_like(wav) * 0.09
                debug_wav = wav + (debug_noise * (binary_mask_interpolated > CUTTING_PROBABILITY).float())
                torchaudio.save(os.path.join(OUT_DIR, "debug_" + os.path.basename(f_path)), debug_wav, sr)


            # Output path construction
            file_stem = os.path.splitext(fname)[0]
            if SAVE_EXTENSION == "flac":
                out_path = os.path.join(OUT_DIR, file_stem + ".flac")
            elif "wave" in SAVE_EXTENSION:
                out_path = os.path.join(OUT_DIR, file_stem + ".wav")

            # Saving
            if SAVE_EXTENSION == "flac":
                torchaudio.save(out_path, cleaned, sr, format="flac", backend="soundfile")
            elif SAVE_EXTENSION == "wave_16":
                torchaudio.save(out_path, cleaned, sr, encoding="PCM_S", bits_per_sample=16)
            elif SAVE_EXTENSION == "wave_32float":
                torchaudio.save(out_path, cleaned, sr, encoding="PCM_F", bits_per_sample=32)
            else:
                print(f"Specified saving extension: '{SAVE_EXTENSION}' is unsupported. Exiting.")
                sys.exit(1)

            print(f"Saved: {out_path}")

            # Cleanup

            # inputs and outputs
            if 'wav' in locals(): del wav
            if 'cleaned' in locals(): del cleaned
            if 'wav_for_inference' in locals(): del wav_for_inference

            # masks and intermediate tensors
            if 'binary_mask' in locals(): del binary_mask
            if 'mel_mask' in locals(): del mel_mask

            # Delete debug pieces
            if 'debug_wav' in locals(): del debug_wav
            if 'debug_noise' in locals(): del debug_noise
            if 'binary_mask_interpolated' in locals(): del binary_mask_interpolated

            # Current buffer ( GPU )
            if 'curr_buffer' in locals(): curr_buffer.zero_()

            gc.collect()

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            del e
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    processing()
