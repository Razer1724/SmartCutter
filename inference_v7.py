"""
SmartCutter v7 inference - ONE model for ANY sample rate.

The detector runs at 32 kHz internally; the original audio is only used for the actual cutting, at
its own sample rate, so there is no per-sample-rate checkpoint any more. The checkpoint carries its
own config, tuned threshold and edge-head setting (chosen on your validation set during training).
"""
import gc
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

from cutter import CutConfig, smart_cut
from features_v7 import (FPS, INTERNAL_SR, PEAK_TARGET, SEG_FRAMES, SEG_SAMPLES, SEGMENT_SEC,
                         FeatureExtractor, fit_length, resample)
from model_v7 import SmartCutterV7

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
#  INFER AND PROCESSING CONFIG
FORCE_CPU = False                       #   By default runs with GPU acceleration (CUDA)
DEBUG_MASK_PRED = True                 #   Also write debug_<file>: noise laid over exactly the regions that were cut
SAVE_EXTENSION = "flac"                 #   Available: "flac" (24-bit), "wave_16" and "wave_32float"
INFER_BATCH = 4                         #   8 s chunks per forward pass (lower if you run out of VRAM)

#  SMART CUTTER CONFIG ( Safe defaults. )
SILENCE_TARGET_DURATION = 0.100         #   Target duration for silence gaps (e.g. 500 ms gap -> 100 ms)
MIN_SEGMENT_DURATION_MS = 100           #   Minimum length for a detected silence to be cut
ENABLE_BRIDGING = True                  #   Fill small gaps in the mask (same as v3)
CUTTING_PROBABILITY = None              #   None = threshold tuned on your validation set during training; or e.g. 0.5
USE_EDGE_REFINEMENT = None              #   None = whatever validation preferred; True / False to force

#  PATHS
IN_DIR = "infer_input"
OUT_DIR = "infer_output"
MODEL_PATH = os.path.join("ckpts", "v7_model.pth")

# Established safe params
SEARCH_WINDOW_MS = 25
FADE_DURATION_MS = 10
SAFETY_BUFFER_MS = 5
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


class Detector:
    def __init__(self, model_path=MODEL_PATH, force_cpu=FORCE_CPU):
        self.device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
        if ck.get("format") != "smartcutter_v7":
            raise RuntimeError(f"{model_path} is not a v7 checkpoint.")
        if ck.get("internal_sr") != INTERNAL_SR:
            raise RuntimeError(f"Checkpoint was trained for {ck.get('internal_sr')} Hz internal rate, code uses {INTERNAL_SR}.")
        self.model = SmartCutterV7(ck["config"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.fx = FeatureExtractor().to(self.device)
        self.threshold = float(ck["threshold"])
        self.edge_refine = bool(ck["edge_refine"])
        self.metrics = ck.get("metrics", {})
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    def cut_config(self):
        return CutConfig(
            threshold=self.threshold if CUTTING_PROBABILITY is None else float(CUTTING_PROBABILITY),
            min_segment_ms=MIN_SEGMENT_DURATION_MS,
            safety_buffer_ms=SAFETY_BUFFER_MS,
            search_window_ms=SEARCH_WINDOW_MS,
            fade_ms=FADE_DURATION_MS,
            target_silence_s=SILENCE_TARGET_DURATION,
            bridging=ENABLE_BRIDGING,
            edge_refine=self.edge_refine if USE_EDGE_REFINEMENT is None else bool(USE_EDGE_REFINEMENT),
        )

    @torch.no_grad()
    def track(self, wav, sr):
        """
        wav: 1-D float tensor at ANY sample rate. Returns (probs [T], edge [2,T]) numpy at 100 frames/s,
        frame t centred on t/100 seconds. Processes 8 s chunks with 50 % overlap and Hann-weighted blending.
        """
        n = wav.shape[0]
        peak = wav.abs().max().item()
        scale = PEAK_TARGET / peak if peak > 0 else 1.0          # file-level peak norm, identical to training

        chunk = int(round(SEGMENT_SEC * sr))
        stride = int(round(SEGMENT_SEC / 2 * sr))
        frames_per_stride = int(round(SEGMENT_SEC / 2 * FPS))
        starts, s = [], 0
        while True:
            starts.append(s)
            if s + chunk >= n:
                break
            s += stride

        t_total = (n * FPS) // int(sr) + 1
        acc_p = np.zeros(t_total + SEG_FRAMES, np.float64)
        acc_e = np.zeros((2, t_total + SEG_FRAMES), np.float64)
        acc_w = np.zeros(t_total + SEG_FRAMES, np.float64)
        weight = 0.05 + 0.95 * np.hanning(SEG_FRAMES)             # floor > 0 so file-edge frames keep a value

        for b0 in range(0, len(starts), INFER_BATCH):
            group = starts[b0:b0 + INFER_BATCH]
            batch = []
            for st in group:
                x = wav[st:st + chunk].to(self.device) * scale
                if x.shape[0] < chunk:
                    x = torch.nn.functional.pad(x, (0, chunk - x.shape[0]))
                batch.append(fit_length(resample(x.unsqueeze(0), sr, INTERNAL_SR), SEG_SAMPLES).squeeze(0))
            x = torch.stack(batch)
            spec, stats = self.fx(x)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(spec, stats)
            p = torch.sigmoid(out["sil"].float()).cpu().numpy()
            e = torch.sigmoid(out["edge"].float()).cpu().numpy()
            for i, st in enumerate(group):
                f0 = (st // stride) * frames_per_stride           # exact: start frame of this chunk
                acc_p[f0:f0 + SEG_FRAMES] += p[i] * weight
                acc_e[:, f0:f0 + SEG_FRAMES] += e[i] * weight
                acc_w[f0:f0 + SEG_FRAMES] += weight

        acc_w = np.maximum(acc_w, 1e-9)
        return (acc_p / acc_w)[:t_total].astype(np.float32), (acc_e / acc_w)[:, :t_total].astype(np.float32)


def load_audio_mono(path):
    """Returns ([1,N] float tensor, sr). Stereo -> channel with the lowest DC offset (same rule as v3)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(x.T))
    if wav.shape[0] > 1:
        dc = wav.mean(dim=1).abs()
        ch = int(torch.argmin(dc))
        wav = wav[ch].unsqueeze(0)
        print(f"    -> Converted Stereo to Mono (Selected Ch {ch}, DC: {dc[ch]:.6f})")
    return wav, int(sr)


def save_audio(path_no_ext, wav, sr):
    data = wav.squeeze(0).numpy() if wav.shape[0] == 1 else wav.t().numpy()
    if SAVE_EXTENSION == "flac":
        out = path_no_ext + ".flac"
        sf.write(out, data, sr, format="FLAC", subtype="PCM_24")
    elif SAVE_EXTENSION == "wave_16":
        out = path_no_ext + ".wav"
        sf.write(out, data, sr, format="WAV", subtype="PCM_16")
    elif SAVE_EXTENSION == "wave_32float":
        out = path_no_ext + ".wav"
        sf.write(out, data, sr, format="WAV", subtype="FLOAT")
    else:
        print(f"Specified saving extension: '{SAVE_EXTENSION}' is unsupported. Exiting.")
        sys.exit(1)
    return out


def main():
    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found. Train first (run_train_v7.bat) or copy a v7 checkpoint there.")
        return
    det = Detector()
    cfg = det.cut_config()
    print(f"[v7] device {det.device} | threshold {cfg.threshold:.2f} | edge refinement {'ON' if cfg.edge_refine else 'off'}"
          f" | any sample rate (detector runs at {INTERNAL_SR} Hz)")

    files = []
    for ext in ("wav", "flac", "mp3", "ogg"):
        files += glob.glob(os.path.join(IN_DIR, f"*.{ext}"))
    files = sorted(files)
    if not files:
        print(f"No audio files in '{IN_DIR}'.")
        return

    for f_path in files:
        fname = os.path.basename(f_path)
        print(f"Processing {fname} ...")
        try:
            wav, sr = load_audio_mono(f_path)
            probs, edge = det.track(wav[0], sr)
            cleaned, starts, ends = smart_cut(wav, probs, sr, cfg, edge)
            print(f"    -> {sr} Hz, {len(starts)} cuts, {wav.shape[1]/sr:.2f}s -> {cleaned.shape[1]/sr:.2f}s")

            if DEBUG_MASK_PRED:
                region = torch.zeros(wav.shape[1])
                for s, e in zip(starts.tolist(), ends.tolist()):
                    region[s:e] = 1.0
                dbg = wav + torch.randn_like(wav) * 0.09 * region.unsqueeze(0)
                sf.write(os.path.join(OUT_DIR, "debug_" + os.path.splitext(fname)[0] + ".wav"), dbg.squeeze(0).numpy(), sr, subtype="PCM_16")

            peak = cleaned.abs().max()
            if peak >= 0.95:
                cleaned = cleaned * (0.95 / peak.item())

            out = save_audio(os.path.join(OUT_DIR, os.path.splitext(fname)[0]), cleaned, sr)
            print(f"Saved: {out}")
        except Exception as ex:                                  # keep batch processing alive
            print(f"[ERROR] {fname}: {ex}")
        finally:
            gc.collect()
            if det.device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
