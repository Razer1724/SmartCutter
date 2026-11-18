import os
import sys
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from model_v3 import DSCA_ResUNet_v3
from smartcutter.data import paired_files
from smartcutter.frontend import legacy_features

# Config
INTENDED_SR_FOR_TRAIN = 48000  # Options: 32000, 40000, 48000
SAMPLE_RATE = INTENDED_SR_FOR_TRAIN
HOP_LENGTH = SAMPLE_RATE // 100

SEGMENT_LEN = SAMPLE_RATE * 8
TRAIN_DIR = "training_files"
VAL_DIR = "validation_files"
CKPT_DIR = "ckpts"

# Hparams
BATCH_SIZE = 8
ACCUMULATION_STEPS = 1
MAX_EPOCHS = 1000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 25
VALIDATE_INTERVAL = 1

# Augmentation
DYNAMIC_AUGMENTATION = True

RANDOM_GAIN_AUG = True
rand_gain_max = 1.0
rand_gain_min = 0.8

DECOY_AUG = True
NOISE_AUG_ON_SILENCE = True
STRICT_MASK = True

# Validation configuration
AUTO_SYNC_VAL_PERCENTAGE = True       
VALIDATION_SET_PERCENTAGE = 0.20      # Use 0.20 (20%) for your 1-hour dataset

# Debug
DEBUG_TRAINING_MASK = False


if SAMPLE_RATE in [48000, 40000]:
    N_FFT = 2048
    N_MELS = 160
else: # 32khz
    N_FFT = 1024
    N_MELS = 128

def save_audio_mask_debug(raw_wav, mask, filename="mask_check.wav"):
    """
    Overlays white noise on the raw audio where the mask is active.
    """
    noise = torch.randn_like(raw_wav) * 0.2

    # Where mask is 1, use noise. Where mask is 0, use original audio.
    # Result = (Audio * (1 - Mask)) + (Noise * Mask)
    debug_audio = (raw_wav * (1 - mask)) + (noise * mask)

    # Normalize to prevent clipping
    if debug_audio.abs().max() > 0:
        debug_audio = debug_audio / debug_audio.abs().max()

    torchaudio.save(filename, debug_audio.cpu(), SAMPLE_RATE)
    print(f"[DEBUG] Saved audio mask check to: {filename}")

class ValidationGatedWeight:
    """
    Automated Deep Supervision Balancer.
    Uses a Logistic ( Sigmoid ) Sunset to keep auxiliary heads active until 
    the model reaches a competence threshold, then executes a clean handover.
    """
    def __init__(self, initial_w2=0.3, initial_w3=0.2, alpha=0.9, center=0.65, steepness=15):
        self.w_aux2 = initial_w2
        self.w_aux3 = initial_w3
        self.ema_dice = 0.0
        self.alpha = alpha

        # hyperparameters for the sunset curve
        self.center = center
        self.steepness = steepness

    def update(self, val_dice_mean):
        # update EMA for stability ( Prevents jitter from one bad val. batch )
        self.ema_dice = (self.alpha * self.ema_dice) + ((1 - self.alpha) * val_dice_mean)

        # logistic Sunset Function
        # logic: gate is ~1.0 when Dice < center, and ~0.0 when Dice > center.
        # we use np.clip to prevent overflow in exp if dice is extremely low.
        diff = self.steepness * (self.ema_dice - self.center)
        gate = 1.0 / (1.0 + np.exp(np.clip(diff, -20, 20)))

        # asymmetric Decay
        # aux3 (Deepest/Bottleneck) is foundational; we keep it longer.
        # aux2 (Mid-level) is for refinement; we let it sunset slightly faster.
        self.w_aux3 = 0.2 * gate
        self.w_aux2 = 0.3 * (gate ** 1.5) # Quadratic-adjusted gate for faster exit

        # once influence is negligible, we kill it to clean up the gradients.
        if self.w_aux2 < 0.005: self.w_aux2 = 0.0
        if self.w_aux3 < 0.005: self.w_aux3 = 0.0

        return self.w_aux2, self.w_aux3

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs.reshape(-1))
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class SlicedDataset(Dataset):
    def __init__(self, file_pairs, segment_len=SEGMENT_LEN, overlap=0.20, augment=False):
        self.samples = []
        self.augment = augment

        # Mel Spectrogram Config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, 
            n_mels=N_MELS, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, pad_mode="constant"
        )
        print(f"[DATASET] Processing {len(file_pairs)} file pairs...")

        for r_file, m_file in file_pairs:
            # Load
            raw_wav, sr_r = torchaudio.load(r_file)
            marked_wav, sr_m = torchaudio.load(m_file)

            if sr_r != sr_m or raw_wav.shape != marked_wav.shape:
                raise ValueError(f"Misaligned raw/marked pair: {r_file}")
            if not torch.isfinite(raw_wav).all() or not torch.isfinite(marked_wav).all():
                raise ValueError(f"Non-finite samples: {r_file}")
            # Derive labels before resampling or gain changes can create differences.
            diff = torch.abs(marked_wav - raw_wav).amax(dim=0, keepdim=True)
            mask = (diff > (1e-5 if STRICT_MASK else 0.05)).float()
            if sr_r != SAMPLE_RATE:
                raw_wav = torchaudio.functional.resample(raw_wav, sr_r, SAMPLE_RATE)
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=raw_wav.shape[-1], mode="nearest")[0]

            if DEBUG_TRAINING_MASK:
                debug_filename = f"DEBUG_{os.path.basename(r_file)}"
                save_audio_mask_debug(raw_wav, mask, debug_filename)
                sys.exit(f"Check {debug_filename} to verify the threshold/masking quality.")


            # Slicing
            stride = int(segment_len * (1 - overlap))
            length = raw_wav.shape[1]

            for i in range(0, length - segment_len + 1, stride):
                r_slice = raw_wav[:, i : i + segment_len]
                m_slice = mask[:, i : i + segment_len]

                # meaningless silence filtering
                has_audio = r_slice.abs().max() > 0.001 # audio content presence check
                has_error = m_slice.max() > 0.5 # mask presence check

                if has_audio or has_error: 
                    self.samples.append((r_slice, m_slice))

        print(f"[DATASET] Loaded {len(self.samples)} segments.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_crop, mask_crop = self.samples[idx]
        raw_crop = raw_crop.clone()
        mask_crop = mask_crop.clone()

        # --------------------   Dynamic Augmentation
        if self.augment and DYNAMIC_AUGMENTATION:

            if RANDOM_GAIN_AUG:
                # Random Gain (Applied to input only)
                if torch.rand(1).item() < 0.5:
                    if raw_crop.abs().max() > 1e-5:
                        gain = torch.empty(1).uniform_(rand_gain_min, rand_gain_max)
                        raw_crop = raw_crop * gain

            if DECOY_AUG:
                # Decoy (Random gentle noise injection in non-mask areas to prevent false positives)
                if torch.rand(1).item() < 0.15: # 15% chance
                     # Create low level noise
                    decoy = torch.randn_like(raw_crop) * torch.empty(1).uniform_(0.0001, 0.001) # Range is 0.0001 (-80dB) to 0.001 (-60dB)
                    # Decoy only where there is no mask
                    raw_crop = raw_crop + (decoy * (1 - mask_crop))

            if NOISE_AUG_ON_SILENCE:
                # Noise aug on pure silences ( 0s ) to bridge gaps and improve robustness.
                if torch.rand(1).item() < 0.4: # 40% chance of being applied
                    # -80dB to -60dB
                    stab_noise = torch.randn_like(raw_crop) * torch.empty(1).uniform_(0.0001, 0.001)
                    raw_crop = raw_crop + (stab_noise * mask_crop)


        # This exact feature function is also used by v3 inference.
        combined_input = legacy_features(raw_crop, self.mel_transform).amax(dim=0)
        frames = combined_input.shape[-1]
        padded = torch.nn.functional.pad(mask_crop, (0, frames * HOP_LENGTH - mask_crop.shape[-1]), value=1)
        # A frame is removable only if every annotated sample in it is removable.
        mask_final = padded.reshape(1, frames, HOP_LENGTH).amin(dim=-1).squeeze(0)

        return combined_input, mask_final


def get_file_pairs(folder):
    return paired_files(folder) if os.path.isdir(folder) else []

def validate(model, loader, device, bce_crit, dice_crit):
    model.eval()
    batch_dice_scores = []
    total_loss = 0

    THRESHOLD = 0.5

    with torch.no_grad():
        for mels, masks in loader:
            mels, masks = mels.to(device), masks.to(device)

            # Expand the masks
            masks_expanded = masks.view(masks.shape[0], 1, 1, -1).expand(-1, -1, N_MELS, -1) # (Batch, Time) -> (Batch, 1, N_MELS, Time)

            logits = model(mels)

            # Loss calculation ( using logits )
            loss = bce_crit(logits, masks_expanded) + dice_crit(logits, masks_expanded)
            total_loss += loss.item()

            # Hard dice score for metrics
            preds_prob = torch.sigmoid(logits)
            preds_hard = (preds_prob > THRESHOLD).float()

            # Calculate Dice on the binary mask
            intersection = (preds_hard * masks_expanded).sum()
            union = preds_hard.sum() + masks_expanded.sum()

            hard_dice = (2. * intersection + 1e-6) / (union + 1e-6)
            batch_dice_scores.append(hard_dice.item())

    # Calculate Statistics
    mean_dice = np.mean(batch_dice_scores)
    std_dice = np.std(batch_dice_scores)
    avg_loss = total_loss / len(loader)

    # Conservative Score: "Safe Lower Bound"
    gen_score = mean_dice - std_dice

    return avg_loss, gen_score, mean_dice, std_dice

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(CKPT_DIR): os.makedirs(CKPT_DIR)
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    if not os.path.exists(VAL_DIR): os.makedirs(VAL_DIR)

    print(f"[DEVICE] {device}")

    # Load training data
    train_pairs = get_file_pairs(TRAIN_DIR)
    if not train_pairs:
        print(f"[ERROR] No files in {TRAIN_DIR}. Please add *_raw and *_marked files.")
        return

    train_ds = SlicedDataset(train_pairs, overlap=0.50, augment=True)

    # Load validation data
    val_pairs = get_file_pairs(VAL_DIR)

    if AUTO_SYNC_VAL_PERCENTAGE and val_pairs:
        target_total = int(len(train_ds) * VALIDATION_SET_PERCENTAGE)
        segments_per_speaker = max(1, target_total // len(val_pairs))

        final_val_segments = []

        for r_file, m_file in val_pairs:
            speaker_ds = SlicedDataset([(r_file, m_file)], overlap=0.0, augment=False)

            if len(speaker_ds) == 0: continue
            take_count = min(len(speaker_ds), segments_per_speaker)

            generator = torch.Generator().manual_seed(42)

            speaker_subset, _ = torch.utils.data.random_split(
                speaker_ds, 
                [take_count, len(speaker_ds) - take_count],
                generator=generator
            )
            final_val_segments.append(speaker_subset)

        if len(final_val_segments) > 0:
            val_ds = ConcatDataset(final_val_segments)
            print(f"[ValidationHandler] Final Sync -> Train: {len(train_ds)} | Val: {len(val_ds)}")
        else:
            print("[WARNING] SmartVal found no valid segments! Falling back to full validation files.")
            val_ds = SlicedDataset(val_pairs, overlap=0.0, augment=False)
    else:
        val_ds = SlicedDataset(val_pairs, overlap=0.0)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("v3 training requires nonempty training and validation segments (at least 8 seconds)")

    # Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Model setup
    model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(device)

    # optimizer init
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)

    # scheduler init
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=LR_PATIENCE
    )

    # ValidationGatedWeight
    aux_gate = ValidationGatedWeight(initial_w2=0.3, initial_w3=0.2)
    w_aux2, w_aux3 = 0.3, 0.2

    # Binary Cross Entropy + Dice for better boundary precision
    bce_crit = nn.BCEWithLogitsLoss()
    dice_crit = DiceLoss()

    # Trackers
    best_val_score = -1.0
    early_stop_counter = 0
    best_run_info = {"epoch": 0, "score": -1.0}

    # Loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []

        # Zero grad
        optimizer.zero_grad(set_to_none=True)

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch: {epoch+1} [Train]")

        for i, (mels, masks) in enumerate(pbar):
            mels, masks = mels.to(device), masks.to(device)

            # Expand the masks
            masks_expanded = masks.view(masks.shape[0], 1, 1, -1).expand(-1, -1, N_MELS, -1) # (Batch, Time) -> (Batch, 1, N_MEL, Time)
            # Forward
            outputs = model(mels) # Get model preds ( dict with logits )


            # Calculate losses
            l_main = bce_crit(outputs["main"], masks_expanded) + dice_crit(outputs["main"], masks_expanded)
            loss = l_main
            # Aux 2 ( Mid-level details )
            if w_aux2 > 0:
                l_aux2 = bce_crit(outputs["aux2"], masks_expanded) + dice_crit(outputs["aux2"], masks_expanded)
                loss += (w_aux2 * l_aux2)
            # Aux 3 ( Coarse/Deep details )
            if w_aux3 > 0:
                l_aux3 = bce_crit(outputs["aux3"], masks_expanded) + dice_crit(outputs["aux3"], masks_expanded)
                loss += (w_aux3 * l_aux3)

            # Backward
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Step
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * ACCUMULATION_STEPS
            train_losses.append(actual_loss)
            pbar.set_postfix({
                'loss': f"{actual_loss:.4f}", 
                'ax2': f"{w_aux2:.3f}",
                'ax3': f"{w_aux3:.3f}"
            })

        # Metrics
        avg_train_loss = np.mean(train_losses)
        train_stability = 1.0 / (1.0 + np.std(train_losses))
        curr_lr = optimizer.param_groups[0]['lr']

        # Display
        print(f"\n      ->  TRAIN AVG LOSS: {avg_train_loss:.4f} | TRAIN Stability: {train_stability:.2%} | LR: {curr_lr:.2e}")


        # Validation & Scheduler
        if val_loader and (epoch + 1) % VALIDATE_INTERVAL == 0:
            val_loss, val_score, val_mean_dice, val_std_dice = validate(model, val_loader, device, bce_crit, dice_crit)

            w_aux2, w_aux3 = aux_gate.update(val_mean_dice)
            print(f"      ->  VAL AVG LOSS: {val_loss:.4f} | Conservative Score: {val_score:.4f} | Mean: {val_mean_dice:.4f} | Std: {val_std_dice:.4f}")
            print(f"\n      ->  Aux Weights for the next epoch: w2={w_aux2:.3f}, w3={w_aux3:.3f}")

            # schedule according to scoring
            scheduler.step(val_score)

            # Save Best Model
            if val_score > best_val_score:
                best_val_score = val_score
                early_stop_counter = 0

                # capture current best stats
                best_run_info = {
                    "epoch": epoch + 1,
                    "score": val_score,
                    "loss": val_loss,
                    "dice_mean": val_mean_dice,
                    "dice_std": val_std_dice,
                    "lr": curr_lr
                }

                # save best performing model
                torch.save({"architecture": "v3", "format_version": 1, "sample_rate": SAMPLE_RATE,
                            "frontend": "peak-cap-v1", "state_dict": model.state_dict()},
                           os.path.join(CKPT_DIR, f"v3_best_model_{SAMPLE_RATE}.pth"))
                print(f"\n  ----->  New Record! Score: {val_score:.4f}  <-----")
            else:
                early_stop_counter += 1
                print(f"\n      ->  No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

            # early stopping
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("\n[!!!] Early Stopping Triggered.")
                break

        # most recent model saving
        torch.save({"architecture": "v3", "format_version": 1, "sample_rate": SAMPLE_RATE,
                    "frontend": "peak-cap-v1", "state_dict": model.state_dict()},
                   os.path.join(CKPT_DIR, f"v3_latest_model_{SAMPLE_RATE}.pth"))

    # print the summary
    print("\n" + "="*50)
    print("           TRAINING SUMMARY")
    print("="*50)

    if best_run_info["epoch"] == 0:
        print(" [!] No best model was saved.")
    else:
        print(f" Best Model Found At Epoch:   {best_run_info['epoch']}")
        print("-" * 50)
        print(f" Conservative Score:          {best_run_info['score']:.4f}")
        print(f" Raw Mean Dice:               {best_run_info['dice_mean']:.4f} ( Avg Overlap )")
        print(f" Stability (Std Dev):         {best_run_info['dice_std']:.4f}  ( Lower is stable )")
        print(f" Validation Loss:             {best_run_info['loss']:.4f}")
        print(f" Final LR:                    {best_run_info['lr']:.2e}")
        print(f" Batch Size:                  {BATCH_SIZE}")
        print(f" Accumulation Steps:          {ACCUMULATION_STEPS}")

    print("="*50 + "\n")



if __name__ == "__main__":
    train()
