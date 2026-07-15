import os
import sys
import glob
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from model_v6 import CGTA_ResUNet

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
ENABLE_BRIDGING = True

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
            hop_length=HOP_LENGTH
        )
        print(f"[DATASET] Processing {len(file_pairs)} file pairs...")

        for r_file, m_file in file_pairs:
            # Load
            raw_wav, sr_r = torchaudio.load(r_file)
            marked_wav, sr_m = torchaudio.load(m_file)

            # Resample if needed
            if sr_r != SAMPLE_RATE: raw_wav = torchaudio.transforms.Resample(sr_r, SAMPLE_RATE)(raw_wav)
            if sr_m != SAMPLE_RATE: marked_wav = torchaudio.transforms.Resample(sr_m, SAMPLE_RATE)(marked_wav)

            # Match lengths
            min_len = min(raw_wav.shape[1], marked_wav.shape[1])
            raw_wav, marked_wav = raw_wav[:, :min_len], marked_wav[:, :min_len]

            # Normalize
            max_val = torch.max(torch.abs(marked_wav))
            if max_val > 0.95:
                scale = 0.95 / max_val
                marked_wav *= scale
                raw_wav *= scale

            # Get the absolute difference
            diff = torch.abs(marked_wav - raw_wav)

            # Binary threshold
            if STRICT_MASK:
                mask = (diff > 1e-5).float() # 1e-7
            else:
                mask = (diff > 0.05).float()

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


        # --------------------   Processing
        # Prevent any potential clipping before stft
        raw_crop = torch.clamp(raw_crop, -1.0, 1.0)
        # Transform to mel
        mel = self.mel_transform(raw_crop).squeeze(0) 

        # log normalization
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        min_db = -80.0
        max_db = 0.0
        mel = torch.clamp(mel, min=min_db, max=max_db)
        mel = (mel - min_db) / (max_db - min_db) # Scaled 0.0 to 1.0

        # Deltas
        delta = F_audio.compute_deltas(mel.unsqueeze(0)).squeeze(0)
        combined_input = torch.stack([mel, delta], dim=0)

        # Resize mask to match Mel dimensions
        mask_final = torch.nn.functional.interpolate(
            mask_crop.unsqueeze(0).unsqueeze(0),
            size=(1, combined_input.shape[2]), # (1, Time)
            mode='nearest'
        ).squeeze(0).squeeze(0)

        if ENABLE_BRIDGING:
            # Gap bridging
            bridge_frames = 5 # At 100fps, 50ms is ~5 frames
            # Dilation
            mask_final = torch.nn.functional.max_pool1d(mask_final.view(1,1,-1), bridge_frames, 1, bridge_frames//2)
            # Erosion
            mask_final = -torch.nn.functional.max_pool1d(-mask_final, bridge_frames, 1, bridge_frames//2).squeeze()

        return combined_input, mask_final


def get_file_pairs(folder):
    if not os.path.exists(folder):
        return []

    raw_files = sorted(glob.glob(os.path.join(folder, "*_raw.wav")) + glob.glob(os.path.join(folder, "*_raw.flac")))
    marked_files = sorted(glob.glob(os.path.join(folder, "*_marked.wav")) + glob.glob(os.path.join(folder, "*_marked.flac")))

    # Check to ensure pairs match up
    if len(raw_files) != len(marked_files):
        print(f"[WARNING] Mismatch in {folder}: {len(raw_files)} raw vs {len(marked_files)} marked.")

    return list(zip(raw_files, marked_files))

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
    model = CGTA_ResUNet(n_channels=2, n_classes=1).to(device)

    # optimizer init
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)

    # scheduler init
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=LR_PATIENCE
    )

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
            logits = model(mels) # Get model preds ( logits )


            # Calculate losses
            loss = bce_crit(logits, masks_expanded) + dice_crit(logits, masks_expanded)

            # Backward
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Step
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * ACCUMULATION_STEPS
            train_losses.append(actual_loss)
            pbar.set_postfix({'loss': f"{actual_loss:.4f}"})
        # Metrics
        avg_train_loss = np.mean(train_losses)
        train_stability = 1.0 / (1.0 + np.std(train_losses))
        curr_lr = optimizer.param_groups[0]['lr']

        # Display
        print(f"\n      ->  TRAIN AVG LOSS: {avg_train_loss:.4f} | TRAIN Stability: {train_stability:.2%} | LR: {curr_lr:.2e}")


        # Validation & Scheduler
        if val_loader and (epoch + 1) % VALIDATE_INTERVAL == 0:
            val_loss, val_score, val_mean_dice, val_std_dice = validate(model, val_loader, device, bce_crit, dice_crit)

            print(f"      ->  VAL AVG LOSS: {val_loss:.4f} | Conservative Score: {val_score:.4f} | Mean: {val_mean_dice:.4f} | Std: {val_std_dice:.4f}")

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
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "v6_best_model_universal.pth"))
                print(f"\n  ----->  New Record! Score: {val_score:.4f}  <-----")
            else:
                early_stop_counter += 1
                print(f"\n      ->  No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

            # early stopping
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("\n[!!!] Early Stopping Triggered.")
                break

        # most recent model saving
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "v6_latest_model_universal.pth"))

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