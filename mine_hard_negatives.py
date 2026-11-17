"""
Mine hard negatives for v7.

Put audio that contains NOTHING that should be cut (continuous speech with the natural pauses, breaths and
quiet tails you want to keep) into  hard_negative_candidates/ . This script runs the current v7 model over it;
every place the model WOULD cut is a false cut, and a crop around it is saved into  hard_negatives/ .
The next training run oversamples those crops with an all-zero label ("never cut here"), which is the most
direct way to remove a specific false-cut habit.

Run it, retrain (train_v7.py resumes/rebuilds the cache automatically), repeat.
"""
import glob
import os

import soundfile as sf

from cutter import find_cuts
from inference_v7 import Detector, MODEL_PATH, load_audio_mono

CANDIDATE_DIR = "hard_negative_candidates"
OUT_DIR = "hard_negatives"
CONTEXT_S = 1.5          # audio kept on each side of a false cut
MIN_CROP_S = 4.0
MAX_PER_FILE = 30


def main():
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found - train a first model before mining.")
        return
    det = Detector()
    cfg = det.cut_config()

    files = []
    for ext in ("wav", "flac", "mp3", "ogg"):
        files += glob.glob(os.path.join(CANDIDATE_DIR, f"*.{ext}"))
    if not files:
        print(f"No audio in '{CANDIDATE_DIR}'.")
        return

    total = 0
    for f_path in sorted(files):
        stem = os.path.splitext(os.path.basename(f_path))[0]
        wav, sr = load_audio_mono(f_path)
        probs, edge = det.track(wav[0], sr)
        starts, ends = find_cuts(probs, wav, sr, cfg, edge)
        print(f"{stem}: {len(starts)} false cuts")
        if len(starts) == 0:
            continue

        # longest first (most damaging), then merge overlapping crops
        order = sorted(range(len(starts)), key=lambda i: -(ends[i] - starts[i]))[:MAX_PER_FILE]
        crops = []
        for i in sorted(order, key=lambda i: starts[i]):
            ctx = max(CONTEXT_S, (MIN_CROP_S - (ends[i] - starts[i]) / sr) / 2)
            a, b = max(0, int(starts[i] - ctx * sr)), min(wav.shape[1], int(ends[i] + ctx * sr))
            if crops and a <= crops[-1][1]:
                crops[-1][1] = max(crops[-1][1], b)
            else:
                crops.append([a, b])

        file_peak = wav.abs().max().item()
        g = min(1.0, 0.999 / max(file_peak, 1e-9))
        for k, (a, b) in enumerate(crops):
            base = os.path.join(OUT_DIR, f"{stem}_fp{k:02d}")
            sf.write(base + ".flac", (wav[0, a:b] * g).numpy(), sr, format="FLAC", subtype="PCM_24")
            with open(base + ".peak.txt", "w") as fh:         # keeps training's peak-normalisation identical to inference
                fh.write(f"{file_peak * g:.8f}")
            total += 1
    print(f"\nSaved {total} hard-negative crops to '{OUT_DIR}'. Retrain to learn from them.")


if __name__ == "__main__":
    main()
