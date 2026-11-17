"""
SmartCutter v7 - data layer.

Storage
-------
Training pairs (*_raw / *_marked) are converted ONCE into a cache folder:
    cache_v7/<id>.flac      raw audio, mono, NATIVE sample rate, 24-bit FLAC (lossless)
    cache_v7/<id>.npy       silence intervals [[start, end], ...] in native samples
    cache_v7/index.json     what is in the cache + a signature of the source files
Audio is then read straight from FLAC with random access (a 30 min file seeks in ~4 ms), so memory
use is tiny and DataLoader workers on Windows no longer each copy gigabytes of float32 tensors.
(v3 held every 8 s segment as float32 audio + float32 mask in RAM: ~3 MB per segment.)

Labels are derived analytically from the sample-accurate intervals: frame t is "silence" if the
time t/100 s lies inside an interval. Nothing is stored at sample resolution.

Hard negatives
--------------
Any audio you put in `hard_negatives/` is trained on with an all-zero mask ("never cut anything in
here"). mine_hard_negatives.py can fill this folder automatically from audio you know has no
cuttable silence.
"""
import glob
import json
import math
import os
import random
import re

import numpy as np
import soundfile as sf
import torch
from scipy.signal import lfilter
from torch.utils.data import Dataset

from cutter import close_probs
from features_v7 import (FPS, INTERNAL_SR, PEAK_TARGET, SEG_FRAMES, SEG_SAMPLES, SEGMENT_SEC,
                         fit_length, resample)

CACHE_VERSION = 1
BRIDGE_GAP_MS = 40.0       # gaps this short between silences are merged (what mask-bridging does)
MIN_CUT_MS = 100.0         # silences shorter than this are never cut -> not an edge target / metric target
EDGE_SIGMA = 1.5           # gaussian width (frames) of the start/end edge targets
LOW_SRS = (8000, 11025, 16000, 22050, 24000)   # band-limit augmentation ("pretend source rate")


# ============================================================================= interval helpers
def mask_to_intervals(mask):
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return np.stack([s, e], axis=1).astype(np.int64) if len(s) else np.zeros((0, 2), np.int64)


def merge_intervals_sec(iv, sr, gap_ms=BRIDGE_GAP_MS, min_ms=MIN_CUT_MS):
    """Sample intervals -> seconds, merge tiny gaps, drop too-short ones. Returns float [K,2]."""
    if len(iv) == 0:
        return np.zeros((0, 2))
    iv = iv.astype(np.float64) / sr
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a - out[-1][1] <= gap_ms / 1000.0:
            out[-1][1] = b
        else:
            out.append([a, b])
    out = np.asarray(out)
    return out[(out[:, 1] - out[:, 0]) >= min_ms / 1000.0]


def frame_labels(iv, sr, start, n_frames):
    """Binary per-frame silence labels for the crop starting at native sample `start`."""
    t0 = start / sr
    lab = np.zeros(n_frames, np.float32)
    if len(iv):
        s = iv[:, 0] / sr - t0
        e = iv[:, 1] / sr - t0
        sel = (e > 0) & (s < n_frames / FPS)
        for a_, b_ in zip(s[sel], e[sel]):
            a = max(0, math.ceil(a_ * FPS - 1e-6))
            b = min(n_frames, math.ceil(b_ * FPS - 1e-6))
            if b > a:
                lab[a:b] = 1.0
    return lab


def edge_heatmaps(gt_sec, start, sr, n_frames, sigma=EDGE_SIGMA):
    """[2, T] gaussian targets at the exact (sub-frame) silence start / end positions inside the crop."""
    heat = np.zeros((2, n_frames), np.float32)
    if len(gt_sec) == 0:
        return heat
    t0 = start / sr
    t = np.arange(n_frames, dtype=np.float32)
    for a, b in gt_sec:
        for ch, pos in ((0, (a - t0) * FPS), (1, (b - t0) * FPS)):
            if -3 * sigma <= pos <= n_frames - 1 + 3 * sigma:
                heat[ch] = np.maximum(heat[ch], np.exp(-((t - pos) ** 2) / (2 * sigma ** 2)))
    return heat


# ============================================================================= cache building
def _read_mono(path, channel=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True)       # [N, C]
    if channel is None:
        channel = int(np.argmin(np.abs(x.mean(axis=0)))) if x.shape[1] > 1 else 0   # same rule as inference
    return np.ascontiguousarray(x[:, min(channel, x.shape[1] - 1)]), sr, channel


def find_pairs(folder):
    if not os.path.isdir(folder):
        return []
    raws, marks = {}, {}
    for f in os.listdir(folder):
        m = re.match(r"^(.*)_raw\.(wav|flac)$", f, re.I)
        if m:
            raws[m.group(1)] = os.path.join(folder, f)
        m = re.match(r"^(.*)_marked\.(wav|flac)$", f, re.I)
        if m:
            marks[m.group(1)] = os.path.join(folder, f)
    for k in sorted(set(raws) ^ set(marks)):
        print(f"[WARNING] {folder}: '{k}' has only a {'_raw' if k in raws else '_marked'} file - skipped.")
    return [(k, raws[k], marks[k]) for k in sorted(set(raws) & set(marks))]


def find_plain(folder):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac"))]
    return [(os.path.splitext(f)[0], os.path.join(folder, f)) for f in sorted(files)]


def _sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, st.st_mtime_ns]


def build_cache(train_dir, val_dir, neg_dir, cache_dir, strict_mask=True):
    """Convert source audio to the FLAC cache (skipped automatically when nothing changed)."""
    os.makedirs(cache_dir, exist_ok=True)
    jobs = []   # (split, stem, raw, marked|None)
    for split, d in (("train", train_dir), ("val", val_dir)):
        jobs += [(split, s, r, m) for s, r, m in find_pairs(d)]
    jobs += [("neg", s, p, None) for s, p in find_plain(neg_dir)]

    signature = {"version": CACHE_VERSION, "strict": bool(strict_mask),
                 "src": [[j[0], j[1]] + _sig(j[2]) + (_sig(j[3]) if j[3] else []) for j in jobs]}
    index_path = os.path.join(cache_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        if idx.get("signature") == signature:
            print(f"[CACHE] Up to date ({len(idx['entries'])} files in {cache_dir}).")
            return idx["entries"]

    print(f"[CACHE] Building FLAC cache for {len(jobs)} files ...")
    for f in glob.glob(os.path.join(cache_dir, "*")):
        if f.lower().endswith((".flac", ".npy", ".json")):
            os.remove(f)

    entries = []
    for n, (split, stem, raw_p, mark_p) in enumerate(jobs):
        eid = f"{split}_{n:03d}_{re.sub(r'[^A-Za-z0-9_-]', '_', stem)[:40]}"
        raw, sr, ch = _read_mono(raw_p)
        if mark_p is not None:
            marked, sr_m, _ = _read_mono(mark_p, channel=ch)
            if sr_m != sr:
                marked = resample(torch.from_numpy(marked), sr_m, sr).numpy()
            n_ = min(len(raw), len(marked))
            raw, marked = raw[:n_], marked[:n_]
            mask = np.abs(marked - raw) > (1e-5 if strict_mask else 0.05)
            frac = float(mask.mean())
            if frac == 0.0:
                print(f"[WARNING] {stem}: raw and marked are identical -> no silence labels at all.")
            elif frac > 0.9:
                print(f"[WARNING] {stem}: {frac:.0%} of samples differ between raw and marked - "
                      f"gain/dither mismatch? Check the pair.")
            iv = mask_to_intervals(mask)
        else:
            iv = np.zeros((0, 2), np.int64)

        peak = float(np.abs(raw).max())
        if peak > 0.999:                                   # PCM_24 cannot hold > 1.0
            raw = raw * (0.999 / peak)
            peak = 0.999
        if mark_p is None:                                 # miner writes the ORIGINAL file's peak next to the crop
            side = os.path.splitext(raw_p)[0] + ".peak.txt"
            if os.path.exists(side):
                try:
                    peak = float(open(side).read().strip())
                except ValueError:
                    pass

        sf.write(os.path.join(cache_dir, eid + ".flac"), raw, sr, subtype="PCM_24", format="FLAC")
        np.save(os.path.join(cache_dir, eid + ".npy"), iv)
        entries.append({"id": eid, "split": split, "name": stem, "sr": int(sr), "n": int(len(raw)),
                        "peak": max(peak, 1e-6), "n_intervals": int(len(iv))})
        print(f"   [{split:5s}] {stem}: {len(raw)/sr/60:.1f} min @ {sr} Hz, {len(iv)} silence intervals")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"signature": signature, "entries": entries}, f)
    return entries


def load_entries(entries, cache_dir):
    out = []
    for e in entries:
        iv = np.load(os.path.join(cache_dir, e["id"] + ".npy"))
        d = dict(e)
        d["path"] = os.path.join(cache_dir, e["id"] + ".flac")
        d["iv"] = iv
        d["gt_sec"] = merge_intervals_sec(iv, e["sr"])
        d["lo"], d["hi"] = 0, e["n"]
        out.append(d)
    return out


def split_entries(entries, val_frac_if_missing=0.10):
    """-> train, val, neg. If there are no validation files, the last part of each train file is held out."""
    train = [e for e in entries if e["split"] == "train"]
    val = [e for e in entries if e["split"] == "val"]
    neg = [e for e in entries if e["split"] == "neg"]
    if not val and train:
        print(f"[VAL] No validation files - holding out the last {val_frac_if_missing:.0%} of every training file.")
        new_train = []
        for e in train:
            cut = int(e["n"] * (1.0 - val_frac_if_missing))
            a, b = dict(e), dict(e)
            a["hi"] = cut
            b["lo"] = cut
            new_train.append(a)
            val.append(b)
        train = new_train
    return train, val, neg


def make_val_windows(val_entries, max_windows=300, seed=42):
    """Deterministic non-overlapping 8 s windows, balanced across files. -> list of (entry_idx, start)."""
    per = []
    for i, e in enumerate(val_entries):
        n_nat = int(round(SEGMENT_SEC * e["sr"]))
        starts = list(range(e["lo"], e["hi"] - n_nat + 1, n_nat))
        per.append([(i, s) for s in starts])
    total = sum(len(p) for p in per)
    if total > max_windows:
        rng = random.Random(seed)
        quota = max(1, max_windows // max(1, len(per)))
        per = [sorted(rng.sample(p, quota)) if len(p) > quota else p for p in per]
    return [w for p in per for w in p]


# ============================================================================= dataset
def _read_crop(path, start, n):
    with sf.SoundFile(path) as f:
        f.seek(start)
        x = f.read(n, dtype="float32")
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return x


class V7Dataset(Dataset):
    """
    mode='train': infinite random crops (length = how many samples one "epoch" yields).
    mode='val'  : fixed list of windows, no augmentation.
    Returns (wave[SEG_SAMPLES] @ INTERNAL_SR, labels[T], edge_heat[2,T], index).
    """

    def __init__(self, entries, mode="train", length=1000, neg_entries=None, neg_prob=0.10,
                 windows=None, aug=True, bridging=True):
        self.entries = entries
        self.mode = mode
        self.length = length if mode == "train" else len(windows)
        self.windows = windows
        self.neg = neg_entries or []
        self.neg_prob = neg_prob if self.neg else 0.0
        self.aug = aug and mode == "train"
        self.bridging = bridging

        if mode == "train":
            self.pool, w = [], []
            for e in entries:
                span = e["hi"] - e["lo"]
                if span >= int(round(SEGMENT_SEC * e["sr"])):
                    self.pool.append(e)
                    w.append(span)
                else:
                    print(f"[WARNING] {e['name']}: shorter than {SEGMENT_SEC:.0f} s - not used for training.")
            self.weights = w
            if not self.pool:
                raise RuntimeError("No training file is long enough (need >= 8 s).")

    def __len__(self):
        return self.length

    # ---- augmentation helpers
    @staticmethod
    def _silence_noise(x, iv_rel, sr):
        mask = np.zeros(len(x), np.float32)
        for s, e in iv_rel:
            mask[max(0, s):min(len(x), e)] = 1.0
        if mask.max() == 0:
            return x
        k = max(3, int(sr * 0.001) | 1)                      # 1 ms soft edges
        mask = np.convolve(mask, np.ones(k, np.float32) / k, mode="same")
        rng = np.random.default_rng(random.getrandbits(63))
        noise = rng.standard_normal(len(x)).astype(np.float32)
        if random.random() < 0.5:                            # coloured (low-pass) noise
            noise = lfilter([1.0], [1.0, -random.uniform(0.5, 0.95)], noise).astype(np.float32)
            noise /= (noise.std() + 1e-9)
        sigma = 10 ** (random.uniform(-110.0, -60.0) / 20.0)
        return x + sigma * noise * mask

    def _to_internal(self, x, sr):
        t = torch.from_numpy(x).unsqueeze(0)
        lows = [s for s in LOW_SRS if s < sr]
        if self.aug and lows and random.random() < 0.35:     # pretend the source was a low-rate recording
            vsr = random.choice(lows)
            t = resample(resample(t, sr, vsr), vsr, INTERNAL_SR)
        else:
            t = resample(t, sr, INTERNAL_SR)
        return fit_length(t, SEG_SAMPLES).squeeze(0).contiguous()

    def _build(self, e, start):
        sr = e["sr"]
        n_nat = int(round(SEGMENT_SEC * sr))
        x = _read_crop(e["path"], start, n_nat) * (PEAK_TARGET / e["peak"])   # file-level peak norm == inference

        if self.aug:
            if random.random() < 0.7:                                          # level
                x = x * (10 ** (random.uniform(-8.0, 0.8) / 20.0))   # never clips: file peak is 0.9
            if random.random() < 0.5:                                          # polarity
                x = -x
            if random.random() < 0.10:                                         # DC offset
                x = x + random.choice((-1, 1)) * 10 ** (random.uniform(-80.0, -54.0) / 20.0)
            if random.random() < 0.40 and len(e["iv"]):                        # noise inside labelled silences
                iv = e["iv"]
                sel = (iv[:, 1] > start) & (iv[:, 0] < start + n_nat)
                self_iv = iv[sel] - start
                if len(self_iv):
                    x = self._silence_noise(x, self_iv, sr)
            x = np.clip(x, -1.0, 1.0).astype(np.float32)

        wave = self._to_internal(x, sr)
        lab = frame_labels(e["iv"], sr, start, SEG_FRAMES)
        if self.bridging:
            lab = close_probs(lab)
        heat = edge_heatmaps(e["gt_sec"], start, sr, SEG_FRAMES)
        return wave, torch.from_numpy(lab), torch.from_numpy(heat)

    def __getitem__(self, i):
        if self.mode == "val":
            ei, start = self.windows[i]
            wave, lab, heat = self._build(self.entries[ei], start)
            return wave, lab, heat, i

        if self.neg and random.random() < self.neg_prob:
            e = random.choice(self.neg)
            n_nat = int(round(SEGMENT_SEC * e["sr"]))
            if e["hi"] - e["lo"] < n_nat:                    # short negative -> tile it
                x = sf.read(e["path"], dtype="float32")[0]
                x = np.tile(x, int(math.ceil(n_nat / max(len(x), 1))))[:n_nat]
                sr = e["sr"]
                x = x * (PEAK_TARGET / e["peak"])
                if self.aug:
                    x = x * (10 ** (random.uniform(-8.0, 0.8) / 20.0)) * (-1 if random.random() < 0.5 else 1)
                x = np.clip(x, -1, 1).astype(np.float32)
                wave = self._to_internal(x, sr)
                return wave, torch.zeros(SEG_FRAMES), torch.zeros(2, SEG_FRAMES), -1
            start = random.randint(e["lo"], e["hi"] - n_nat)
        else:
            e = random.choices(self.pool, weights=self.weights, k=1)[0]
            n_nat = int(round(SEGMENT_SEC * e["sr"]))
            start = random.randint(e["lo"], e["hi"] - n_nat)
        wave, lab, heat = self._build(e, start)
        return wave, lab, heat, -1
