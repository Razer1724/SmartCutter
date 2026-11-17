"""
SmartCutter - shared cutting logic (v3 and v7 both use this).

Input is a per-frame probability track at 100 frames/second (frame t centred on t/100 s),
which is sample-rate independent. Everything here converts frames -> samples by TIME.

Fixes relative to the original inference.py
-------------------------------------------
1. A file with no detected silence used to come out as ALL ZEROS. It is now returned unchanged.
2. Zero-crossing snapping used to move a cut start/end to the nearest crossing in EITHER direction
   (up to 25 ms), which could pull a cut into speech. Now a cut START only moves forward (into the
   silence) and a cut END only moves backward (into the silence).
3. A gap only slightly longer than the target silence was "cut" and refilled with the target length,
   so the file could get LONGER. Gaps must now save at least MIN_NET_SAVING_MS to be cut.
4. Frame->sample mapping: the original stretched the frame mask over the file with
   align_corners=True, which drifts up to ~1-2 frames (10-20 ms) by the end of a long file.
   Positions are now exact (frame t <-> sample t * sr / 100), with sub-frame threshold crossings.
5. Zero crossings are searched locally around each cut instead of building sign/diff tensors
   the size of the whole file (hours of audio = GBs of RAM).

Unchanged on purpose: single probability threshold, grayscale gap-bridging (closing) of the mask,
mono channel selection, fades, fixed-length silence insertion.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

FPS = 100


@dataclass
class CutConfig:
    threshold: float = 0.5              # CUTTING_PROBABILITY
    min_segment_ms: float = 100.0       # silences shorter than this are never cut
    safety_buffer_ms: float = 5.0       # shrink each detected silence by this much on both sides
    search_window_ms: float = 25.0      # max distance a cut may be snapped to a zero crossing
    fade_ms: float = 10.0
    target_silence_s: float = 0.100     # silence length inserted at every cut
    min_net_saving_ms: float = 10.0     # only cut if it makes the file at least this much shorter
    bridging: bool = True               # fill small gaps in the mask (closing), as in v3
    bridge_frames: int = 5
    edge_refine: bool = False           # v7 only: refine boundaries with the edge heads
    edge_radius_frames: int = 3
    edge_min_peak: float = 0.3


# ----------------------------------------------------------------------------- mask -> segments
def close_probs(p, k=5):
    """Grayscale closing (dilate then erode) - identical operation to the original v3 bridging."""
    t = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).view(1, 1, -1)
    t = F.max_pool1d(t, k, 1, k // 2)
    t = -F.max_pool1d(-t, k, 1, k // 2)
    return t.view(-1).numpy()


def _refine_peak(heat, pos_f, radius, min_peak):
    """Move a boundary (float frame position) to the nearest strong edge-head peak, sub-frame precise."""
    T = len(heat)
    c = int(round(pos_f))
    lo, hi = max(0, c - radius), min(T - 1, c + radius)
    if hi < lo:
        return pos_f
    seg = heat[lo:hi + 1]
    k = int(np.argmax(seg))
    if seg[k] < min_peak:
        return pos_f
    pos = lo + k
    if 0 < pos < T - 1:
        y0, y1, y2 = heat[pos - 1], heat[pos], heat[pos + 1]
        denom = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / denom if denom < -1e-9 else 0.0
        return pos + float(np.clip(off, -0.5, 0.5))
    return float(pos)


def probs_to_segments(p, total_frames_f, cfg, edge_start=None, edge_end=None):
    """
    p: [T] probabilities. total_frames_f: file length in frames (float) - end of a segment that
    runs to the end of the file is placed here.
    Returns (starts, ends) float arrays in FRAMES, already length-filtered and safety-shrunk.
    """
    p = np.asarray(p, dtype=np.float32)
    if cfg.bridging:
        p = close_probs(p, cfg.bridge_frames)
    T = len(p)
    above = p > cfg.threshold
    if not above.any():
        return np.zeros(0), np.zeros(0)

    d = np.diff(above.astype(np.int8), prepend=0, append=0)
    s_idx = np.where(d == 1)[0]
    e_idx = np.where(d == -1)[0]
    starts = np.empty(len(s_idx))
    ends = np.empty(len(e_idx))
    for i, s in enumerate(s_idx):
        if s == 0:
            starts[i] = 0.0
        else:
            p0, p1 = float(p[s - 1]), float(p[s])
            starts[i] = (s - 1) + (cfg.threshold - p0) / max(p1 - p0, 1e-9)
    for i, e in enumerate(e_idx):
        if e >= T:
            ends[i] = total_frames_f
        else:
            p0, p1 = float(p[e - 1]), float(p[e])
            ends[i] = (e - 1) + (p0 - cfg.threshold) / max(p0 - p1, 1e-9)

    if cfg.edge_refine and edge_start is not None and edge_end is not None:
        for i in range(len(starts)):
            if s_idx[i] != 0:
                starts[i] = _refine_peak(edge_start, starts[i], cfg.edge_radius_frames, cfg.edge_min_peak)
        for i in range(len(ends)):
            if e_idx[i] < T:
                ends[i] = _refine_peak(edge_end, ends[i], cfg.edge_radius_frames, cfg.edge_min_peak)

    keep = (ends - starts) >= cfg.min_segment_ms / 1000.0 * FPS
    starts, ends = starts[keep], ends[keep]
    buf = cfg.safety_buffer_ms / 1000.0 * FPS
    starts, ends = starts + buf, ends - buf
    ok = ends > starts
    return starts[ok], ends[ok]


# ----------------------------------------------------------------------------- sample-level snapping
def snap_to_zero_crossings(starts_s, ends_s, wav, sr, cfg):
    """
    starts_s / ends_s: int sample positions. wav: 1-D numpy mono signal.
    Start moves FORWARD to the first zero crossing within the window (into the silence);
    end moves BACKWARD to the last zero crossing within the window (into the silence).
    """
    win = int(sr * cfg.search_window_ms / 1000.0)
    n = len(wav)
    out_s, out_e = [], []
    for s, e in zip(starts_s, ends_s):
        s, e = int(s), int(e)
        s = max(0, min(s, n))
        e = max(0, min(e, n))
        if e <= s:
            continue
        hi = min(s + win, e)
        seg = np.sign(wav[s:hi + 1])
        zc = np.nonzero(np.diff(seg))[0]
        if len(zc):
            s = s + int(zc[0])
        lo = max(e - win, s)
        seg = np.sign(wav[lo:e + 1])
        zc = np.nonzero(np.diff(seg))[0]
        if len(zc):
            e = lo + int(zc[-1])
        if e > s:
            out_s.append(s)
            out_e.append(e)
    return np.asarray(out_s, dtype=np.int64), np.asarray(out_e, dtype=np.int64)


def get_cosine_fade(length, device="cpu"):
    t = torch.linspace(0, np.pi, length, device=device)
    return 0.5 * (1 - torch.cos(t))


def apply_fade(waveform, fade_samples, mode="both"):
    if fade_samples <= 0 or waveform.shape[1] < fade_samples * 2:
        return waveform
    curve = get_cosine_fade(fade_samples, waveform.device)
    if mode in ("in", "both"):
        waveform[:, :fade_samples] *= curve
    if mode in ("out", "both"):
        waveform[:, -fade_samples:] *= curve.flip(0)
    return waveform


def find_cuts(probs, waveform, sr, cfg, edge_probs=None):
    """Returns (starts, ends) as int sample positions of the regions to remove."""
    n = waveform.shape[1]
    total_frames_f = n * FPS / sr
    es = ee = None
    if edge_probs is not None:
        es, ee = edge_probs[0], edge_probs[1]
    seg_s, seg_e = probs_to_segments(probs, total_frames_f, cfg, es, ee)
    if len(seg_s) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    s = np.rint(seg_s * sr / FPS).astype(np.int64)
    e = np.rint(seg_e * sr / FPS).astype(np.int64)
    mono = waveform.mean(dim=0).cpu().numpy() if waveform.shape[0] > 1 else waveform[0].cpu().numpy()
    s, e = snap_to_zero_crossings(s, e, mono, sr, cfg)

    # Fix 3: only cut when it actually shortens the file.
    need = int(sr * cfg.target_silence_s) + int(sr * cfg.min_net_saving_ms / 1000.0)
    keep = (e - s) >= need
    return s[keep], e[keep]


def render(waveform, starts, ends, sr, cfg):
    """Cut [start,end) regions out and put fixed-length digital silence in their place."""
    n = waveform.shape[1]
    target = int(sr * cfg.target_silence_s)
    fade = int(sr * cfg.fade_ms / 1000.0)
    silence = torch.zeros((waveform.shape[0], target), dtype=waveform.dtype)

    pieces, last = [], 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if s > last:
            chunk = waveform[:, last:s].clone()
            if last > 0:
                chunk = apply_fade(chunk, fade, "in")
            chunk = apply_fade(chunk, fade, "out")
            pieces.append(chunk)
        pieces.append(silence)
        last = e
    if last < n:
        tail = waveform[:, last:].clone()
        if last > 0:
            tail = apply_fade(tail, fade, "in")
        pieces.append(tail)
    return torch.cat(pieces, dim=1)


def smart_cut(waveform, probs, sr, cfg, edge_probs=None):
    """
    waveform: [C, N] CPU float tensor (original audio at its original sample rate).
    probs: [T] 100 fps probability track. Returns (cleaned [C, N'], starts, ends).
    If nothing is cuttable the input is returned UNCHANGED (original code returned zeros).
    """
    waveform = waveform.cpu()
    starts, ends = find_cuts(probs, waveform, sr, cfg, edge_probs)
    if len(starts) == 0:
        return waveform, starts, ends
    return render(waveform, starts, ends, sr, cfg), starts, ends
