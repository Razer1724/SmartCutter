"""
SmartCutter v7 - validation metrics.

Frame Dice (what v3 reported) does not measure what matters. Cutting SPEECH is far worse than
missing a bit of silence, and boundary position matters. These metrics are computed on the
segments the real cutter would produce (same bridging, threshold, minimum length, safety buffer
and "must actually shorten the file" rules as inference), on a 1 ms grid:

  false cut   : ms of audio that would be removed but is NOT labelled silence   (the bad error)
  missed      : ms of labelled, cuttable silence that would NOT be removed
  boundary    : |predicted - true| position of each silence start/end, in ms

and one scalar used for model selection:   score = 1 / (1 + (FP_WEIGHT*false + missed) / silence_ms)
"""
import numpy as np

from cutter import CutConfig, probs_to_segments
from features_v7 import FPS, SEGMENT_SEC

FP_WEIGHT = 5.0          # a false cut costs this many times a missed ms
EDGE_MARGIN_S = 0.25     # ignore the first/last 250 ms of each window (chunk-edge context is poor)
MATCH_RADIUS_S = 0.30


def predicted_segments_sec(p, edge, cfg, win_sec=SEGMENT_SEC):
    """p [T] probs, edge [2,T] probs or None -> (starts, ends) seconds relative to the window."""
    es = ee = None
    if edge is not None:
        es, ee = edge[0], edge[1]
    s, e = probs_to_segments(p, win_sec * FPS, cfg, es, ee)
    s, e = s / FPS, e / FPS
    keep = (e - s) >= cfg.target_silence_s + cfg.min_net_saving_ms / 1000.0     # same rule as find_cuts()
    return s[keep], e[keep]


def _raster(starts, ends, n_ms):
    r = np.zeros(n_ms, bool)
    for a, b in zip(starts, ends):
        i0, i1 = max(0, int(round(a * 1000))), min(n_ms, int(round(b * 1000)))
        if i1 > i0:
            r[i0:i1] = True
    return r


class Accumulator:
    def __init__(self):
        self.fp = self.fn = self.sil = self.speech = self.total = 0
        self.errs = []
        self.missed_edges = 0

    def add(self, pred_s, pred_e, gt, win_sec=SEGMENT_SEC):
        """gt: float array [K,2] seconds relative to window start (may extend outside the window)."""
        n_ms = int(win_sec * 1000)
        lo, hi = int(EDGE_MARGIN_S * 1000), n_ms - int(EDGE_MARGIN_S * 1000)
        pr = _raster(pred_s, pred_e, n_ms)
        gr = _raster(gt[:, 0], gt[:, 1], n_ms) if len(gt) else np.zeros(n_ms, bool)
        sl = slice(lo, hi)
        self.fp += int((pr & ~gr)[sl].sum())
        self.fn += int((~pr & gr)[sl].sum())
        self.sil += int(gr[sl].sum())
        self.speech += int((~gr)[sl].sum())
        self.total += hi - lo

        for a, b in (gt if len(gt) else []):
            for pos, cand in ((a, pred_s), (b, pred_e)):
                if EDGE_MARGIN_S <= pos <= win_sec - EDGE_MARGIN_S:
                    if len(cand) and np.abs(cand - pos).min() <= MATCH_RADIUS_S:
                        self.errs.append(float(np.abs(cand - pos).min() * 1000))
                    else:
                        self.missed_edges += 1

    def result(self):
        cost = (FP_WEIGHT * self.fp + self.fn) / max(self.sil, 1)
        errs = np.asarray(self.errs) if self.errs else np.zeros(1)
        return {
            "score": 1.0 / (1.0 + cost),
            "false_cut_pct": 100.0 * self.fp / max(self.speech, 1),     # % of non-silence that would be cut
            "false_cut_s_per_min": self.fp / 1000.0 / max(self.total / 60000.0, 1e-9),
            "missed_pct": 100.0 * self.fn / max(self.sil, 1),           # % of cuttable silence left in
            "boundary_mae_ms": float(errs.mean()),
            "boundary_p90_ms": float(np.percentile(errs, 90)),
            "missed_edges": self.missed_edges,
            "silence_ms": self.sil,
        }


def evaluate(probs, edges, gts, cfg, use_edges=False):
    """probs [W,T], edges [W,2,T], gts list of [K,2] arrays -> metrics dict."""
    acc = Accumulator()
    for w in range(len(probs)):
        s, e = predicted_segments_sec(probs[w], edges[w] if use_edges else None, cfg)
        acc.add(s, e, gts[w])
    return acc.result()


def sweep_threshold(probs, edges, gts, thresholds, base_cfg=None):
    """Best threshold (edge heads off) and whether edge refinement helps at that threshold."""
    base = base_cfg or CutConfig()
    best = None
    for thr in thresholds:
        cfg = CutConfig(**{**base.__dict__, "threshold": float(thr), "edge_refine": False})
        r = evaluate(probs, edges, gts, cfg, use_edges=False)
        if best is None or r["score"] > best[1]["score"]:
            best = (float(thr), r)
    thr, r_plain = best
    cfg = CutConfig(**{**base.__dict__, "threshold": thr, "edge_refine": True})
    r_edge = evaluate(probs, edges, gts, cfg, use_edges=True)
    return thr, r_plain, r_edge
