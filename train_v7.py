"""
SmartCutter v7 training.

Data layout (same as v3):
    training_files/    X_raw.wav|flac + X_marked.wav|flac      (paired by name X)
    validation_files/  same format (optional; otherwise the last 10 % of each training file is held out)
    hard_negatives/    any audio that must NEVER be cut (optional)
Audio is converted once to a FLAC cache (cache_v7/) and read from there with random crops.

Output:  ckpts/v7_model.pth   - self-describing (weights + model config + tuned threshold + edge setting).
         ckpts/v7_last_state.pt - full training state, used to resume automatically.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_v7 import (V7Dataset, build_cache, load_entries, make_val_windows, split_entries)
from features_v7 import (FPS, INTERNAL_SR, SEG_SAMPLES, SEGMENT_SEC, FeatureExtractor)
from metrics_v7 import FP_WEIGHT, sweep_threshold
from model_v7 import DEFAULT_CONFIG, SmartCutterV7, count_params

# ----------------------------------------------------------------------------- config
TRAIN_DIR = "training_files"
VAL_DIR = "validation_files"
NEG_DIR = "hard_negatives"
CACHE_DIR = "cache_v7"
CKPT_DIR = "ckpts"
DEPLOY_NAME = "v7_model.pth"
STATE_NAME = "v7_last_state.pt"

STRICT_MASK = True               # raw-vs-marked difference threshold 1e-5 (False -> 0.05)

BATCH_SIZE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
MIN_LR_FRAC = 0.01               # cosine decays to LEARNING_RATE * this
MAX_EPOCHS = 150
EPOCH_STEPS = None               # None = auto (one pass over the training audio, 40..400 steps)
EARLY_STOP_PATIENCE = 50         # epochs without a new best validation score (None = off)
GRAD_CLIP = 1.0
EMA_DECAY = 0.999

# loss: penalise false cuts (cutting speech) harder than missed silence
BCE_POS_WEIGHT = 0.75            # < 1 : missing silence is cheaper than cutting speech
TVERSKY_ALPHA = 0.7              # weight on false positives
TVERSKY_BETA = 0.3               # weight on false negatives
EDGE_LOSS_WEIGHT = 0.5

NEG_PROB = 0.10                  # chance a training crop comes from hard_negatives/ (if any)
NUM_WORKERS = min(6, os.cpu_count() or 1)
VAL_MAX_WINDOWS = 300
THRESHOLDS = np.arange(0.20, 0.801, 0.05)
RESUME = True

SMALL_CONFIG = dict(dim=64, stem_channels=(8, 16, 16, 24, 32), stats_dim=16,
                    tcn_dilations=(1, 2, 4, 8), gru_layers=1)   # --small: quick smoke tests only


# ----------------------------------------------------------------------------- helpers
class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def tversky_loss(logits, y, alpha, beta, smooth=1.0):
    p = torch.sigmoid(logits)
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def compute_loss(out, y, heat, pos_weight):
    sil = out["sil"].float()
    edge = out["edge"].float()
    l_sil = F.binary_cross_entropy_with_logits(sil, y, pos_weight=pos_weight) \
        + tversky_loss(sil, y, TVERSKY_ALPHA, TVERSKY_BETA)
    l_edge = F.binary_cross_entropy_with_logits(edge, heat, weight=1.0 + 9.0 * heat)
    return l_sil + EDGE_LOSS_WEIGHT * l_edge, l_sil.detach(), l_edge.detach()


def lr_at(step, total_steps, base_lr):
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    prog = min(1.0, (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS))
    return base_lr * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))


def builtin(obj):
    return json.loads(json.dumps(obj))       # tuples -> lists, numpy -> python (weights_only-safe)


def save_deploy(path, ema_model, thr, use_edge, metrics, epoch):
    torch.save({
        "format": "smartcutter_v7",
        "state_dict": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
        "config": builtin(ema_model.config),
        "internal_sr": INTERNAL_SR,
        "threshold": float(thr),
        "edge_refine": bool(use_edge),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "epoch": int(epoch),
    }, path)


def worker_init(_):
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, fx, loader, val_ds, val_entries, device, amp_ctx, pos_weight):
    model.eval()
    P, E, order, tot_loss, n = [], [], [], 0.0, 0
    for wave, y, heat, idx in loader:
        wave, y, heat = wave.to(device), y.to(device), heat.to(device)
        spec, stats = fx(wave)
        with amp_ctx():
            out = model(spec, stats)
        loss, _, _ = compute_loss(out, y, heat, pos_weight)
        tot_loss += loss.item() * len(idx)
        n += len(idx)
        P.append(torch.sigmoid(out["sil"].float()).cpu())
        E.append(torch.sigmoid(out["edge"].float()).cpu())
        order += idx.tolist()
    P = torch.cat(P).numpy()
    E = torch.cat(E).numpy()

    gts = []
    for i in order:
        ei, start = val_ds.windows[i]
        e = val_entries[ei]
        gt = e["gt_sec"] - start / e["sr"]
        if len(gt):
            gt = gt[(gt[:, 1] > 0) & (gt[:, 0] < SEGMENT_SEC)]
        gts.append(gt.reshape(-1, 2))

    thr, r_plain, r_edge = sweep_threshold(P, E, gts, THRESHOLDS)
    use_edge = r_edge["score"] > r_plain["score"]
    r = dict(r_edge if use_edge else r_plain)
    r["val_loss"] = tot_loss / max(n, 1)
    r["score_no_edge"] = r_plain["score"]
    r["score_edge"] = r_edge["score"]
    r["bnd_mae_no_edge"] = r_plain["boundary_mae_ms"]
    r["bnd_mae_edge"] = r_edge["boundary_mae_ms"]
    return thr, use_edge, r


# ----------------------------------------------------------------------------- main
def train(args):
    global WARMUP_STEPS
    if args.warmup is not None:
        WARMUP_STEPS = args.warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(CKPT_DIR, exist_ok=True)
    for d in (TRAIN_DIR, VAL_DIR, NEG_DIR):
        os.makedirs(d, exist_ok=True)

    batch = args.batch or BATCH_SIZE
    workers = NUM_WORKERS if args.workers is None else args.workers
    max_epochs = args.epochs or MAX_EPOCHS

    # --- data
    raw_entries = build_cache(TRAIN_DIR, VAL_DIR, NEG_DIR, CACHE_DIR, strict_mask=STRICT_MASK)
    entries = load_entries(raw_entries, CACHE_DIR)
    train_e, val_e, neg_e = split_entries(entries)
    if not train_e:
        print(f"[ERROR] No training pairs found in '{TRAIN_DIR}'. Expected X_raw.wav + X_marked.wav pairs.")
        return

    total_sec = sum((e["hi"] - e["lo"]) / e["sr"] for e in train_e)
    steps = args.epoch_steps or EPOCH_STEPS or int(np.clip(total_sec / (SEGMENT_SEC * batch), 40, 400))
    windows = make_val_windows(val_e, args.val_windows or VAL_MAX_WINDOWS)
    print(f"[DATA] train {total_sec/60:.1f} min in {len(train_e)} files | hard negatives {len(neg_e)} | "
          f"val windows {len(windows)} | {steps} steps/epoch x batch {batch}")

    train_ds = V7Dataset(train_e, "train", length=steps * batch, neg_entries=neg_e, neg_prob=NEG_PROB)
    val_ds = V7Dataset(val_e, "val", windows=windows, aug=False)
    loader_kw = dict(num_workers=workers, pin_memory=device.type == "cuda",
                     persistent_workers=workers > 0, worker_init_fn=worker_init)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **loader_kw)

    # --- model
    cfg = dict(DEFAULT_CONFIG)
    if args.small:
        cfg.update(SMALL_CONFIG)
    model = SmartCutterV7(cfg).to(device)
    fx = FeatureExtractor().to(device)
    ema = ModelEMA(model, EMA_DECAY)
    print(f"[MODEL] SmartCutterV7: {count_params(model)/1e6:.2f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_cuda and amp_dtype == torch.float16) else None

    def amp_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda)

    pos_weight = torch.tensor(BCE_POS_WEIGHT, device=device)
    total_steps = max_epochs * steps

    # --- resume
    start_epoch, step, best, bad = 0, 0, -1.0, 0
    state_path = os.path.join(CKPT_DIR, STATE_NAME)
    if RESUME and os.path.exists(state_path) and not args.fresh:
        st = torch.load(state_path, map_location=device, weights_only=False)
        if st.get("config") == builtin(cfg):
            model.load_state_dict(st["model"])
            ema.module.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
            opt.load_state_dict(st["opt"])
            if scaler and st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            start_epoch, step, best, bad = st["epoch"], st["step"], st["best"], st["bad"]
            print(f"[RESUME] continuing from epoch {start_epoch} (best score so far {best:.4f})")
        else:
            print("[RESUME] saved state has a different model config - starting fresh.")

    print(f"[SCORE] 1/(1+(w*false_cut_ms + missed_ms)/silence_ms), w={FP_WEIGHT:g}  (higher is better)")
    best_info = None
    for epoch in range(start_epoch, max_epochs):
        model.train()
        t0 = time.time()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        for wave, y, heat, _ in pbar:
            wave, y, heat = wave.to(device, non_blocking=True), y.to(device, non_blocking=True), heat.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, LEARNING_RATE)

            spec, stats = fx(wave)                       # features computed on the GPU, in fp32
            with amp_ctx():
                out = model(spec, stats)
            loss, l_sil, l_edge = compute_loss(out, y, heat, pos_weight)
            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            ema.update(model)
            step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{losses[-1]:.3f}", sil=f"{l_sil.item():.3f}", edge=f"{l_edge.item():.3f}")

        thr, use_edge, r = validate(ema.module, fx, val_loader, val_ds, val_e, device, amp_ctx, pos_weight)
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train {np.mean(losses):.4f} | val loss {r['val_loss']:.4f} | LR {lr_now:.1e} | {time.time()-t0:.0f}s")
        print(f"          score {r['score']:.4f} @thr {thr:.2f} | false-cut {r['false_cut_pct']:.3f}% of non-silence "
              f"({r['false_cut_s_per_min']:.2f} s/min) | missed {r['missed_pct']:.1f}% | "
              f"boundary MAE {r['boundary_mae_ms']:.1f} ms (p90 {r['boundary_p90_ms']:.1f}) | "
              f"edge-heads {'ON' if use_edge else 'off'} ({r['bnd_mae_no_edge']:.1f} -> {r['bnd_mae_edge']:.1f} ms)")

        if r["score"] > best:
            best, bad, best_info = r["score"], 0, (epoch + 1, thr, use_edge, r)
            save_deploy(os.path.join(CKPT_DIR, DEPLOY_NAME), ema.module, thr, use_edge, r, epoch + 1)
            print(f"   ---> new best, saved {DEPLOY_NAME}")
        else:
            bad += 1

        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "ema_updates": ema.updates,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "epoch": epoch + 1, "step": step, "best": best, "bad": bad, "config": builtin(cfg)}, state_path)

        if EARLY_STOP_PATIENCE and bad >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] no improvement for {bad} epochs.")
            break

    print("\n" + "=" * 60)
    if best_info:
        ep, thr, use_edge, r = best_info
        print(f" Best epoch {ep}: score {r['score']:.4f}  threshold {thr:.2f}  edge-heads {'ON' if use_edge else 'off'}")
        print(f" false cut {r['false_cut_pct']:.3f}% | missed {r['missed_pct']:.1f}% | boundary MAE {r['boundary_mae_ms']:.1f} ms")
        print(f" -> {os.path.join(CKPT_DIR, DEPLOY_NAME)}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--epoch-steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--val-windows", type=int)
    ap.add_argument("--warmup", type=int, help="override WARMUP_STEPS (short test runs)")
    ap.add_argument("--small", action="store_true", help="tiny model for smoke tests")
    ap.add_argument("--fresh", action="store_true", help="ignore saved training state")
    train(ap.parse_args())
