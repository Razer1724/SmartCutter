"""Reproducible training with vocal-preservation checkpoint selection."""

import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import ARCHITECTURE, COMPATIBLE_ARCHITECTURES, MODEL_VERSION, __version__
from .data import AudioDataset, collate_audio
from .engine import Predictor
from .evaluate import calibrate
from .model import SmartCutterV9


def segmentation_loss(outputs, batch, config):
    valid = batch["valid"].float()
    targets = batch["cut"]
    # Cut=1 means removable. Weight NEGATIVE targets more heavily: pos_weight
    # would emphasize the opposite error (retaining removable silence).
    weights = torch.where(targets > 0.5, 1.0, config.protected_weight) * valid
    cut_loss = (
        F.binary_cross_entropy_with_logits(outputs["cut"], targets, reduction="none") * weights
    ).sum() / weights.sum().clamp_min(1)
    boundary_targets = batch["boundaries"]
    logits = outputs["boundaries"]
    probability = logits.sigmoid()
    pt = probability * boundary_targets + (1 - probability) * (1 - boundary_targets)
    alpha = 0.75 * boundary_targets + 0.25 * (1 - boundary_targets)
    boundary_loss = (
        F.binary_cross_entropy_with_logits(logits, boundary_targets, reduction="none")
        * (1 - pt).square()
        * alpha
        * valid.unsqueeze(-1)
    ).sum() / (2 * valid.sum()).clamp_min(1)
    return cut_loss + config.boundary_weight * boundary_loss


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(records, config, output_dir, device="cpu", resume=None, manifest_sha256=""):
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and any(
        (output_dir / name).exists() for name in ("best.pt", "latest.pt", "history.jsonl")
    ):
        raise FileExistsError(
            "Run directory already contains training results; use --resume or a new output directory"
        )
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "validation"]
    if not train_records or not val_records:
        raise ValueError("Training requires nonempty train and validation splits")
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(config.train.seed)
    model = SmartCutterV9(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, stalled, best_rank = 0, 0, None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            state.get("architecture") not in COMPATIBLE_ARCHITECTURES
            or state.get("format_version") != 1
        ):
            raise ValueError("Resume requires a v9 training checkpoint")
        for section in ("frontend", "model"):
            if state["config"][section] != config.to_dict()[section]:
                raise ValueError(f"Resume {section} configuration differs")
        if manifest_sha256 and state.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                "Dataset manifest changed since this run; start a new run for a changed dataset"
            )
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        stalled = state["stalled"]
        best_rank = tuple(state["best_rank"]) if state["best_rank"] is not None else None
        torch.set_rng_state(state["rng_torch"])
        random.setstate(state["rng_python"])
        numpy_state = state["rng_numpy"]
        np.random.set_state(
            (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        )
        generator.set_state(state["rng_loader"])
        if device.type == "cuda" and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    if start_epoch >= config.train.epochs:
        raise ValueError(
            "Epoch limit has already been reached; increase train.epochs when resuming"
        )
    datasets = [
        AudioDataset(train_records, config, augment=True),
        AudioDataset(val_records, config),
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=i == 0,
            num_workers=config.train.workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_audio,
            worker_init_fn=seed_worker,
            generator=generator if i == 0 else None,
        )
        for i, dataset in enumerate(datasets)
    ]
    predictor = Predictor(model, config, device)
    for epoch in range(start_epoch, config.train.epochs):
        losses = []
        for training, loader in ((True, loaders[0]), (False, loaders[1])):
            model.train(training)
            total, count = 0.0, 0
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with (
                    torch.set_grad_enabled(training),
                    torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
                ):
                    outputs = model(batch["features"], batch["energy"], batch["valid"])
                    loss = segmentation_loss(outputs, batch, config.train)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; inspect audio and optimizer settings")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                frames = int(batch["valid"].sum())
                total += float(loss.detach()) * frames
                count += frames
            losses.append(total / count)
        scheduler.step(losses[1])
        policy, report, candidates = calibrate(predictor, val_records)
        config.cuts = policy
        totals = report["totals"]
        rank = (
            totals["protected_modified_seconds"],
            totals["damaged_protected_events"],
            -totals["eligible_removed_seconds"],
            losses[1],
        )
        improved = best_rank is None or rank < best_rank
        if improved:
            best_rank, stalled = rank, 0
        else:
            stalled += 1
        numpy_state = np.random.get_state()
        payload = {
            "format_version": 1,
            "architecture": ARCHITECTURE,
            "model_version": MODEL_VERSION,
            "software_version": __version__,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_rank": best_rank,
            "stalled": stalled,
            "manifest_sha256": manifest_sha256,
            "rng_torch": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "rng_loader": generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "validation": report,
        }
        save_checkpoint(output_dir / "latest.pt", payload)
        if improved:
            save_checkpoint(output_dir / "best.pt", payload)
        summary = {
            "epoch": epoch + 1,
            "train_loss": losses[0],
            "validation_loss": losses[1],
            "best": improved,
            "policy": asdict(policy),
            "validation": totals,
        }
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")
        (output_dir / "validation.json").write_text(
            json.dumps({"selected": report, "candidates": candidates}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary), flush=True)
        if stalled >= config.train.patience:
            break
    return output_dir / "best.pt"
