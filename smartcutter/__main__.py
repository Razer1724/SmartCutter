"""Command-line training, inference, annotation conversion and evaluation."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import MODEL_VERSION, __version__
from .config import Config, CutConfig
from .cuts import EditPlan, file_hash, render_file
from .data import import_pairs, read_manifest, split_records, write_manifest
from .engine import load_predictor, plan_file
from .evaluate import calibrate, evaluate
from .train import train


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parser():
    root = argparse.ArgumentParser(
        description=f"SmartCutter {MODEL_VERSION}: vocal-preserving audio cutting"
    )
    root.add_argument(
        "--version", action="version", version=f"SmartCutter {MODEL_VERSION} ({__version__})"
    )
    commands = root.add_subparsers(dest="command", required=True)
    config = commands.add_parser("init-config", help="Write a complete editable configuration")
    config.add_argument("output")
    config.add_argument("--temporal", choices=("conformer", "tcn"), default="conformer")
    imported = commands.add_parser(
        "import-pairs", help="Convert aligned raw/marked pairs to unreviewed interval annotations"
    )
    imported.add_argument("folder")
    imported.add_argument("output")
    imported.add_argument("--speaker-id", required=True)
    imported.add_argument("--source-prefix", required=True)
    imported.add_argument("--threshold", type=float, default=1e-5)
    split = commands.add_parser("split", help="Split complete speaker/source groups")
    split.add_argument("manifest")
    split.add_argument("output")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--test-fraction", type=float, default=0.15)
    validate = commands.add_parser("validate-data")
    validate.add_argument("manifest")
    training = commands.add_parser("train")
    training.add_argument("manifest")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--resume")
    training.add_argument("--device", default="auto")
    inference = commands.add_parser("infer")
    inference.add_argument("input", help="Audio file or directory of WAV/FLAC files")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True, help="Output directory")
    inference.add_argument("--policy", help="Calibrated CutConfig JSON override")
    inference.add_argument(
        "--dry-run", action="store_true", help="Write edit plans without rendering audio"
    )
    inference.add_argument("--overwrite", action="store_true")
    inference.add_argument("--device", default="auto")
    inference.add_argument("--legacy-sample-rate", type=int)
    rendering = commands.add_parser("render", help="Render a previously reviewed edit plan")
    rendering.add_argument("input")
    rendering.add_argument("plan")
    rendering.add_argument("output")
    rendering.add_argument("--overwrite", action="store_true")
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate one or more checkpoints on exactly the same recordings"
    )
    evaluation.add_argument("manifest")
    evaluation.add_argument("--checkpoint", action="append", required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--calibrate", action="store_true")
    evaluation.add_argument("--policy-out")
    evaluation.add_argument(
        "--policy",
        action="append",
        default=[],
        help="One policy JSON per checkpoint, in order; '-' uses its stored policy",
    )
    evaluation.add_argument("--legacy-sample-rate", type=int)
    evaluation.add_argument("--device", default="auto")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "device") and args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if args.command == "init-config":
            config = Config()
            config.model.temporal = args.temporal
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            write_json(args.output, config.to_dict())
        elif args.command == "import-pairs":
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            records = import_pairs(args.folder, args.speaker_id, args.source_prefix, args.threshold)
            write_manifest(records, args.output)
            print(
                f"Imported {len(records)} recordings. Review intervals before setting reviewed=true."
            )
        elif args.command == "split":
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            records = read_manifest(args.manifest, require_reviewed=False, check_splits=False)
            write_manifest(
                split_records(records, args.seed, args.validation_fraction, args.test_fraction),
                args.output,
            )
        elif args.command == "validate-data":
            records = read_manifest(args.manifest)
            print(
                json.dumps(
                    {
                        "recordings": len(records),
                        "splits": {
                            s: sum(r.split == s for r in records)
                            for s in ("train", "validation", "test", "unsplit")
                        },
                    }
                )
            )
        elif args.command == "train":
            records = read_manifest(args.manifest)
            config = Config.load(args.config)
            best = train(
                records, config, args.output, args.device, args.resume, file_hash(args.manifest)
            )
            print(f"Saved best checkpoint: {best}")
        elif args.command == "infer":
            predictor = load_predictor(args.checkpoint, args.device, args.legacy_sample_rate)
            policy = (
                CutConfig(**json.loads(Path(args.policy).read_text(encoding="utf-8")))
                if args.policy
                else None
            )
            source, output = Path(args.input), Path(args.output)
            files = (
                sorted(p for p in source.iterdir() if p.suffix.lower() in (".wav", ".flac"))
                if source.is_dir()
                else [source]
            )
            if not files:
                raise ValueError("No WAV/FLAC files found")
            output.mkdir(parents=True, exist_ok=True)
            for file in files:
                destination = output / file.name
                plan_path = output / (file.name + ".edits.json")
                if (
                    source.is_dir()
                    and source.resolve() == output.resolve()
                    or file.resolve() == destination.resolve()
                ):
                    raise ValueError("Choose an output directory different from the input")
                if not args.overwrite and (
                    plan_path.exists() or (destination.exists() and not args.dry_run)
                ):
                    raise FileExistsError(f"Output already exists for {file.name}")
                plan = plan_file(predictor, file, policy)
                plan.save(plan_path)
                if not args.dry_run:
                    render_file(file, destination, plan, args.overwrite)
                print(
                    json.dumps(
                        {
                            "input": str(file),
                            "cuts": len(plan.cuts),
                            "plan": str(plan_path),
                            "output": None if args.dry_run else str(destination),
                        }
                    )
                )
        elif args.command == "render":
            render_file(args.input, args.output, EditPlan.load(args.plan), args.overwrite)
        elif args.command == "evaluate":
            if args.calibrate and args.split != "validation":
                raise ValueError(
                    "Use --split validation for calibration; test data must remain held out"
                )
            if args.policy_out and (not args.calibrate or len(args.checkpoint) != 1):
                raise ValueError("--policy-out requires --calibrate and exactly one checkpoint")
            if args.policy and len(args.policy) != len(args.checkpoint):
                raise ValueError(
                    "Supply one --policy per checkpoint, using '-' for stored policies"
                )
            records = [r for r in read_manifest(args.manifest) if r.split == args.split]
            reports = []
            for index, checkpoint in enumerate(args.checkpoint):
                predictor = load_predictor(checkpoint, args.device, args.legacy_sample_rate)
                if args.policy and args.policy[index] != "-":
                    predictor.config.cuts = CutConfig(
                        **json.loads(Path(args.policy[index]).read_text(encoding="utf-8"))
                    )
                if args.calibrate:
                    policy, report, candidates = calibrate(predictor, records)
                    report = {**report, "calibration_candidates": candidates}
                    if args.policy_out:
                        write_json(args.policy_out, asdict(policy))
                else:
                    report = evaluate(predictor, records)[0]
                reports.append(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": file_hash(checkpoint),
                        **report,
                    }
                )
            write_json(
                args.output,
                {
                    "manifest_sha256": file_hash(args.manifest),
                    "split": args.split,
                    "models": reports,
                },
            )
            print(
                json.dumps(
                    {
                        "report": args.output,
                        "models": [
                            {"checkpoint": r["checkpoint"], "totals": r["totals"]} for r in reports
                        ],
                    }
                )
            )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"SmartCutter {MODEL_VERSION}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
