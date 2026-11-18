import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from smartcutter.config import Config
from smartcutter.data import Recording, write_manifest
from smartcutter.model import TemporalSegmenter


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config.from_dict(
            {
                "frontend": {"n_mels": 16, "windows_ms": [25]},
                "model": {"width": 16, "layers": 1, "heads": 2, "conv_kernel": 5, "dropout": 0},
                "train": {"segment_seconds": 0.2},
            }
        )
        model = TemporalSegmenter(self.config)
        with torch.no_grad():
            model.cut_head.weight.zero_()
            model.cut_head.bias.fill_(-20)
        self.checkpoint = self.root / "fixture.pt"
        torch.save(
            {
                "format_version": 1,
                "architecture": "temporal_segmenter",
                "state_dict": model.state_dict(),
                "config": self.config.to_dict(),
            },
            self.checkpoint,
        )
        self.audio = self.root / "input.wav"
        data = np.random.default_rng(42).uniform(-0.1, 0.1, (4411, 2))
        sf.write(self.audio, data, 44100, subtype="PCM_24")

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *arguments, code=0):
        result = subprocess.run(
            [sys.executable, "-m", "smartcutter", *map(str, arguments)],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_preview_render_batch_and_overwrite_protection(self):
        preview = self.root / "preview"
        self.cli(
            "infer", self.audio, "--checkpoint", self.checkpoint, "--output", preview, "--dry-run"
        )
        edl = preview / "input.wav.edits.json"
        self.assertTrue(edl.exists())
        self.assertFalse((preview / "input.wav").exists())
        output = self.root / "rendered.wav"
        self.cli("render", self.audio, edl, output)
        self.assertTrue(np.array_equal(sf.read(self.audio)[0], sf.read(output)[0]))
        self.cli("render", self.audio, edl, output, code=2)
        self.cli(
            "infer", self.audio, "--checkpoint", self.checkpoint, "--output", self.root / "batch"
        )
        self.assertEqual(sf.info(self.root / "batch/input.wav").subtype, "PCM_24")

    def test_calibration_report_serializes_and_test_set_stays_held_out(self):
        manifest = self.root / "manifest.jsonl"
        write_manifest(
            [
                Recording(
                    "validation",
                    str(self.audio),
                    "speaker",
                    "source",
                    "validation",
                    [],
                    reviewed=True,
                )
            ],
            manifest,
        )
        report, policy = self.root / "report.json", self.root / "policy.json"
        self.cli(
            "evaluate",
            manifest,
            "--checkpoint",
            self.checkpoint,
            "--split",
            "validation",
            "--calibrate",
            "--policy-out",
            policy,
            "--output",
            report,
        )
        result = json.loads(report.read_text())
        self.assertEqual(len(result["models"][0]["calibration_candidates"]), 6)
        self.assertFalse(json.loads(policy.read_text())["enabled"])
        self.cli(
            "evaluate",
            manifest,
            "--checkpoint",
            self.checkpoint,
            "--calibrate",
            "--output",
            report,
            code=2,
        )
        self.cli(
            "evaluate",
            manifest,
            "--checkpoint",
            self.checkpoint,
            "--split",
            "validation",
            "--policy",
            policy,
            "--output",
            report,
        )

    def test_init_config_help_and_missing_checkpoint_errors(self):
        version = self.cli("--version")
        self.assertEqual(version.stdout.strip(), "SmartCutter v9 (9.0.0)")
        self.cli("--help")
        config = self.root / "new.json"
        self.cli("init-config", config, "--temporal", "tcn")
        self.assertEqual(Config.load(config).model.temporal, "tcn")
        self.cli("init-config", config, code=2)
        self.cli(
            "infer",
            self.audio,
            "--checkpoint",
            self.root / "missing.pt",
            "--output",
            self.root / "out",
            code=2,
        )


if __name__ == "__main__":
    unittest.main()
