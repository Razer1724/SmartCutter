import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch
import torchaudio

import inference
from model_v3 import DSCA_ResUNet_v3
from smartcutter.engine import load_predictor
from train_v3 import get_file_pairs

torch.set_num_threads(2)


class FixedModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)


class LegacyRegressionTests(unittest.TestCase):
    def test_old_launcher_loads_best_checkpoint_and_preserves_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming, outgoing, checkpoints = root / "input", root / "output", root / "ckpts"
            incoming.mkdir()
            checkpoints.mkdir()
            audio = np.random.default_rng(1).uniform(-0.1, 0.1, (4000, 2))
            sf.write(incoming / "sample.wav", audio, 32000, subtype="PCM_16")
            torch.save(
                {"architecture": "v3", "state_dict": {}, "sample_rate": 32000},
                checkpoints / "v3_best_model_32000.pth",
            )
            with mock.patch.multiple(
                inference,
                IN_DIR=str(incoming),
                OUT_DIR=str(outgoing),
                CKPT_DIR=str(checkpoints),
                FORCE_CPU=True,
                DEBUG_MASK_PRED=False,
                SEGMENT_LEN=0.2,
                MODEL_VERSION="v3",
                DSCA_ResUNet_v3=lambda **kwargs: FixedModel(-20.0),
            ):
                inference.processing()
            self.assertTrue(
                np.array_equal(
                    sf.read(incoming / "sample.wav")[0], sf.read(outgoing / "sample.wav")[0]
                )
            )

    def test_v3_metadata_and_bare_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3.pt"
            state = DSCA_ResUNet_v3().state_dict()
            torch.save(
                {
                    "architecture": "v3",
                    "format_version": 1,
                    "sample_rate": 32000,
                    "frontend": "peak-cap-v1",
                    "state_dict": state,
                },
                checkpoint,
            )
            predictor = load_predictor(checkpoint)
            probabilities, _ = predictor.predict(torch.zeros(2, 4411), 44100)
            self.assertEqual(len(probabilities), 11)
            self.assertTrue(torch.isfinite(probabilities).all())
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "legacy-sample-rate"):
                load_predictor(checkpoint)
            predictor = load_predictor(checkpoint, legacy_sample_rate=32000)
            self.assertEqual(predictor.config.frontend.sample_rate, 32000)

    def test_no_cut_and_short_gap_keep_audio(self):
        audio = torch.randn(2, 32000) * 0.1
        for first, last in ((0, 0), (40, 45)):
            mask = torch.zeros(100)
            mask[first:last] = 1
            output, _ = inference.SmartCutter(audio, mask, 32000)
            self.assertTrue(torch.equal(output, audio))

    def test_snap_cannot_cross_predicted_bounds(self):
        audio = torch.full((1, 1000), 0.1)
        audio[:, 291:611] = -0.1
        mask = torch.zeros(1000)
        mask[300:600] = 1
        starts, ends = inference.find_cuts(mask, audio, 1000)
        self.assertGreaterEqual(starts[0], 305)
        self.assertLessEqual(ends[0], 595)

    def test_all_aggregation_modes_use_probabilities(self):
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, n_mels=128, hop_length=320, pad_mode="constant"
        )
        original = inference.MASK_MODE
        try:
            for mode in ("Soft", "Hard", "Hybrid", "PowerMean"):
                inference.MASK_MODE = mode
                for value in (-5.0, 0.4):
                    actual = inference._run_inference(
                        FixedModel(value),
                        transform,
                        torch.ones(1, 3200),
                        torch.device("cpu"),
                        torch.zeros(1, 2, 128, 11),
                    )
                    torch.testing.assert_close(
                        actual, torch.full_like(actual, torch.sigmoid(torch.tensor(value)).item())
                    )
        finally:
            inference.MASK_MODE = original

    def test_overlap_add_keeps_first_last_and_partial_frames(self):
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, n_mels=128, hop_length=320, pad_mode="constant"
        )
        old_seconds = inference.SEGMENT_LEN
        inference.SEGMENT_LEN = 0.2
        try:
            for samples in (1, 3201, 16017):
                result = inference.process_grid_aligned(
                    FixedModel(1.0),
                    transform,
                    torch.zeros(1, samples),
                    32000,
                    320,
                    torch.device("cpu"),
                    torch.zeros(1, 2, 128, 21),
                )
                self.assertEqual(result.shape[-1], math.ceil(samples / 320))
                torch.testing.assert_close(
                    result, torch.full_like(result, torch.sigmoid(torch.tensor(1.0)).item())
                )
        finally:
            inference.SEGMENT_LEN = old_seconds

    def test_pair_matching_rejects_missing_partner(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a_raw.wav", "b_raw.wav", "b_marked.wav", "c_marked.wav"):
                (Path(directory) / name).touch()
            with self.assertRaises(ValueError):
                get_file_pairs(directory)


if __name__ == "__main__":
    unittest.main()
