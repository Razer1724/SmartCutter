import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from smartcutter.config import Config, CutConfig
from smartcutter.cuts import Cut, EditPlan, file_hash, plan_cuts, render_audio, render_file
from smartcutter.data import (
    AudioDataset,
    Recording,
    collate_audio,
    import_pairs,
    make_targets,
    paired_files,
    read_manifest,
    split_records,
    write_manifest,
)
from smartcutter.engine import Predictor, load_predictor
from smartcutter.evaluate import calibrate, score_plan
from smartcutter.frontend import AudioFrontend, pool_channels
from smartcutter.model import TemporalSegmenter
from smartcutter.train import segmentation_loss, train

torch.set_num_threads(2)


def tiny_config(temporal="conformer"):
    return Config.from_dict(
        {
            "frontend": {"n_mels": 16, "windows_ms": [25, 40]},
            "model": {
                "width": 16,
                "layers": 1,
                "heads": 2,
                "conv_kernel": 5,
                "dropout": 0.0,
                "temporal": temporal,
            },
            "train": {
                "segment_seconds": 0.2,
                "batch_size": 2,
                "epochs": 1,
                "noise_probability": 0,
                "reverb_probability": 0,
            },
        }
    )


class CutTests(unittest.TestCase):
    def setUp(self):
        self.sr = 32000
        self.audio = torch.randn(2, self.sr) * 0.1
        self.policy = CutConfig(snap_ms=0)

    def plan(self, p, **kwargs):
        return plan_cuts(
            p, self.sr, self.audio.shape[-1], 2, 0.01, kwargs.pop("policy", self.policy), **kwargs
        )

    def test_no_cuts_preserves_every_sample(self):
        output = render_audio(self.audio, self.plan(torch.zeros(100)))
        self.assertTrue(torch.equal(output, self.audio))

    def test_short_and_uncertain_gaps_are_preserved(self):
        for value, first, last in [(1, 40, 45), (0.94, 20, 80)]:
            p = torch.zeros(100)
            p[first:last] = value
            self.assertEqual(self.plan(p).cuts, [])

    def test_requires_sustained_confidence(self):
        p = torch.full((100,), 0.93)
        p[30] = 1
        self.assertEqual(self.plan(p).cuts, [])

    def test_renderer_preserves_protected_samples_and_room_tone(self):
        p = torch.zeros(100)
        p[20:80] = 1
        plan = self.plan(p)
        self.assertEqual(len(plan.cuts), 1)
        cut = plan.cuts[0]
        output = render_audio(self.audio, plan)
        self.assertTrue(torch.equal(output[:, :6400], self.audio[:, :6400]))
        self.assertTrue(torch.equal(output[:, -6400:], self.audio[:, -6400:]))
        self.assertEqual(output.shape[-1], self.sr - cut.end + cut.start)
        self.assertGreaterEqual(19200 - (cut.end - cut.start), 3200)
        self.assertEqual(plan.retained_spans()[-1]["output_end"], output.shape[-1])

    def test_snap_cannot_expand_into_vocals(self):
        waveform = torch.full((1, 1000), 0.1)
        waveform[:, 291:611] = -0.1
        p = torch.zeros(100)
        p[30:60] = 1
        plan = plan_cuts(p, 1000, 1000, 1, 0.01, CutConfig(snap_ms=25), waveform=waveform.numpy())
        self.assertGreaterEqual(plan.cuts[0].start, 350)
        self.assertLessEqual(plan.cuts[0].end, 550)

    def test_boundaries_only_move_inward(self):
        p = torch.zeros(100)
        p[20:80] = 1
        boundary = torch.zeros(100, 2)
        boundary[21, 0] = 1
        boundary[79, 1] = 1
        plan = self.plan(p, policy=CutConfig(snap_ms=20), boundaries=boundary)
        self.assertGreaterEqual(plan.cuts[0].detected_start, 6400)
        self.assertLessEqual(plan.cuts[0].detected_end, 25600)

    def test_subframe_and_partial_frame_files(self):
        for sr, count in [(32000, 1), (44100, 44101), (48000, 48013)]:
            frames = math.ceil(count / sr / 0.01)
            plan = plan_cuts(torch.zeros(frames), sr, count, 1, 0.01)
            self.assertEqual(plan.cuts, [])

    def test_probability_and_grid_validation(self):
        for p in (torch.zeros(99), torch.full((100,), float("nan")), torch.full((100,), -1.0)):
            with self.assertRaises(ValueError):
                self.plan(p)

    def test_rejects_unsafe_edit_plan(self):
        with self.assertRaises(ValueError):
            EditPlan(1000, 1000, 1, [Cut(300, 600, 300, 600, 10, 0.99)]).validate()
        with self.assertRaises(ValueError):
            EditPlan(1000, 1000, 1, [Cut(300.5, 600, 250, 650, 10, 0.99)]).validate()


class FileTests(unittest.TestCase):
    def test_native_rate_channels_and_precision_survive_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.default_rng(1)
            audio = rng.uniform(-0.5, 0.5, (1001, 2))
            for subtype in ("PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
                source, output = directory / f"{subtype}.wav", directory / f"{subtype}_out.wav"
                sf.write(source, audio, 44100, subtype=subtype)
                plan = EditPlan(44100, len(audio), 2, source_sha256=file_hash(source))
                render_file(source, output, plan)
                self.assertTrue(np.array_equal(sf.read(source)[0], sf.read(output)[0]))
                info = sf.info(output)
                self.assertEqual(
                    (info.samplerate, info.channels, info.subtype), (44100, 2, subtype)
                )

    def test_streamed_fades_match_tensor_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "in.wav", directory / "out.wav"
            audio = np.random.default_rng(2).uniform(-0.5, 0.5, (180000, 2))
            sf.write(source, audio, 32000, subtype="DOUBLE")
            plan = EditPlan(32000, len(audio), 2, [Cut(70000, 100000, 69000, 101000, 160, 0.99)])
            render_file(source, output, plan)
            expected = render_audio(torch.from_numpy(audio.T.copy()), plan).numpy().T
            self.assertTrue(np.array_equal(sf.read(output)[0], expected))

    def test_edit_roundtrip_and_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "in.wav", directory / "out.wav"
            sf.write(source, np.zeros(100), 32000)
            plan = EditPlan(32000, 100, 1, source_sha256=file_hash(source))
            plan.save(directory / "edits.json")
            restored = EditPlan.load(directory / "edits.json")
            self.assertEqual(plan.to_dict(), restored.to_dict())
            sf.write(source, np.ones(100) * 0.1, 32000)
            with self.assertRaisesRegex(ValueError, "hash"):
                render_file(source, output, restored)


class ModelTests(unittest.TestCase):
    def test_short_input_and_antiphase_stereo(self):
        frontend = AudioFrontend(tiny_config().frontend)
        mono = torch.sin(torch.arange(3200) * 0.3).unsqueeze(0) * 0.1
        a, e = frontend(mono)
        b, f = pool_channels(*frontend(torch.cat([mono, -mono])))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(e, f)
        short, energy = frontend(torch.ones(1, 1))
        self.assertEqual(short.shape[-1], 1)
        self.assertTrue(torch.isfinite(energy).all())

    def test_both_architectures_forward_backward_and_padding(self):
        for temporal in ("conformer", "tcn"):
            config = tiny_config(temporal)
            model = TemporalSegmenter(config)
            features = torch.randn(2, 2, 17, 9)
            energy = torch.randn(2, 2, 9)
            valid = torch.arange(9)[None, :] < torch.tensor([5, 9])[:, None]
            outputs = model(features, energy, valid)
            batch = {"valid": valid, "cut": torch.zeros(2, 9), "boundaries": torch.zeros(2, 9, 2)}
            loss = segmentation_loss(outputs, batch, config.train)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.cut_head.weight.grad.abs().sum().item(), 0)
            model.eval()
            padded = model(features, energy, valid)
            single = model(features[:1, :, :, :5], energy[:1, :, :5])
            torch.testing.assert_close(padded["cut"][:1, :5], single["cut"], atol=2e-6, rtol=2e-6)

    def test_protected_errors_cost_more(self):
        cfg = tiny_config().train
        batch = {
            "valid": torch.ones(1, 2, dtype=torch.bool),
            "cut": torch.tensor([[0.0, 1.0]]),
            "boundaries": torch.zeros(1, 2, 2),
        }
        bad_vocal = {"cut": torch.tensor([[5.0, 5.0]]), "boundaries": torch.zeros(1, 2, 2)}
        extra_silence = {"cut": torch.tensor([[-5.0, -5.0]]), "boundaries": torch.zeros(1, 2, 2)}
        self.assertGreater(
            segmentation_loss(bad_vocal, batch, cfg), segmentation_loss(extra_silence, batch, cfg)
        )

    def test_any_channel_can_veto_removal(self):
        class Model(torch.nn.Module):
            def forward(self, features, energy):
                batch, frames = features.shape[0], features.shape[-1]
                cut = torch.full((batch, frames), 20.0)
                cut[-1] = -20
                return {"cut": cut, "boundaries": torch.zeros(batch, frames, 2)}

        predictor = Predictor(Model(), tiny_config())
        p, _ = predictor.predict(torch.ones(2, 3200), 32000)
        self.assertLess(float(p.max()), 1e-6)

    def test_chunk_blending_covers_endpoints_and_tails(self):
        class ConstantPredictor(Predictor):
            def _chunk(self, chunk):
                n = math.ceil(chunk.shape[-1] / self.config.frontend.hop_samples)
                return torch.full((n,), 0.75), torch.full((n, 2), 0.25)

        predictor = ConstantPredictor(torch.nn.Identity(), tiny_config())
        for sr, samples in ((32000, 1), (44100, 44101), (48000, 48013)):
            p, b = predictor.predict(torch.zeros(1, samples), sr)
            self.assertEqual(len(p), math.ceil(samples / sr / 0.01))
            torch.testing.assert_close(p, torch.full_like(p, 0.75))
            torch.testing.assert_close(b, torch.full_like(b, 0.25))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        sf.write(self.audio, np.zeros(9600), 32000, subtype="FLOAT")

    def tearDown(self):
        self.temp.cleanup()

    def record(self, **kwargs):
        values = dict(
            id="a",
            audio=str(self.audio),
            speaker_id="s",
            source_id="source",
            split="train",
            remove=[[0.1, 0.2]],
            reviewed=True,
        )
        values.update(kwargs)
        return Recording(**values)

    def test_missing_or_duplicate_partners_fail(self):
        (self.root / "a_raw.wav").touch()
        (self.root / "b_marked.wav").touch()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired_files(self.root)
        (self.root / "a_raw.flac").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_files(self.root)

    def test_legacy_import_requires_alignment_and_review(self):
        raw = np.zeros(32000)
        raw[16000:] = 0.2
        marked = raw.copy()
        marked[3200:6400] = 0.1
        sf.write(self.root / "a_raw.wav", raw, 32000, subtype="FLOAT")
        sf.write(self.root / "a_marked.wav", marked, 32000, subtype="FLOAT")
        records = import_pairs(self.root, "speaker", "session")
        self.assertEqual(records[0].remove, [[0.1, 0.2]])
        self.assertFalse(records[0].reviewed)
        write_manifest(records, self.root / "manifest.jsonl")
        with self.assertRaisesRegex(ValueError, "review"):
            read_manifest(self.root / "manifest.jsonl")
        sf.write(self.root / "a_marked.wav", marked[:-1], 32000)
        with self.assertRaisesRegex(ValueError, "identical"):
            import_pairs(self.root, "speaker", "session")

    def test_manifest_rejects_leakage_and_protected_overlap(self):
        for records in (
            [self.record(), self.record(id="b", split="test")],
            [self.record(), self.record(id="b", speaker_id="other", split="test")],
            [self.record(protected=[{"start": 0.15, "end": 0.25, "kind": "breath"}])],
        ):
            write_manifest(records, self.root / "manifest.jsonl")
            with self.assertRaises(ValueError):
                read_manifest(self.root / "manifest.jsonl")

    def test_split_keeps_connected_groups_together(self):
        records = [self.record(id=str(i), speaker_id=f"s{i}", source_id=f"r{i}") for i in range(7)]
        records[1].speaker_id = records[0].speaker_id
        records[2].source_id = records[1].source_id
        split_records(records)
        self.assertEqual(records[0].split, records[2].split)
        write_manifest(records, self.root / "manifest.jsonl")
        self.assertEqual(len(read_manifest(self.root / "manifest.jsonl")), 7)

    def test_short_files_and_tail_crops_are_included(self):
        dataset = AudioDataset([self.record()], tiny_config())
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[1][0].shape[-1], 10)
        batch = collate_audio([dataset[0], dataset[1]])
        self.assertEqual(batch["valid"].sum().item(), 30)
        self.assertEqual(batch["features"].shape, (2, 2, 16, 20))

    def test_labels_protect_partial_cells_and_reverb_tails(self):
        cut, _ = make_targets([[0.015, 0.08]], 0, 0.1, 10, 0.01)
        self.assertEqual(cut.tolist(), [0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        tailed, _ = make_targets([[0.02, 0.08]], 0, 0.1, 10, 0.01, 0.03)
        self.assertEqual(tailed[:5].sum(), 0)
        self.assertEqual(tailed.sum(), 3)


class EvaluationTests(unittest.TestCase):
    def test_calibration_selects_useful_safe_cuts_and_falls_back_on_damage(self):
        class FixedPredictor:
            config = Config()
            calls = 0
            unsafe = False

            def predict(self, waveform, sr):
                self.calls += 1
                p = torch.zeros(100)
                p[20:80] = 0.96
                if self.unsafe:
                    p[:] = 1
                return p, torch.zeros(100, 2)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "val.wav"
            sf.write(source, np.zeros(32000), 32000)
            record = Recording(
                "val", str(source), "s", "r", "validation", [[0.2, 0.8]], reviewed=True
            )
            predictor = FixedPredictor()
            policy, report, _ = calibrate(predictor, [record])
            self.assertTrue(policy.enabled)
            self.assertGreater(report["totals"]["eligible_removed_seconds"], 0.4)
            self.assertEqual(predictor.calls, 1)
            predictor.unsafe = True
            policy, report, _ = calibrate(predictor, [record])
            self.assertFalse(policy.enabled)
            self.assertEqual(report["totals"]["protected_modified_seconds"], 0)

    def test_scores_fade_damage_and_excludes_safe_removal(self):
        record = Recording(
            "id",
            "unused",
            "s",
            "r",
            "test",
            [[0.3, 0.6]],
            [{"start": 0.2, "end": 0.3, "kind": "vocal"}],
            True,
        )
        safe = EditPlan(1000, 1000, 1, [Cut(350, 550, 325, 575, 5, 0.99)])
        result = score_plan(safe, record)
        self.assertEqual(result["damaged_protected_events"], 0)
        self.assertAlmostEqual(result["eligible_removed_seconds"], 0.2)
        damaged = EditPlan(1000, 1000, 1, [Cut(300, 550, 290, 575, 5, 0.99)])
        result = score_plan(damaged, record)
        self.assertEqual(result["damaged_protected_events"], 1)
        self.assertAlmostEqual(result["protected_modified_seconds"], 0.005)

    def test_calibration_refuses_test_data(self):
        record = Recording("id", "unused", "s", "r", "test", [])
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate(None, [record])


class IntegrationTests(unittest.TestCase):
    def test_train_checkpoint_inference_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            records = []
            for i, split in enumerate(("train", "validation", "test")):
                path = directory / f"{split}.wav"
                waveform = np.zeros(9600, dtype=np.float32)
                waveform[:3200] = np.sin(np.arange(3200) * 0.2) * 0.1
                sf.write(path, waveform, 32000, subtype="FLOAT")
                records.append(
                    Recording(str(i), str(path), str(i), str(i), split, [[0.1, 0.3]], reviewed=True)
                )
            config = tiny_config()
            best = train(records, config, directory / "run", manifest_sha256="synthetic")
            predictor = load_predictor(best)
            p, b = predictor.predict(torch.zeros(2, 4411), 44100)
            self.assertEqual((p.shape, b.shape), (torch.Size([11]), torch.Size([11, 2])))
            self.assertTrue(torch.isfinite(p).all())
            config.train.epochs = 2
            train(
                records,
                config,
                directory / "run",
                resume=directory / "run/latest.pt",
                manifest_sha256="synthetic",
            )
            checkpoint = torch.load(directory / "run/latest.pt", weights_only=True)
            self.assertEqual(checkpoint["architecture"], "v9")
            self.assertEqual(checkpoint["model_version"], "v9")
            self.assertEqual(checkpoint["software_version"], "9.0.0")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("optimizer", checkpoint)
            uninterrupted = tiny_config()
            uninterrupted.train.epochs = 2
            train(records, uninterrupted, directory / "continuous", manifest_sha256="synthetic")
            reference = torch.load(directory / "continuous/latest.pt", weights_only=True)
            for key, value in reference["state_dict"].items():
                torch.testing.assert_close(checkpoint["state_dict"][key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
