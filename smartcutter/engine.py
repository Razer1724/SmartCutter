"""Checkpoint loading and overlapping inference with complete frame coverage."""

import math

import numpy as np
import soundfile as sf
import torch
import torchaudio

from . import COMPATIBLE_ARCHITECTURES
from .config import Config
from .cuts import file_hash, plan_cuts
from .frontend import AudioFrontend, legacy_features, pool_channels
from .model import SmartCutterV9


class Predictor:
    def __init__(self, model, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.frontend = AudioFrontend(config.frontend).to(self.device)

    def _chunk(self, chunk):
        features, energy = self.frontend(chunk.to(self.device))
        pooled, pooled_energy = pool_channels(features, energy)
        if features.shape[0] > 1:
            # A protected prediction in any channel vetoes removal, including
            # anti-phase stereo and vocals audible only in a quieter channel.
            features = torch.cat([pooled, features], dim=0)
            energy = torch.cat([pooled_energy, energy], dim=0)
        outputs = self.model(features, energy)
        return outputs["cut"].sigmoid().amin(dim=0), outputs["boundaries"].sigmoid().amax(dim=0)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate):
        self.model.eval()
        if waveform.ndim != 2 or waveform.shape[-1] == 0 or sample_rate <= 0:
            raise ValueError("Expected nonempty channel-first audio and a positive sample rate")
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains non-finite samples")
        audio = waveform.float().cpu()
        analysis_rate = self.config.frontend.sample_rate
        if sample_rate != analysis_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, analysis_rate)
        hop = self.config.frontend.hop_samples
        frames = math.ceil(audio.shape[-1] / hop)
        chunk_frames = max(
            2, round(self.config.train.segment_seconds / self.config.frontend.hop_seconds)
        )
        stride = max(1, chunk_frames // 2)
        scores = torch.zeros(frames)
        boundaries = torch.zeros(frames, 2)
        weights = torch.zeros(frames)
        # A positive floor covers the first frame, final tail and very short files.
        window = torch.hann_window(chunk_frames, periodic=False).clamp_min(0.05)
        for first in range(0, frames, stride):
            last = min(frames, first + chunk_frames)
            chunk = audio[:, first * hop : min(audio.shape[-1], last * hop)]
            cut, boundary = self._chunk(chunk)
            length = last - first
            cut, boundary = cut[:length].cpu(), boundary[:length].cpu()
            weight = window[:length]
            scores[first:last] += cut * weight
            boundaries[first:last] += boundary * weight[:, None]
            weights[first:last] += weight
        if not torch.all(weights > 0):
            raise RuntimeError("Internal error: an inference frame has no prediction")
        return scores / weights, boundaries / weights[:, None]

    def plan(self, waveform, sample_rate, policy=None):
        probabilities, boundaries = self.predict(waveform, sample_rate)
        return plan_cuts(
            probabilities,
            sample_rate,
            waveform.shape[-1],
            waveform.shape[0],
            self.config.frontend.hop_seconds,
            policy or self.config.cuts,
            boundaries,
            waveform.cpu().numpy(),
        )


class LegacyPredictor(Predictor):
    def __init__(self, state, sample_rate, device="cpu"):
        from model_v3 import DSCA_ResUNet_v3

        if sample_rate not in (32000, 40000, 48000):
            raise ValueError("Specify the v3 training rate: 32000, 40000 or 48000")
        config = Config.from_dict({"frontend": {"sample_rate": sample_rate}})
        super().__init__(DSCA_ResUNet_v3(), config, device)
        self.model.load_state_dict(state)
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048 if sample_rate in (40000, 48000) else 1024,
            n_mels=160 if sample_rate in (40000, 48000) else 128,
            hop_length=sample_rate // 100,
            pad_mode="constant",
        ).to(self.device)

    def _chunk(self, chunk):
        features = legacy_features(chunk.to(self.device), self.transform)
        probabilities = self.model(features).sigmoid().mean(dim=2).squeeze(1).amin(dim=0)
        frames = math.ceil(chunk.shape[-1] / self.config.frontend.hop_samples)
        return probabilities[:frames], torch.zeros(frames, 2, device=self.device)


def load_predictor(checkpoint, device="cpu", legacy_sample_rate=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint")
    if payload.get("architecture") in COMPATIBLE_ARCHITECTURES:
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported v9 checkpoint format")
        config = Config.from_dict(payload["config"])
        model = SmartCutterV9(config)
        model.load_state_dict(payload["state_dict"])
        return Predictor(model, config, device)
    if payload.get("architecture") == "v3":
        return LegacyPredictor(payload["state_dict"], payload["sample_rate"], device)
    if legacy_sample_rate is None:
        raise ValueError("A bare v3 state_dict requires --legacy-sample-rate")
    return LegacyPredictor(payload, legacy_sample_rate, device)


def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(audio.T)), sr


def plan_file(predictor, path, policy=None):
    waveform, sr = read_audio(path)
    plan = predictor.plan(waveform, sr, policy)
    plan.source_sha256 = file_hash(path)
    return plan
