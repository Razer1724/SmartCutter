"""Features retain absolute level; no per-file peak normalization is applied."""

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from .config import FrontendConfig


class AudioFrontend(nn.Module):
    def __init__(self, config: FrontendConfig):
        super().__init__()
        self.config = config
        self.transforms = nn.ModuleList()
        for window_ms in config.windows_ms:
            window = round(config.sample_rate * window_ms / 1000)
            fft = 2 ** (window - 1).bit_length()
            self.transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=config.sample_rate,
                    n_fft=fft,
                    win_length=window,
                    hop_length=config.hop_samples,
                    n_mels=config.n_mels,
                    normalized=True,
                    norm="slaney",
                    pad_mode="constant",
                )
            )

    def forward(self, audio):
        """[channels, samples] -> per-channel [C, resolutions, mel, T], [C, 2, T].

        Frame k represents [k * hop, (k + 1) * hop). Explicit timestamps,
        including the final partial frame, are used everywhere downstream.
        """
        if audio.ndim != 2 or audio.shape[-1] == 0:
            raise ValueError("Expected nonempty [channels, samples] audio")
        if not torch.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples")
        audio = audio.float()
        hop = self.config.hop_samples
        frames = (audio.shape[-1] + hop - 1) // hop
        features = []
        for transform in self.transforms:
            power = transform(audio)[..., :frames]
            db = 10 * torch.log10(power.clamp_min(1e-12))
            features.append((db.clamp(-100, 20) + 100) / 120)
        cells = F.pad(audio, (0, frames * hop - audio.shape[-1])).reshape(
            audio.shape[0], frames, hop
        )
        rms = cells.square().mean(-1).sqrt()
        peak = cells.abs().amax(-1)
        energy = torch.stack([rms, peak], dim=1)
        energy = (20 * torch.log10(energy.clamp_min(1e-6))).clamp(-100, 6) / 100
        return torch.stack(features, dim=1), energy


def pool_channels(features, energy):
    """Pool spectral power, never waveforms: opposing channel phases cannot cancel."""
    return features.amax(0, keepdim=True), energy.amax(0, keepdim=True)


def legacy_features(waveform, mel_transform):
    """Shared v3 preprocessing. This is versioned as peak-cap-v1 in new checkpoints."""
    peak = waveform.abs().amax()
    waveform = waveform * torch.clamp(0.95 / peak.clamp_min(1e-12), max=1.0)
    frames = (waveform.shape[-1] + mel_transform.hop_length - 1) // mel_transform.hop_length
    mel = torchaudio.transforms.AmplitudeToDB()(mel_transform(waveform))[..., :frames]
    mel = (mel.clamp(-80.0, 0.0) + 80.0) / 80.0
    delta = torchaudio.functional.compute_deltas(mel)
    return torch.stack([mel, delta], dim=1)
