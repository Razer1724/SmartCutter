"""Serializable configuration shared by training, evaluation and inference."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FrontendConfig:
    sample_rate: int = 32000
    hop_ms: float = 10.0
    windows_ms: tuple = (25.0, 80.0)
    n_mels: int = 96

    def __post_init__(self):
        self.windows_ms = tuple(self.windows_ms)
        if (
            not math.isfinite(self.hop_ms)
            or self.sample_rate < 8000
            or self.hop_ms <= 0
            or self.n_mels < 8
            or self.hop_samples < 1
        ):
            raise ValueError("Invalid frontend dimensions")
        if not self.windows_ms or any(
            not math.isfinite(w) or w < self.hop_ms for w in self.windows_ms
        ):
            raise ValueError("Analysis windows must be at least one hop long")

    @property
    def hop_samples(self):
        return round(self.sample_rate * self.hop_ms / 1000)

    @property
    def hop_seconds(self):
        return self.hop_samples / self.sample_rate


@dataclass
class ModelConfig:
    temporal: str = "conformer"
    width: int = 192
    layers: int = 6
    heads: int = 4
    conv_kernel: int = 31
    dropout: float = 0.1

    def __post_init__(self):
        if self.temporal not in ("conformer", "tcn"):
            raise ValueError("temporal must be conformer or tcn")
        if self.width < 8 or self.heads < 1 or self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if self.layers < 1 or self.conv_kernel < 3 or self.conv_kernel % 2 == 0:
            raise ValueError("Use positive layers and an odd convolution kernel >= 3")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")


@dataclass
class CutConfig:
    enabled: bool = True
    enter_probability: float = 0.98
    exit_probability: float = 0.90
    min_confident_ms: float = 30.0
    min_gap_ms: float = 180.0
    target_gap_ms: float = 100.0
    padding_ms: float = 25.0
    fade_ms: float = 5.0
    snap_ms: float = 10.0
    boundary_probability: float = 0.5
    use_boundaries: bool = True

    def __post_init__(self):
        if not 0 <= self.exit_probability <= self.enter_probability <= 1:
            raise ValueError("Require 0 <= exit_probability <= enter_probability <= 1")
        if not 0 <= self.boundary_probability <= 1:
            raise ValueError("Invalid boundary probability")
        for key in (
            "min_confident_ms",
            "min_gap_ms",
            "target_gap_ms",
            "padding_ms",
            "fade_ms",
            "snap_ms",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")


@dataclass
class TrainConfig:
    segment_seconds: float = 8.0
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    protected_weight: float = 5.0
    boundary_weight: float = 0.3
    workers: int = 0
    seed: int = 42
    patience: int = 15
    gain_db: tuple = (-18.0, 6.0)
    noise_probability: float = 0.3
    reverb_probability: float = 0.2

    def __post_init__(self):
        self.gain_db = tuple(self.gain_db)
        if self.segment_seconds < 0.1 or min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("Invalid training sizes")
        if (
            self.learning_rate <= 0
            or self.protected_weight < 1
            or self.boundary_weight < 0
            or self.workers < 0
        ):
            raise ValueError("Invalid training hyperparameters")
        if len(self.gain_db) != 2 or self.gain_db[0] > self.gain_db[1]:
            raise ValueError("gain_db must contain an ordered pair")
        if not 0 <= self.noise_probability <= 1 or not 0 <= self.reverb_probability <= 1:
            raise ValueError("Augmentation probabilities must be in [0, 1]")


@dataclass
class Config:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cuts: CutConfig = field(default_factory=CutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {"frontend", "model", "cuts", "train"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        return cls(
            **{
                k: t(**value.get(k, {}))
                for k, t in (
                    ("frontend", FrontendConfig),
                    ("model", ModelConfig),
                    ("cuts", CutConfig),
                    ("train", TrainConfig),
                )
            }
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)
