"""Frequency encoder followed by a temporal Conformer or a dilated TCN."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import Config


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def feed_forward(width, dropout):
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, width * 4),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(width * 4, width),
        nn.Dropout(dropout),
    )


class ConformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.ff1 = feed_forward(d, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d)
        self.point_in = nn.Conv1d(d, 2 * d, 1)
        self.depthwise = nn.Conv1d(d, d, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=d)
        self.depth_norm = nn.LayerNorm(d)
        self.point_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ff2 = feed_forward(d, cfg.dropout)
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        keep = valid.unsqueeze(-1)
        x = (x + 0.5 * self.ff1(x)) * keep
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=~valid, need_weights=False)
        x = (x + self.dropout(y)) * keep
        y = self.conv_norm(x).transpose(1, 2)
        y = F.glu(self.point_in(y), dim=1) * valid.unsqueeze(1)
        y = self.depthwise(y).transpose(1, 2)
        y = self.point_out(F.silu(self.depth_norm(y)))
        x = (x + self.dropout(y)) * keep
        return self.final_norm(x + 0.5 * self.ff2(x)) * keep


class TemporalConvBlock(nn.Module):
    def __init__(self, cfg, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.width)
        self.conv = nn.Conv1d(
            cfg.width, cfg.width, 5, dilation=dilation, padding=2 * dilation, groups=cfg.width
        )
        self.ff = feed_forward(cfg.width, cfg.dropout)

    def forward(self, x, valid):
        keep = valid.unsqueeze(-1)
        y = self.conv((self.norm(x) * keep).transpose(1, 2)).transpose(1, 2)
        x = (x + F.silu(y)) * keep
        return (x + self.ff(x)) * keep


class SmartCutterV9(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        channels = [len(config.frontend.windows_ms), 32, 64, 96]
        self.spectral = nn.ModuleList(
            nn.Sequential(nn.Conv2d(a, b, 3, stride=(2, 1), padding=1), ChannelNorm(b), nn.SiLU())
            for a, b in zip(channels[:-1], channels[1:])
        )
        self.frequency_attention = nn.Conv2d(channels[-1], 1, 1)
        # Preserve absolute band identity when pooling frequency. Without this,
        # similar local patterns in rumble and fricative bands become ambiguous.
        self.frequency_position = nn.Parameter(
            torch.randn(1, channels[-1], (config.frontend.n_mels + 7) // 8, 1) * 0.02
        )
        self.project = nn.Linear(channels[-1] + 2, config.model.width)
        self.blocks = nn.ModuleList(
            ConformerBlock(config.model)
            if config.model.temporal == "conformer"
            else TemporalConvBlock(config.model, 2 ** (i % 6))
            for i in range(config.model.layers)
        )
        self.cut_head = nn.Linear(config.model.width, 1)
        self.boundary_head = nn.Linear(config.model.width, 2)

    def forward(self, features, energy, valid=None):
        if valid is None:
            valid = torch.ones(
                features.shape[0], features.shape[-1], dtype=torch.bool, device=features.device
            )
        if not valid.any(dim=1).all():
            raise ValueError("Each item needs at least one valid frame")
        keep = valid[:, None, None, :]
        x = features * keep
        for layer in self.spectral:
            x = layer(x) * keep
        frequency_position = F.interpolate(
            self.frequency_position, size=(x.shape[2], 1), mode="bilinear", align_corners=False
        )
        x = (x + frequency_position) * keep
        attention = self.frequency_attention(x).softmax(dim=2)
        x = (x * attention).sum(dim=2).transpose(1, 2)
        x = self.project(torch.cat([x, energy.transpose(1, 2)], dim=-1))
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, x.shape[2], 2, device=x.device) * (-math.log(10000.0) / x.shape[2])
        )
        angles = positions * scale
        positional = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2).to(x.dtype)
        x = (x + positional) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid)
        return {"cut": self.cut_head(x).squeeze(-1), "boundaries": self.boundary_head(x)}


# Import compatibility for integrations written before this release was named v9.
TemporalSegmenter = SmartCutterV9
