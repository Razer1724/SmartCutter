"""
SmartCutter v7 model.

v3 was a 2D ResUNet that predicted ONE value per time frame, copied across all 160 mel rows,
and then averaged the rows back together at inference. Two thirds of its 13.7M parameters
sat in 256-512 channel blocks on a frequency axis that was thrown away.

v7 solves the 1D problem with a 1D model:

  log-mel [B,1,128,T] --2D stem (4x stride-(2,1))--> collapse frequency 128 -> 8 -> [B,256,T]
  hop-aligned time stats [B,3,T] --small conv--> fused in
  --> N dilated residual temporal blocks   (local, multi-scale, ~1.5 s radius)
  --> BiGRU at half frame-rate             (global context: "what is the noise floor / is this a pause")
  --> heads:  silence logit [B,T]   +   start/end edge logits [B,2,T]

Normalisation is deliberately NOT computed across time (BatchNorm in the 2D stem, per-frame
channel LayerNorm in the 1D parts). v3's GroupNorm + global pooling made a chunk's output depend on
the other content in the same chunk (an all-quiet chunk gets normalised "as if" it were typical);
here a frame's absolute level means the same thing in every chunk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from features_v7 import N_MELS, N_STATS

DEFAULT_CONFIG = dict(
    n_mels=N_MELS,
    stem_channels=(32, 48, 64, 96, 128),
    dim=256,
    stats_dim=64,
    tcn_kernel=5,
    tcn_dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8),
    gru_layers=2,
)


class ResBlock2d(nn.Module):
    """Residual 3x3 block that halves frequency (stride (2,1)) and never touches time resolution."""

    def __init__(self, cin, cout, stride=(2, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class FreqCollapseStem(nn.Module):
    def __init__(self, n_mels, channels, out_dim):
        super().__init__()
        assert n_mels % 2 ** (len(channels) - 1) == 0
        self.inc = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
                                 nn.BatchNorm2d(channels[0]), nn.GELU())
        self.stages = nn.ModuleList([ResBlock2d(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        f_out = n_mels // 2 ** (len(channels) - 1)
        self.proj = nn.Conv1d(channels[-1] * f_out, out_dim, 1)

    def forward(self, spec):                       # [B,1,F,T]
        x = self.inc(spec)
        for s in self.stages:
            x = s(x)
        b, c, f, t = x.shape
        return self.proj(x.reshape(b, c * f, t))   # [B,out_dim,T]


class ChannelLN(nn.Module):
    """LayerNorm over channels, independently at every time step (no statistics across time)."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):                          # [B,C,T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    def __init__(self, dim, kernel, dilation, expansion=2):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel - 1) // 2,
                            dilation=dilation, groups=dim)
        self.ln = ChannelLN(dim)
        self.pw1 = nn.Conv1d(dim, dim * expansion, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(dim * expansion, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), 0.1))

    def forward(self, x):
        y = self.pw2(self.act(self.pw1(self.ln(self.dw(x)))))
        return x + self.gamma * y


class BiGRUContext(nn.Module):
    """Global context at half frame-rate, added back as a residual."""

    def __init__(self, dim, layers):
        super().__init__()
        self.down = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.ln = ChannelLN(dim)
        self.gru = nn.GRU(dim, dim // 2, num_layers=layers, batch_first=True, bidirectional=True)
        self.up = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
        self.mix = nn.Conv1d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), 0.1))

    def forward(self, x):
        t = x.shape[-1]
        h = self.ln(self.down(x)).transpose(1, 2)
        h, _ = self.gru(h)
        h = self.up(h.transpose(1, 2))
        if h.shape[-1] < t:
            h = F.pad(h, (0, t - h.shape[-1]), mode="replicate")
        h = h[..., :t]
        return x + self.gamma * self.mix(h)


class SmartCutterV7(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        dim = cfg["dim"]

        self.stem = FreqCollapseStem(cfg["n_mels"], cfg["stem_channels"], dim)
        self.stats_embed = nn.Sequential(
            nn.Conv1d(N_STATS, cfg["stats_dim"], 3, padding=1), nn.GELU(),
            nn.Conv1d(cfg["stats_dim"], cfg["stats_dim"], 3, padding=1), nn.GELU())
        self.fuse = nn.Conv1d(dim + cfg["stats_dim"], dim, 1)
        self.tcn = nn.Sequential(*[TCNBlock(dim, cfg["tcn_kernel"], d) for d in cfg["tcn_dilations"]])
        self.context = BiGRUContext(dim, cfg["gru_layers"])
        self.out_ln = ChannelLN(dim)

        self.sil_head = nn.Sequential(nn.Conv1d(dim, dim // 2, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(dim // 2, 1, 1))
        self.edge_head = nn.Sequential(nn.Conv1d(dim, dim // 2, 3, padding=1), nn.GELU(),
                                       nn.Conv1d(dim // 2, 2, 1))

    def forward(self, spec, stats):
        x = self.stem(spec)
        x = self.fuse(torch.cat([x, self.stats_embed(stats)], dim=1))
        x = self.tcn(x)
        x = self.out_ln(self.context(x))
        return {"sil": self.sil_head(x).squeeze(1),      # [B,T]  logit: frame is cuttable silence
                "edge": self.edge_head(x)}               # [B,2,T] logits: silence start / silence end


def count_params(model):
    return sum(p.numel() for p in model.parameters())
