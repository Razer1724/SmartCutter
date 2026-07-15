import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

        self.silu = nn.SiLU(inplace=True)

    def forward(self, g, x):
        # g: Gating Signal (from Decoder)
        # x: Skip Connection (from Encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.silu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class CoordinateAttention(nn.Module):
    # Lightweight attention mechanism.
    # Pools features along Frequency (H) and Time (W) separately.
    # This helps the model capture long-range dependencies in both directions.
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Pool Frequency
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Pool Time

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv1d(in_channels, mip, kernel_size=7, stride=1, padding=3, bias=False)

        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=mip)

        self.act = nn.SiLU()

        self.conv_h = nn.Conv1d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv1d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool separately then concatenate to process spatial info together.
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y.squeeze(-1))
        y = self.gn1(y)
        y = self.act(y) 

        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Compute attention maps for H and W axes.
        a_h = self.conv_h(x_h).sigmoid().unsqueeze(-1)
        a_w = self.conv_w(x_w).sigmoid().unsqueeze(-1).permute(0, 1, 3, 2)

        return identity * a_h * a_w


class ResBlock(nn.Module):
    # Standard Residual Block enhanced with Coordinate Attention.
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.stride = stride 

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # The attention layer added at the end of the block.
        self.attn = CoordinateAttention(out_channels)
        self.silu = nn.SiLU(inplace=True)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.attn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.silu(out)
        return out


class _TemporalTransformerBlock(nn.Module):
    # One lightweight pre-norm self-attention block over the time axis.
    # Uses GroupNorm instead of the usual LayerNorm to stay consistent with the
    # rest of this codebase (GroupNorm is used everywhere else here) -- with
    # num_groups=8 it behaves close enough to LayerNorm-per-group while keeping
    # a single norm implementation across the whole model.
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=dim)
        # Small FFN (2x expansion, not the usual 4x) -- keeps this a light
        # refinement step over the pooled sequence, not a full transformer layer.
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: (B, C, W) -- GroupNorm wants channels-first, MultiheadAttention wants
        # channels-last, so we permute in and out around each sub-layer.
        residual = x
        h = self.norm1(x).permute(0, 2, 1)          # (B, W, C)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = residual + attn_out.permute(0, 2, 1)     # (B, C, W)

        residual = x
        h = self.norm2(x).permute(0, 2, 1)           # (B, W, C)
        h = self.ffn(h)
        x = residual + h.permute(0, 2, 1)             # (B, C, W)
        return x


class TemporalAttentionBridge(nn.Module):
    # Drop-in replacement for DilatedBridge: keeps the same dilated-conv local
    # path (captures local spectral/temporal texture) and adds a second path
    # that does real self-attention over time, so the bottleneck can finally
    # relate frames that are far apart in W -- something CoordinateAttention's
    # pool-then-gate trick and the dilated convs' limited receptive field can't do.
    def __init__(self, in_channels=512, out_channels=512, attn_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        # --- Local path (unchanged from DilatedBridge) ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.local_gn = nn.GroupNorm(num_groups=8, num_channels=out_channels * 3)
        self.local_out = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        # --- Temporal attention path ---
        # Learned attention pooling over frequency (H): a 1x1 conv predicts a
        # per-pixel importance score, softmax-normalized across H, so collapsing
        # to a per-timestep sequence can learn to weight informative frequency
        # bands (e.g. voiced harmonics) higher than flat/noisy ones, instead of
        # just averaging everything together.
        self.freq_score = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Project down to a smaller working dim before attention -- MultiheadAttention's
        # in/out projections cost O(dim^2), so running at in_channels (512) directly
        # would dominate the parameter budget for little benefit at this bottleneck size.
        self.in_proj = nn.Conv1d(in_channels, attn_dim, kernel_size=1)
        self.attn_dim = attn_dim

        self.blocks = nn.ModuleList([
            _TemporalTransformerBlock(attn_dim, num_heads) for _ in range(num_layers)
        ])

        # Project refined temporal context to FiLM (scale, shift) parameters for
        # the local path. FiLM was chosen over concat+1x1 because it fuses the two
        # paths via broadcasting (B,C,1,W) against (B,C,H,W) directly -- no explicit
        # H-times replication of the temporal features needed, which keeps memory
        # flat regardless of how large H is, and it's a smaller/cheaper projection
        # (attn_dim -> out_channels*2) than concatenating full-size feature maps.
        self.film_proj = nn.Conv1d(attn_dim, out_channels * 2, kernel_size=1)

        # Zero-init so the bridge starts out equivalent to the local-only path
        # (gamma=0 -> scale=1, beta=0). The attention branch has to earn its
        # influence during training instead of injecting noise from step one.
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.silu = nn.SiLU(inplace=True)

    def _positional_encoding(self, w, device, dtype):
        # Standard sinusoidal encoding, built fresh for the actual sequence length
        # every forward pass -- W is not fixed (8s training chunks vs arbitrary
        # WOLA inference chunks), so nothing here assumes a max length.
        position = torch.arange(w, device=device, dtype=dtype).unsqueeze(1)          # (W, 1)
        div_term = torch.exp(
            torch.arange(0, self.attn_dim, 2, device=device, dtype=dtype) *
            (-math.log(10000.0) / self.attn_dim)
        )
        pe = torch.zeros(w, self.attn_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe                                                                     # (W, attn_dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # Local dilated-conv path
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x4 = self.conv4(x)
        local = torch.cat([x1, x2, x4], dim=1)
        local = self.silu(self.local_out(self.local_gn(local)))

        # Temporal attention path
        freq_w = self.freq_score(x)                       # (B, 1, H, W)
        freq_w = torch.softmax(freq_w, dim=2)              # normalize across frequency
        pooled = (x * freq_w).sum(dim=2)                   # (B, C, W) attention-pooled over H

        t = self.in_proj(pooled)                           # (B, attn_dim, W)
        pe = self._positional_encoding(w, x.device, x.dtype)
        t = t + pe.transpose(0, 1).unsqueeze(0)             # broadcast pos enc over batch

        for block in self.blocks:
            t = block(t)

        film = self.film_proj(t)                           # (B, out_channels*2, W)
        gamma, beta = film.chunk(2, dim=1)                  # each (B, out_channels, W)
        gamma = gamma.unsqueeze(2)                          # (B, out_channels, 1, W) -> broadcasts over H
        beta = beta.unsqueeze(2)

        out = local * (1 + gamma) + beta
        return self.silu(out)


class CGTA_ResUNet(nn.Module):
    # Same as CGA_ResUNet, except the bottleneck (self.bridge) is a
    # TemporalAttentionBridge instead of a DilatedBridge -- everything else
    # (encoder, decoder, attention gates) is untouched.
    def __init__(self, n_channels=2, n_classes=1):
        super(CGTA_ResUNet, self).__init__()

        # Initial Feature Extraction ( encoder )
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(inplace=True)
        )

        # Encoder Path
        self.enc1 = ResBlock(32, 64, stride=2)   
        self.enc2 = ResBlock(64, 128, stride=2)  
        self.enc3 = ResBlock(128, 256, stride=(2, 1)) 
        self.enc4 = ResBlock(256, 512, stride=(2, 1)) 

        # Bridge (Bottleneck) -- Temporal Attention instead of Dilated ASPP
        self.bridge = TemporalAttentionBridge(512, 512)

        # Decoder with attention gates
        # Up 4
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True)
        )
        # Gate 4: Filters 'enc3' (256 ch) using 'up4' (256 ch)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = ResBlock(512, 256)

        # Up 3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(inplace=True)
        )
        # Gate 3: Filters 'enc2' (128 ch) using 'up3' (128 ch)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResBlock(256, 128)

        # Up 2: Symmetric (2, 2) -> Use PixelShuffle
        self.up2_conv = nn.Conv2d(128, 64 * 4, kernel_size=1)
        self.up2_ps = nn.PixelShuffle(2)

        # Gate 2: Filters 'enc1' (64 ch) using 'up2' (64 ch)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = ResBlock(128, 64)

        # Up 1
        self.up1_conv = nn.Conv2d(64, 32 * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(2)
        # Gate 1: Filters 'x1' (32 ch) using 'up1' (32 ch)
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = ResBlock(64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        e1 = self.enc1(x1) 
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        e4 = self.enc4(e3)

        b = self.bridge(e4)

        # Decoder 4
        d4 = self.up4(b)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='nearest')

        # Attention gate 4
        x4_gated = self.att4(g=d4, x=e3)
        d4 = torch.cat([x4_gated, d4], dim=1)

        d4 = self.dec4(d4)

        # Decoder 3
        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='nearest')

        # Attention gate 3
        x3_gated = self.att3(g=d3, x=e2)
        d3 = torch.cat([x3_gated, d3], dim=1)

        d3 = self.dec3(d3)

        # Decoder 2
        d2 = self.up2_conv(d3)
        d2 = self.up2_ps(d2)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='nearest')

        # Attention gate 2
        x2_gated = self.att2(g=d2, x=e1)
        d2 = torch.cat([x2_gated, d2], dim=1)

        d2 = self.dec2(d2)

        # Decoder 1
        d1 = self.up1_conv(d2)
        d1 = self.up1_ps(d1)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='nearest')

        # Attention gate 1
        x1_gated = self.att1(g=d1, x=x1)
        d1 = torch.cat([x1_gated, d1], dim=1)

        d1 = self.dec1(d1)

        # Output
        logits = self.outc(d1)

        return logits


if __name__ == "__main__":
    # Quick sanity check: parameter count vs DilatedBridge, and forward passes
    # at a few different (H, W) combinations to confirm nothing is hardcoded.
    from model_v5 import DilatedBridge

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    old_bridge = DilatedBridge(512, 512)
    new_bridge = TemporalAttentionBridge(512, 512)

    old_n = count_params(old_bridge)
    new_n = count_params(new_bridge)
    print(f"[PARAMS] DilatedBridge:          {old_n:,}")
    print(f"[PARAMS] TemporalAttentionBridge: {new_n:,}  (+{new_n - old_n:,}, {new_n / old_n:.2f}x)")

    model = CGTA_ResUNet(n_channels=2, n_classes=1)
    total_n = count_params(model)
    print(f"[PARAMS] CGTA_ResUNet total:      {total_n:,}")

    # N_MELS is 160 for 48k/40k and 128 for 32k in this repo -- test both, plus
    # a handful of W values (short chunk, ~8s chunk, an odd/non-round length).
    for h in (128, 160):
        for w in (37, 199, 801):
            for b in (1, 2):
                x = torch.randn(b, 2, h, w)
                with torch.no_grad():
                    y = model(x)
                assert y.shape == (b, 1, h, w), f"shape mismatch: in={x.shape} out={y.shape}"
                print(f"[OK] B={b} H={h} W={w} -> {tuple(y.shape)}")

    print("\nAll shape checks passed.")
