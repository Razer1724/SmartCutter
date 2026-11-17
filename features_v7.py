"""
SmartCutter v7 - shared feature extraction.

Training and inference BOTH import this file, so the two can never disagree about
how audio becomes model input (v3 had this logic duplicated in two places).

Sample-rate independence
------------------------
The detector always runs at INTERNAL_SR (32 kHz) with a fixed 100 frames/second
grid. Any input is resampled to INTERNAL_SR for detection only; the resulting
frame mask is a function of TIME, so it maps back onto the original audio at any
sample rate, and the actual cutting is done on the untouched original samples.
Frame t is centred at t / 100 seconds.

Calibration
-----------
The log-mel is normalised per band so that stationary white noise of RMS X dBFS
reads as roughly X dB in every band. DB_FLOOR..DB_CEIL (-120..-20 dBFS) is mapped
to 0..1. Exact digital silence reads exactly 0.0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

# ----------------------------------------------------------------------------- constants
INTERNAL_SR = 32000
FPS = 100                              # frames per second (10 ms hop)
HOP = INTERNAL_SR // FPS               # 320
N_FFT = 1024
N_MELS = 128
F_MIN = 0.0
F_MAX = INTERNAL_SR / 2
DB_FLOOR = -120.0
DB_CEIL = -20.0
ZERO_EPS = 1e-9                        # |x| <= this counts as an exact digital zero
SEGMENT_SEC = 8.0
SEG_SAMPLES = int(SEGMENT_SEC * INTERNAL_SR)   # 256000 (multiple of HOP)
SEG_FRAMES = SEG_SAMPLES // HOP + 1            # 801 (STFT is centred)
PEAK_TARGET = 0.9                      # file-level peak normalisation (train == infer)
N_STATS = 3

assert SEG_SAMPLES % HOP == 0


# ----------------------------------------------------------------------------- resampling
_RESAMPLERS = {}


def resample(x, sr_from, sr_to):
    """x: [..., N] float tensor. Cached polyphase resampler (exact zeros stay exact zeros)."""
    if int(sr_from) == int(sr_to):
        return x
    key = (int(sr_from), int(sr_to))
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(int(sr_from), int(sr_to))
    r = _RESAMPLERS[key]
    if r.kernel.device != x.device:
        r.to(x.device)
    return r(x)


def fit_length(x, n):
    """Crop / zero-pad the last dim to exactly n samples."""
    cur = x.shape[-1]
    if cur == n:
        return x
    if cur > n:
        return x[..., :n]
    return F.pad(x, (0, n - cur))


def norm_db(db):
    return ((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------- features
class FeatureExtractor(nn.Module):
    """
    wave [B, N] (N multiple of HOP, at INTERNAL_SR)  ->
        spec  [B, 1, N_MELS, T]   calibrated log-mel in 0..1
        stats [B, 3, T]           hop-aligned time-domain stats: rms dB, peak dB, exact-zero fraction
    with T = N / HOP + 1 for both (frame t centred on sample t * HOP).

    The stats channels exist because a 1024-sample STFT window smears 32 ms while the
    labels need ~10 ms precision; counting exact zeros / measuring level inside each
    10 ms hop gives the network sample-accurate evidence about where silence begins.
    """

    def __init__(self):
        super().__init__()
        win = torch.hann_window(N_FFT, periodic=True)
        fb = AF.melscale_fbanks(N_FFT // 2 + 1, F_MIN, F_MAX, N_MELS, INTERNAL_SR,
                                norm=None, mel_scale="htk")            # [n_freq, n_mels]
        norm = (win ** 2).sum() * fb.sum(0).clamp_min(1e-6)            # white-noise equivalent
        self.register_buffer("window", win, persistent=False)
        self.register_buffer("fb_t", fb.t().contiguous(), persistent=False)   # [n_mels, n_freq]
        self.register_buffer("norm", norm.view(1, N_MELS, 1), persistent=False)

    @torch.no_grad()
    def forward(self, wave):
        wave = wave.float()
        if wave.dim() == 3:
            wave = wave.squeeze(1)
        assert wave.shape[-1] % HOP == 0, "length must be a multiple of HOP"

        with torch.autocast(device_type=wave.device.type, enabled=False):
            stft = torch.stft(wave, N_FFT, HOP, window=self.window, center=True,
                              pad_mode="reflect", return_complex=True)
            power = stft.real ** 2 + stft.imag ** 2                    # [B, F, T]
            mel = torch.matmul(self.fb_t, power) / self.norm           # [B, M, T]
            spec = norm_db(10.0 * torch.log10(mel.clamp_min(1e-12))).unsqueeze(1)

            pad = HOP // 2
            blocks = F.pad(wave, (pad, pad)).unfold(1, HOP, HOP)       # [B, T, HOP], centred on t*HOP
            rms_db = 10.0 * torch.log10((blocks ** 2).mean(-1).clamp_min(1e-12))
            peak_db = 20.0 * torch.log10(blocks.abs().amax(-1).clamp_min(1e-6))
            zero_frac = (blocks.abs() <= ZERO_EPS).float().mean(-1)
            stats = torch.stack([norm_db(rms_db), norm_db(peak_db), zero_frac], dim=1)

        T = min(spec.shape[-1], stats.shape[-1])
        return spec[..., :T], stats[..., :T]
