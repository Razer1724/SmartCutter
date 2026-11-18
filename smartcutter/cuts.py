"""Conservative, timestamp-based edit planning and native-format rendering."""

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import CutConfig


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


@dataclass
class Cut:
    start: int
    end: int
    approved_start: int
    approved_end: int
    fade_samples: int
    confidence: float
    detected_start: int | None = None
    detected_end: int | None = None


@dataclass
class EditPlan:
    sample_rate: int
    source_samples: int
    channels: int
    cuts: list = field(default_factory=list)
    source_sha256: str = ""
    format_version: int = 1

    def validate(self):
        if any(
            type(value) is not int
            for value in (self.sample_rate, self.source_samples, self.channels)
        ):
            raise ValueError("Edit plan sample counts and rate must be integers")
        if (
            self.format_version != 1
            or min(self.sample_rate, self.source_samples, self.channels) < 1
        ):
            raise ValueError("Invalid edit plan metadata")
        previous_end = 0
        previous_fade = 0
        for cut in self.cuts:
            if any(
                type(getattr(cut, key)) is not int
                for key in ("start", "end", "approved_start", "approved_end", "fade_samples")
            ):
                raise ValueError("Cut coordinates must be integer sample indices")
            if not (
                0
                <= cut.approved_start
                <= cut.start
                < cut.end
                <= cut.approved_end
                <= self.source_samples
            ):
                raise ValueError("Cut lies outside its approved interval")
            if cut.start < previous_end or cut.fade_samples < 0:
                raise ValueError("Cuts must be ordered, disjoint and have nonnegative fades")
            if (
                cut.start - cut.fade_samples < cut.approved_start
                or cut.end + cut.fade_samples > cut.approved_end
            ):
                raise ValueError("Fade would modify audio outside the approved gap")
            if cut.start - previous_end < cut.fade_samples + previous_fade:
                raise ValueError("Adjacent cut fades overlap")
            if not math.isfinite(cut.confidence) or not 0 <= cut.confidence <= 1:
                raise ValueError("Invalid cut confidence")
            if (cut.detected_start is None) != (cut.detected_end is None):
                raise ValueError("Both detected bounds must be supplied together")
            if cut.detected_start is not None and not (
                type(cut.detected_start) is int
                and type(cut.detected_end) is int
                and 0
                <= cut.detected_start
                <= cut.approved_start
                <= cut.approved_end
                <= cut.detected_end
                <= self.source_samples
            ):
                raise ValueError("Invalid detected gap bounds")
            previous_end, previous_fade = cut.end, cut.fade_samples
        return self

    def retained_spans(self):
        self.validate()
        cursor, output_cursor = 0, 0
        result = []
        for cut in self.cuts:
            if cut.start > cursor:
                result.append(
                    {
                        "source_start": cursor,
                        "source_end": cut.start,
                        "output_start": output_cursor,
                        "output_end": output_cursor + cut.start - cursor,
                    }
                )
                output_cursor += cut.start - cursor
            cursor = cut.end
        if cursor < self.source_samples:
            result.append(
                {
                    "source_start": cursor,
                    "source_end": self.source_samples,
                    "output_start": output_cursor,
                    "output_end": output_cursor + self.source_samples - cursor,
                }
            )
        return result

    def to_dict(self):
        self.validate()
        return {**asdict(self), "retained_spans": self.retained_spans()}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("retained_spans", None)
        data["cuts"] = [Cut(**c) for c in data["cuts"]]
        return cls(**data).validate()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inward_snap(waveform, start, end, radius):
    """Only shrink removal. Search all channels' amplitude and local slope."""
    audio = np.asarray(waveform)
    if audio.ndim == 1:
        audio = audio[None, :]

    def choose(lo, hi, target):
        indices = np.arange(max(0, lo), min(audio.shape[-1] - 1, hi) + 1)
        if len(indices) == 0:
            return target
        amplitude = np.abs(audio[:, indices]).max(axis=0)
        slope = np.abs(audio[:, indices] - audio[:, np.maximum(0, indices - 1)]).max(axis=0)
        cost = amplitude + 0.25 * slope + np.abs(indices - target) * 1e-10
        return int(indices[np.argmin(cost)])

    new_start = choose(start, min(start + radius, end - 1), start)
    new_end = choose(max(new_start + 1, end - radius), end, end)
    return new_start, new_end


def plan_cuts(
    probabilities,
    sample_rate,
    source_samples,
    channels,
    hop_seconds,
    config=None,
    boundaries=None,
    waveform=None,
):
    config = config or CutConfig()
    plan = EditPlan(sample_rate, source_samples, channels)
    if not config.enabled:
        return plan.validate()
    p = np.asarray(torch.as_tensor(probabilities).detach().cpu(), dtype=np.float64).reshape(-1)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or hop_seconds <= 0:
        raise ValueError("Expected finite probabilities in [0, 1] and a positive hop")
    expected = math.ceil(source_samples / sample_rate / hop_seconds - 1e-9)
    if len(p) != expected:
        raise ValueError(f"Expected {expected} timestamped frames, got {len(p)}")
    b = None if boundaries is None else np.asarray(torch.as_tensor(boundaries).detach().cpu())
    if b is not None and (
        b.shape != (len(p), 2) or not np.isfinite(b).all() or np.any((b < 0) | (b > 1))
    ):
        raise ValueError("Expected [frames, 2] boundary probabilities in [0, 1]")
    fade = round(sample_rate * config.fade_ms / 1000)
    padding = round(sample_rate * config.padding_ms / 1000)
    # Leave enough original room tone for both fades and the protected margins.
    retained = max(round(sample_rate * config.target_gap_ms / 1000), 2 * (padding + fade))
    confident = max(1, math.ceil(config.min_confident_ms / (hop_seconds * 1000)))
    boundary_window = max(1, math.ceil(config.snap_ms / (hop_seconds * 1000)))
    for first, last in runs(p >= config.exit_probability):
        if not any(e - s >= confident for s, e in runs(p[first:last] >= config.enter_probability)):
            continue
        if config.use_boundaries and b is not None:
            start_slice = b[first : min(last, first + boundary_window + 1), 0]
            end_slice = b[max(first, last - boundary_window - 1) : last, 1]
            if start_slice.max() >= config.boundary_probability:
                first += int(start_slice.argmax())
            if end_slice.max() >= config.boundary_probability:
                last = max(first + 1, last - len(end_slice) + int(end_slice.argmax()))
        gap_start = min(source_samples, math.ceil(first * hop_seconds * sample_rate))
        gap_end = min(source_samples, math.floor(last * hop_seconds * sample_rate))
        length = gap_end - gap_start
        if length < sample_rate * config.min_gap_ms / 1000 or length <= retained:
            continue
        start = gap_start + retained // 2
        end = gap_end - (retained - retained // 2)
        if waveform is not None and config.snap_ms > 0:
            start, end = _inward_snap(
                waveform, start, end, round(sample_rate * config.snap_ms / 1000)
            )
        if end <= start:
            continue
        plan.cuts.append(
            Cut(
                start,
                end,
                gap_start + padding,
                gap_end - padding,
                fade,
                float(p[first:last].min()),
                gap_start,
                gap_end,
            )
        )
    return plan.validate()


def _gain_for_span(start, end, lo, hi, fade_in, fade_out):
    indices = np.arange(start, end)
    gain = np.ones(end - start, dtype=np.float64)
    if fade_in:
        phase = np.clip((indices - lo) / max(1, fade_in - 1), 0, 1)
        gain *= 0.5 - 0.5 * np.cos(np.pi * phase)
    if fade_out:
        phase = np.clip((hi - 1 - indices) / max(1, fade_out - 1), 0, 1)
        gain *= 0.5 - 0.5 * np.cos(np.pi * phase)
    return gain


def _render_spans(plan):
    cursor, fade_in = 0, 0
    for cut in plan.cuts:
        if cursor < cut.start:
            yield cursor, cut.start, fade_in, cut.fade_samples
        cursor, fade_in = cut.end, cut.fade_samples
    if cursor < plan.source_samples:
        yield cursor, plan.source_samples, fade_in, 0


def render_audio(waveform, plan):
    """Tensor renderer used by integrations/tests. No normalization or channel mixing."""
    plan.validate()
    if waveform.shape != (plan.channels, plan.source_samples):
        raise ValueError("Audio shape does not match the edit plan")
    pieces = []
    for lo, hi, fade_in, fade_out in _render_spans(plan):
        gain = torch.as_tensor(
            _gain_for_span(lo, hi, lo, hi, fade_in, fade_out),
            device=waveform.device,
            dtype=waveform.dtype,
        )
        pieces.append(waveform[:, lo:hi] * gain)
    return torch.cat(pieces, dim=1) if pieces else waveform[:, :0].clone()


def render_file(source, destination, plan, overwrite=False):
    """Stream retained spans into an atomic output; preserve PCM subtype by default."""
    plan.validate()
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the source")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if plan.source_sha256 and file_hash(source) != plan.source_sha256:
        raise ValueError("Source hash differs from the previewed edit plan")
    info = sf.info(str(source))
    if (info.samplerate, info.frames, info.channels) != (
        plan.sample_rate,
        plan.source_samples,
        plan.channels,
    ):
        raise ValueError("Source metadata differs from the edit plan")
    fmt = {".wav": "WAV", ".flac": "FLAC"}.get(destination.suffix.lower())
    if not fmt or not sf.check_format(fmt, info.subtype):
        raise ValueError(
            f"Cannot preserve {info.subtype} in {destination.suffix}; use the original file format"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=destination.suffix, delete=False
    ) as temp:
        temporary = Path(temp.name)
    try:
        with (
            sf.SoundFile(str(source)) as reader,
            sf.SoundFile(
                str(temporary),
                "w",
                samplerate=info.samplerate,
                channels=info.channels,
                subtype=info.subtype,
                format=fmt,
            ) as writer,
        ):
            for lo, hi, fade_in, fade_out in _render_spans(plan):
                reader.seek(lo)
                position = lo
                while position < hi:
                    data = reader.read(min(65536, hi - position), dtype="float64", always_2d=True)
                    gain = _gain_for_span(position, position + len(data), lo, hi, fade_in, fade_out)
                    writer.write(data * gain[:, None])
                    position += len(data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
