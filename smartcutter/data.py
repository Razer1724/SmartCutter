"""Explicit interval annotations, checked legacy imports, and lazy audio crops."""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from .cuts import runs
from .frontend import AudioFrontend, pool_channels


@dataclass
class Recording:
    id: str
    audio: str
    speaker_id: str
    source_id: str
    split: str
    remove: list
    protected: list = field(default_factory=list)
    reviewed: bool = False


def intersection_length(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def read_manifest(path, require_reviewed=True, check_splits=True):
    path = Path(path).resolve()
    records, ids = [], set()
    groups = {"speaker_id": {}, "source_id": {}}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = Recording(**json.loads(line))
            record.audio = str((path.parent / record.audio).resolve())
            if not all((record.id, record.speaker_id, record.source_id)) or record.id in ids:
                raise ValueError("IDs must be nonempty and recording IDs must be unique")
            ids.add(record.id)
            if record.split not in ("train", "validation", "test", "unsplit"):
                raise ValueError("split must be train, validation, test, or unsplit")
            if require_reviewed and record.reviewed is not True:
                raise ValueError(
                    "Annotation needs review; set reviewed=true after listening and checking intervals"
                )
            info = sf.info(record.audio)
            if info.frames < 1:
                raise ValueError("Empty recording")
            duration = info.frames / info.samplerate
            previous = 0.0
            for interval in record.remove:
                if len(interval) != 2:
                    raise ValueError("remove intervals must be [start_seconds, end_seconds]")
                start, end = interval
                if not (
                    math.isfinite(start)
                    and math.isfinite(end)
                    and previous <= start < end <= duration + 1e-9
                ):
                    raise ValueError(
                        "remove intervals must be finite, ordered, disjoint and within the audio"
                    )
                previous = end
            for event in record.protected:
                start, end = event["start"], event["end"]
                if not (
                    math.isfinite(start)
                    and math.isfinite(end)
                    and 0 <= start < end <= duration + 1e-9
                ):
                    raise ValueError("Invalid protected event")
                if any(
                    intersection_length((start, end), interval) > 0 for interval in record.remove
                ):
                    raise ValueError("A removable interval overlaps a protected vocal event")
            if check_splits:
                for key, assignments in groups.items():
                    group = getattr(record, key)
                    if group in assignments and assignments[group] != record.split:
                        raise ValueError(f"{key}={group!r} leaks across dataset splits")
                    assignments[group] = record.split
            records.append(record)
        except (TypeError, KeyError, ValueError, RuntimeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError("Manifest is empty")
    return records


def write_manifest(records, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            value = dict(vars(record))
            # Absolute paths keep relocated manifests correct; users may make them relative later.
            value["audio"] = str(Path(record.audio).resolve())
            stream.write(json.dumps(value) + "\n")


def paired_files(folder):
    folder = Path(folder)
    groups = {"raw": {}, "marked": {}}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".wav", ".flac"):
            continue
        for kind in groups:
            suffix = "_" + kind
            if path.stem.endswith(suffix):
                key = path.stem[: -len(suffix)]
                if key in groups[kind]:
                    raise ValueError(f"Duplicate {kind} partner for {key}")
                groups[kind][key] = path
    if groups["raw"].keys() != groups["marked"].keys():
        missing_marked = sorted(groups["raw"].keys() - groups["marked"].keys())
        missing_raw = sorted(groups["marked"].keys() - groups["raw"].keys())
        raise ValueError(
            f"Unpaired files: missing marked={missing_marked}, missing raw={missing_raw}"
        )
    return [(str(groups["raw"][key]), str(groups["marked"][key])) for key in sorted(groups["raw"])]


def import_pairs(folder, speaker_id, source_prefix, threshold=1e-5, merge_holes_ms=1.0):
    if threshold <= 0 or merge_holes_ms < 0:
        raise ValueError("Invalid import threshold or merge duration")
    result = []
    for raw_path, marked_path in paired_files(folder):
        raw, sr = sf.read(raw_path, dtype="float64", always_2d=True)
        marked, marked_sr = sf.read(marked_path, dtype="float64", always_2d=True)
        if sr != marked_sr or raw.shape != marked.shape or not len(raw):
            raise ValueError(
                f"{raw_path}: paired files must have identical rate, channel count and length"
            )
        if not np.isfinite(raw).all() or not np.isfinite(marked).all():
            raise ValueError(f"{raw_path}: non-finite samples")
        mask = np.abs(raw - marked).max(axis=1) > threshold
        if mask.mean() > 0.8:
            raise ValueError(
                f"{raw_path}: >80% changed; check gain, alignment and annotation convention"
            )
        intervals = []
        tolerance = round(sr * merge_holes_ms / 1000)
        for start, end in runs(mask):
            if intervals and start - intervals[-1][1] <= tolerance:
                intervals[-1][1] = end
            else:
                intervals.append([start, end])
        stem = Path(raw_path).stem[:-4]
        result.append(
            Recording(
                stem,
                str(Path(raw_path).resolve()),
                speaker_id,
                f"{source_prefix}:{stem}",
                "unsplit",
                [[s / sr, e / sr] for s, e in intervals],
                reviewed=False,
            )
        )
    if not result:
        raise ValueError("No *_raw / *_marked WAV or FLAC pairs found")
    return result


def split_records(records, seed=42, validation_fraction=0.15, test_fraction=0.15):
    """Connected speaker/source groups always stay together."""
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError(
            "Split fractions must leave nonempty train, validation and test partitions"
        )
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for i, record in enumerate(records):
        for key in (("speaker", record.speaker_id), ("source", record.source_id)):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(root(i), []).append(record)
    values = list(groups.values())
    if len(values) < 3:
        raise ValueError(
            "At least three independent speaker/source groups are needed for three splits"
        )
    random.Random(seed).shuffle(values)
    n_val = max(1, round(len(values) * validation_fraction))
    n_test = max(1, round(len(values) * test_fraction))
    if n_val + n_test >= len(values):
        raise ValueError("Requested fractions leave no training group")
    for i, group in enumerate(values):
        split = "validation" if i < n_val else "test" if i < n_val + n_test else "train"
        for record in group:
            record.split = split
    return records


def make_targets(intervals, offset, duration, frames, hop, tail_seconds=0.0):
    starts = offset + torch.arange(frames, dtype=torch.float64) * hop
    ends = torch.minimum(starts + hop, torch.tensor(offset + duration, dtype=torch.float64))
    cut = torch.zeros(frames)
    boundaries = torch.zeros(frames, 2)
    grid = torch.arange(frames, dtype=torch.float64)
    for start, end in intervals:
        # Added reverberation extends the preceding vocal tail into the gap.
        start += tail_seconds
        if end <= start:
            continue
        cut[(starts >= start - 1e-9) & (ends <= end + 1e-9)] = 1
        for channel, point in enumerate((start, end)):
            local = point - offset
            if 0 <= local < duration:
                bump = torch.exp(-0.5 * ((grid - local / hop) / 1.5) ** 2).float()
                boundaries[:, channel] = torch.maximum(boundaries[:, channel], bump)
    return cut, boundaries


def augment_audio(audio, config):
    gain = torch.empty(()).uniform_(*config.gain_db).item()
    audio = audio * (10 ** (gain / 20))
    tail = 0.0
    if torch.rand(()) < config.reverb_probability:
        tail = 0.15
        # The dataset applies the impulse response at the configured analysis rate.
    if torch.rand(()) < config.noise_probability:
        noise = torch.randn_like(audio)
        noise = F.avg_pool1d(F.pad(noise.unsqueeze(0), (3, 3)), 7, stride=1).squeeze(0)
        scale = max(audio.square().mean().sqrt().item(), 1e-4) * 10 ** (
            -torch.empty(()).uniform_(20, 50).item() / 20
        )
        audio = audio + noise / noise.square().mean().sqrt().clamp_min(1e-8) * scale
    return audio, tail


class AudioDataset(Dataset):
    def __init__(self, records, config, augment=False):
        self.records, self.config, self.augment = records, config, augment
        self.frontend = AudioFrontend(config.frontend)
        self.items = []
        for record in records:
            info = sf.info(record.audio)
            size = max(1, round(info.samplerate * config.train.segment_seconds))
            stride = max(1, size // 2) if augment else size
            for start in range(0, info.frames, stride):
                self.items.append((record, start, min(size, info.frames - start), info.samplerate))
        if not self.items:
            raise ValueError("No audio crops in this split")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, start, count, native_sr = self.items[index]
        with sf.SoundFile(record.audio) as stream:
            stream.seek(start)
            audio = torch.from_numpy(stream.read(count, dtype="float32", always_2d=True).T.copy())
        sr = self.config.frontend.sample_rate
        if native_sr != sr:
            audio = torchaudio.functional.resample(audio, native_sr, sr)
        tail = 0.0
        if self.augment:
            audio, tail = augment_audio(audio, self.config.train)
            if tail:
                length = max(2, round(tail * sr))
                impulse = torch.zeros(length)
                impulse[0] = 1
                positions = torch.randint(1, length, (12,))
                impulse.index_add_(
                    0, positions, torch.randn(12) * 0.15 * torch.exp(-positions / (length / 4))
                )
                fft_size = 2 ** (audio.shape[-1] + length - 2).bit_length()
                audio = torch.fft.irfft(
                    torch.fft.rfft(audio, n=fft_size) * torch.fft.rfft(impulse, n=fft_size),
                    n=fft_size,
                )[..., : audio.shape[-1]]
        with torch.no_grad():
            features, energy = pool_channels(*self.frontend(audio))
        cut, boundaries = make_targets(
            record.remove,
            start / native_sr,
            count / native_sr,
            features.shape[-1],
            self.config.frontend.hop_seconds,
            tail,
        )
        return features[0], energy[0], cut, boundaries


def collate_audio(items):
    longest = max(item[0].shape[-1] for item in items)
    features, energies, cuts, boundaries, valid = [], [], [], [], []
    for feature, energy, cut, boundary in items:
        frames = feature.shape[-1]
        features.append(F.pad(feature, (0, longest - frames)))
        energies.append(F.pad(energy, (0, longest - frames)))
        cuts.append(F.pad(cut, (0, longest - frames)))
        boundaries.append(F.pad(boundary, (0, 0, 0, longest - frames)))
        valid.append(torch.arange(longest) < frames)
    return {
        "features": torch.stack(features),
        "energy": torch.stack(energies),
        "cut": torch.stack(cuts),
        "boundaries": torch.stack(boundaries),
        "valid": torch.stack(valid),
    }
