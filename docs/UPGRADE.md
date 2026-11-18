# SmartCutter v9: upgrading v3 for vocal preservation

The v9 temporal pipeline is implemented independently of the v5/v6 architectures.
It must be trained before use. There is no bundled pretrained temporal model,
v3 checkpoint, or real evaluation dataset. Synthetic tests verify program
behavior; they do not establish better perceptual quality.

## Installation

Use Python 3.10–3.12. From the repository root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1

# CPU build; select a matching PyTorch CUDA build if training on an NVIDIA GPU.
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m smartcutter --help
```

The existing Windows `install.bat` environment can also run the new modules.
`run_train_v9.bat` and `run_infer_v9.bat` use that environment and the
paths shown below. The original launchers remain v3 compatibility entry points.

## Annotation contract

Create a JSONL manifest, one complete recording per line. Times are seconds,
interval ends are exclusive, and paths resolve relative to the manifest.

```json
{"id":"session-001","audio":"audio/session-001.wav","speaker_id":"speaker-01","source_id":"session-001","split":"train","remove":[[0.0,0.8],[2.1,3.4]],"protected":[{"start":0.8,"end":2.1,"kind":"speech"},{"start":3.4,"end":3.7,"kind":"breath"}],"reviewed":true}
```

Every sample outside `remove` is protected by default. Explicit `protected`
events identify the vocal events used for event-damage metrics. When those
events are absent, the evaluator counts the contiguous protected regions.
Protected events must never overlap removable intervals. Annotation applies to
the whole recording across all channels: only annotate a gap as removable if
every channel is safe to shorten. Keep whispers, breaths, humming, fricatives,
quiet consonants, and desired reverb/vocal tails outside `remove`.

`speaker_id` and `source_id` are real grouping identifiers. Clips derived from
one source must share the same source ID. A speaker or source cannot occur in
multiple splits. There is no automatic leakage check based on the audio itself;
accurate identifiers matter. The manifest must be reviewed before training.

The example above is a schema illustration; supply your own audio and labels.
Build a difficult held-out test collection before choosing model variants.
Include low gain, different microphones, noise, singing, breaths, short words,
room reverberation, native sample rates, and stereo with vocals in only one
channel. Include recordings with no removable gaps and fully silent recordings.

## Import existing raw/marked files

```sh
python -m smartcutter import-pairs training_files data/imported.jsonl --speaker-id speaker-01 --source-prefix session-family-01
```

The converter matches exact stems, permits WAV/FLAC partners, and rejects
missing/duplicate partners, mismatched rate/channel/length, and non-finite audio.
It derives intervals where samples differ by more than `1e-5`, joining holes of
at most 1 ms to accommodate a noise-marker annotation. It rejects pairs changed
over more than 80% of the file because gain changes or misalignment may otherwise
label everything as removable. Those checks cannot prove semantic correctness.

Imported annotations have `reviewed=false`. Listen and check each interval
before setting that field to true. Use the appropriate speaker and source IDs
for each recording; run imports separately for different speakers, concatenate
their JSONL records, then correct any shared source identifiers. A source pair
containing over 80% genuinely annotated silence should be annotated directly.

## Split and validate

```sh
python -m smartcutter split data/reviewed.jsonl data/manifest.jsonl --seed 42
python -m smartcutter validate-data data/manifest.jsonl
```

The split command keeps connected speaker/source groups together and requires
at least three independent groups. Default fractions are 15% validation and 15%
test by group count. Check the resulting duration and content balance. Manual
assignments are also supported, subject to the same leakage checks. Training
requires nonempty `train` and `validation` splits. Evaluation uses `test` by
default; threshold calibration is restricted to `validation`.

## Train and resume

```sh
python -m smartcutter train data/manifest.jsonl --config configs/v9.json --output runs/v9 --device cuda
```

Use `--device cpu` for CPU or omit it for automatic CPU/CUDA selection. Initial
settings are experiments, not validated universal defaults. Reduce batch size
for limited GPU memory. Short recordings and final partial crops are included.

The default model uses 32 kHz analysis, 25/80 ms spectral windows, 96 mel bins,
10 ms frame spacing, a frequency-only CNN encoder, attention pooling across
frequency, and six temporal Conformer blocks of width 192 with four heads.
It predicts a removable-gap probability plus start/end boundary probabilities.
It retains temporal resolution throughout the network. RMS/peak features retain
absolute level. There is no inference-only peak normalization.

Training pools channel features without waveform cancellation. Inference also
evaluates individual channels and takes the minimum removal probability, so
any channel can veto a cut. This conservative veto may retain additional noise.

Training includes gain variation, colored noise, and synthetic reverberation.
Reverb augmentation also extends protected tails in the targets. These simple
augmentations complement real difficult recordings; they cannot replace them.

The objective weights incorrectly removing protected audio five times as much
as retaining removable silence, with an auxiliary focal boundary loss. Padded
frames do not contribute to attention keys or losses. On every epoch, full
validation recordings are decoded across a threshold sweep. Selection maximizes
eligible silence removed subject to zero measured protected modification on
validation; if no useful candidate is safe, the saved policy disables cutting.
Validation loss breaks ties and drives the learning-rate scheduler.

Runs contain `best.pt`, `latest.pt`, `history.jsonl`, and `validation.json`.
Checkpoints record the architecture, complete frontend/configuration, calibrated
policy, model/optimizer/scheduler/scaler states, epoch, RNG states, and manifest
hash. Increase `train.epochs` in the configuration before continuing a completed
run:

```sh
python -m smartcutter train data/manifest.jsonl --config configs/v9.json --output runs/v9 --resume runs/v9/latest.pt
```

An existing run is never silently replaced. Changed architecture/frontend or
manifest is rejected on resume. For different experiments, use a new run path.

## Evaluate and compare

```sh
python -m smartcutter evaluate data/manifest.jsonl --checkpoint runs/v9/best.pt --split test --output reports/v9.json
```

Reports include protected seconds removed, protected seconds modified by cuts
or fades, damaged protected events, eligible silence removed, gap recall, and
mean/p95 boundary error for matched gaps. Boundary errors compare detected gap
edges before safety padding against annotated gap edges; misses are reported
through gap recall. Explicit-event metrics cover the events you annotated;
sample-damage metrics also cover all other protected samples. Timing includes
file reads and planning, and calibration timing includes all candidate policies.

Compare the supplied controlled variants using the same manifest and seed:

```sh
python -m smartcutter train data/manifest.jsonl --config configs/tcn.json --output runs/tcn
python -m smartcutter train data/manifest.jsonl --config configs/single_resolution.json --output runs/single
python -m smartcutter train data/manifest.jsonl --config configs/no_boundaries.json --output runs/no-boundaries
python -m smartcutter evaluate data/manifest.jsonl --checkpoint runs/v9/best.pt --checkpoint runs/tcn/best.pt --checkpoint runs/single/best.pt --checkpoint runs/no-boundaries/best.pt --output reports/comparison.json
```

Listen to retained beginnings/endings and joins, including failed and uncertain
examples. Promote a model only if it improves eligible removal at the same or
lower vocal-damage rate. Zero observed validation damage is not a guarantee on
new recordings. No empirical superiority is claimed by this implementation.

## Preview, render, and preserve original audio

```sh
python -m smartcutter infer infer_input --checkpoint runs/v9/best.pt --output previews --dry-run
python -m smartcutter render infer_input/example.wav previews/example.wav.edits.json infer_output/example.wav

# Or plan and render a batch together:
python -m smartcutter infer infer_input --checkpoint runs/v9/best.pt --output infer_output
```

Each edit list includes exact source sample indices, approved regions, fade
lengths, detected boundaries, confidence, source SHA-256, and a source-to-output
time map. The renderer verifies the source hash. It writes to a temporary file
and only then replaces the destination; existing output requires `--overwrite`.

The planner requires sustained confidence, uses hysteresis, and retains
uncertain regions. Boundaries and waveform refinement can only shrink the
removal interval. Padding protects transitions, and fades must remain inside
the approved gap. Short gaps are never lengthened. The remaining pause contains
original room tone, rather than generated samples. Its duration is at least the
configured target (possibly longer due to padding and inward refinement).

The original sample rate, all channels, and PCM/float subtype are retained for
WAV/FLAC output. Audio outside edits/fades is not normalized. Container metadata
tags are not copied. Analysis currently holds each recording in CPU memory;
GPU inference runs in overlapping chunks, and output rendering streams retained
spans. A no-cut file preserves decoded sample values, not identical container
bytes. The output container must support the original subtype.

## Repaired v3 baseline

The original v3 architecture remains intact. Its legacy training/inference now
share `peak-cap-v1` preprocessing, preserve no-cut audio, use probabilities in
all aggregation modes, reject mismatched file pairs, retain channels, constrain
boundary snapping, and cover first/last/partial inference frames. Unconditional
gap bridging is removed because it can erase short protected vocal events.
Training checkpoints include metadata, and the old loader recognizes both
metadata checkpoints and bare state dictionaries, including best/latest names.

Existing v3 weights learned the previous data distribution. The repaired
preprocessing/thresholding changes behavior; recalibrate and, ideally, retrain
the baseline before comparing quality. The legacy training script retains its
original fixed 8-second dataset/optimizer/loss design; the temporal trainer is
the route for the new annotation, augmentation, metric, and resume workflow.

Use v3 weights with the new planner/renderer and whole-recording evaluator:

```sh
python -m smartcutter evaluate data/manifest.jsonl --checkpoint ckpts/v3_model_48000.pth --legacy-sample-rate 48000 --split validation --calibrate --policy-out reports/v3-policy.json --output reports/v3-validation.json
python -m smartcutter evaluate data/manifest.jsonl --checkpoint ckpts/v3_model_48000.pth --legacy-sample-rate 48000 --policy reports/v3-policy.json --split test --output reports/v3-test.json
python -m smartcutter infer infer_input --checkpoint ckpts/v3_model_48000.pth --legacy-sample-rate 48000 --policy reports/v3-policy.json --output infer_output
```

A metadata v3 checkpoint supplies its training sample rate automatically. Bare
state dictionaries require the explicit rate. New temporal checkpoints require
fresh training; v3 weights cannot be loaded into the new architecture.

## Checks and code map

```sh
python -m unittest discover -s tests -v
```

Tests cover regressions, probability semantics, native-rate stereo precision,
approved fade/cut bounds, overlap-add endpoints, short inputs, anti-phase stereo,
annotation leakage, target alignment, both model variants, gradient flow,
padding invariance, training, loading, and resume. CI runs on Linux/Windows with
Python 3.10/3.12. CUDA performance and real listening quality need separate runs.

| Module | Responsibility |
|---|---|
| `smartcutter/config.py` | Versionable configuration |
| `smartcutter/frontend.py` | Shared multiresolution and legacy features |
| `model_v9.py` / `smartcutter/model.py` | SmartCutterV9 encoder, Conformer/TCN, output heads |
| `smartcutter/data.py` | Annotations, imports, splits, augmentation, crops |
| `smartcutter/train.py` | Objective, checkpoints, resume, validation selection |
| `smartcutter/engine.py` | Model loading, channel veto, overlapping inference |
| `smartcutter/cuts.py` | Edit plans, bounded refinement, native rendering |
| `smartcutter/evaluate.py` | Whole-recording metrics and calibration |
| `smartcutter/__main__.py` | CLI and batch operations |
