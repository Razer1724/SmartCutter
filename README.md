# SmartCutter v9

SmartCutter v9 prepares audio for Applio/RVC by shortening removable gaps while
preserving vocals. This upgrade starts from v3 and adds a temporal segmentation
pipeline, explicit annotations, conservative cut planning, and native-rate
multichannel rendering.

Version **9.0.0**. The v9 architecture requires training. No pretrained temporal checkpoint or
real-data quality benchmark is included.

**[Full installation and upgrade guide](docs/UPGRADE.md)**

## What is implemented

- A shared 32 kHz analysis frontend with short/long spectral windows and absolute
  energy features; output retains the original sample rate.
- A frequency encoder and a 5.25-million-parameter Conformer segmenter, with
  removable-gap and boundary heads. A TCN alternative is included.
- Checked raw/marked imports, explicit interval labels, speaker/source split
  validation, short-file/tail crops, and label-aware augmentation.
- Weighted vocal-preservation training, validation-only threshold calibration,
  full training checkpoints, and resume.
- Conservative cut decisions, channel vetoes, bounded waveform refinement,
  native audio rendering, and reviewable edit lists with source/output time maps.
- Whole-recording metrics, experiment configurations, regression tests, and CI.
- Repairs to v3 probability handling, no-cut behavior, file pairing, preprocessing,
  checkpoint loading, stereo handling, and inference frame coverage.

## Install

Use Python 3.10–3.12 in a virtual environment. For CPU:

```sh
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m smartcutter --version
python -m smartcutter --help
```

Use a matching CUDA build of PyTorch for NVIDIA GPU training. The existing
Windows installer environment also supports the new modules.

## Train and use

First prepare a reviewed manifest containing real audio and interval labels,
with independent training, validation, and test groups. Conversion, schema, and
splitting instructions are in [the guide](docs/UPGRADE.md).

```sh
python -m smartcutter validate-data data/manifest.jsonl
python -m smartcutter train data/manifest.jsonl --config configs/v9.json --output runs/v9
python -m smartcutter evaluate data/manifest.jsonl --checkpoint runs/v9/best.pt --output reports/v9.json
python -m smartcutter infer infer_input --checkpoint runs/v9/best.pt --output previews --dry-run
python -m smartcutter infer infer_input --checkpoint runs/v9/best.pt --output infer_output
```

On Windows, `run_train_v9.bat` and `run_infer_v9.bat` use the paths
above in the existing `env` environment. Original launchers remain v3 entry
points. The guide also shows how to use v3 weights with the new planner and
renderer; v3 weights cannot initialize the new temporal architecture.

The `configs/` directory includes matched Conformer, TCN, single-resolution,
and boundary-head experiments. Benchmark them on your held-out recordings and
listen around the cuts before choosing a replacement for v3.

## Verify

```sh
python -m unittest discover -s tests -v
```

The suite exercises both model variants, native-rate rendering, preservation
regressions, labels/splits, training/resume, and CLI flows. CI is configured for
Linux and Windows on Python 3.10 and 3.12.

## License and origin

MIT; see [LICENSE](LICENSE). This project began as machine-learning silence
truncation for Applio/RVC and Codename-RVC-Fork-4. Original community:
[Codename's Sanctuary](https://discord.gg/ymfdwx5jwZ).
