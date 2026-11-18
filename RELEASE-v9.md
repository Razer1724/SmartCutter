# SmartCutter v9 — 9.0.0

This release names the implemented vocal-preserving upgrade **v9**. Its network
and audio processing behavior are unchanged by this naming update.

- Public model: `model_v9.SmartCutterV9`.
- Main configuration: `configs/v9.json`.
- Training entry points: `train_v9.py`, `run_train_v9.bat`.
- Inference entry points: `inference_v9.py`, `run_infer_v9.bat`.
- CLI: `python -m smartcutter`; `--version` reports `SmartCutter v9 (9.0.0)`.
- New checkpoints identify the architecture/model as `v9` and software as `9.0.0`.

The checkpoint and edit-list **schema** version remains `1`. Schema version is
independent of the product's v9 name. Existing `temporal_segmenter` checkpoints
from the immediately preceding implementation still load and resume. The
`TemporalSegmenter` import and `configs/vocal_quality.json` remain compatible.
V3 weights remain supported through the separate v3 adapter.

See `docs/UPGRADE.md` for installation, labels, training, evaluation, and use.
The source distribution includes no pretrained v9 weights or training audio;
train a checkpoint using your reviewed annotations before inference.
