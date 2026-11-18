"""SmartCutter v9: vocal-preserving segmentation and non-destructive cut planning."""

__version__ = "9.0.0"
MODEL_VERSION = "v9"
ARCHITECTURE = "v9"
# Accept checkpoints from the same implementation before it was named v9.
COMPATIBLE_ARCHITECTURES = frozenset((ARCHITECTURE, "temporal_segmenter"))
