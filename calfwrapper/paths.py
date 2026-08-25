"""Repository paths used by the command-line interface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "artifacts" / "checkpoints"
OUTPUTS = ROOT / "outputs"
TRAINING_OUTPUT = OUTPUTS / "training"
REFERENCE_TRIALS = ROOT / "reference" / "trials"
REFERENCE_STUDIES = ROOT / "reference" / "studies"
