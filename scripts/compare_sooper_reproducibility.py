"""Compare two SOOPER runs, including raw metrics and checkpoint state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


VOLATILE_SUMMARY_KEYS = {"checkpoint", "mlflow_run_id", "wall_clock_seconds"}


def equal_nested(left: Any, right: Any, path: str = "root") -> list[str]:
    differences: list[str] = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left, right):
            differences.append(path)
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right, equal_nan=True):
            differences.append(path)
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append(path + ".keys")
        for key in set(left) & set(right):
            differences.extend(equal_nested(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            differences.append(path + ".length")
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(equal_nested(a, b, f"{path}[{index}]"))
    elif isinstance(left, float) and isinstance(right, float) and np.isnan(left):
        if not np.isnan(right):
            differences.append(path)
    elif left != right:
        differences.append(path)
    return differences


def normalized_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    for key in VOLATILE_SUMMARY_KEYS:
        summary.pop(key, None)
    config = summary.get("config", {})
    for key in ("output_dir", "run_name", "tracking_uri"):
        config.pop(key, None)
    return summary


def newest_checkpoint(run: Path) -> Path:
    checkpoints = sorted((run / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints below {run}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    differences = equal_nested(
        normalized_summary(args.left / "summary.json"),
        normalized_summary(args.right / "summary.json"),
        "summary",
    )
    for filename in ("online_episodes.csv", "evaluation_trials.csv"):
        left_bytes = (args.left / "raw" / filename).read_bytes()
        right_bytes = (args.right / "raw" / filename).read_bytes()
        if left_bytes != right_bytes:
            differences.append(f"raw.{filename}")
    left_checkpoint = torch.load(
        newest_checkpoint(args.left), map_location="cpu", weights_only=False
    )
    right_checkpoint = torch.load(
        newest_checkpoint(args.right), map_location="cpu", weights_only=False
    )
    left_checkpoint.pop("config", None)
    right_checkpoint.pop("config", None)
    differences.extend(
        equal_nested(left_checkpoint, right_checkpoint, "checkpoint_state")
    )
    report = {
        "format": "calf-wrapper-sooper-reproducibility-v1",
        "left": str(args.left),
        "right": str(args.right),
        "identical": not differences,
        "differences": differences,
    }
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
