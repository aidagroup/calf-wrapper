#!/usr/bin/env python3
"""Compare reproducibility-critical members of Stable-Baselines3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


ENVIRONMENTS = {
    "pendulum": {
        "artifact_dir": "ppo_Pendulum-v1_9",
        "reference_steps": (30_000, 36_000, 102_000),
    },
    "cartpole": {
        "artifact_dir": "ppo_CartpoleSwingupEnv-v0_42",
        "reference_steps": (99_000, 108_000, 270_000),
    },
}
TRAINED_STATE_MEMBERS = (
    "policy.pth",
    "policy.optimizer.pth",
    "pytorch_variables.pth",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checkpoint_dir(root: Path, artifact_dir: str) -> Path:
    return root / artifact_dir / "checkpoints"


def compare_members(left: Path, right: Path) -> dict[str, bool]:
    with ZipFile(left) as left_zip, ZipFile(right) as right_zip:
        return {
            member: left_zip.read(member) == right_zip.read(member)
            for member in TRAINED_STATE_MEMBERS
        }


def compare_environment(
    environment: str,
    run_a: Path,
    run_b: Path,
    reference_root: Path,
) -> dict:
    config = ENVIRONMENTS[environment]
    artifact_dir = config["artifact_dir"]
    a_dir = checkpoint_dir(run_a, artifact_dir)
    b_dir = checkpoint_dir(run_b, artifact_dir)
    reference_dir = checkpoint_dir(reference_root, artifact_dir)
    a_files = {path.name: path for path in a_dir.glob("*.zip")}
    b_files = {path.name: path for path in b_dir.glob("*.zip")}
    common_names = sorted(a_files.keys() & b_files.keys())

    repeated_checkpoints = []
    for name in common_names:
        member_matches = compare_members(a_files[name], b_files[name])
        repeated_checkpoints.append(
            {
                "checkpoint": name,
                "whole_zip_exact": digest(a_files[name].read_bytes())
                == digest(b_files[name].read_bytes()),
                "trained_state_exact": all(member_matches.values()),
                "member_matches": member_matches,
            }
        )

    reference_checkpoints = []
    for step in config["reference_steps"]:
        name = f"ppo_checkpoint_{step}_steps.zip"
        a_matches = compare_members(a_files[name], reference_dir / name)
        b_matches = compare_members(b_files[name], reference_dir / name)
        reference_checkpoints.append(
            {
                "checkpoint": name,
                "run_a_trained_state_exact": all(a_matches.values()),
                "run_b_trained_state_exact": all(b_matches.values()),
                "run_a_member_matches": a_matches,
                "run_b_member_matches": b_matches,
            }
        )

    return {
        "run_a_checkpoint_count": len(a_files),
        "run_b_checkpoint_count": len(b_files),
        "checkpoint_sets_equal": a_files.keys() == b_files.keys(),
        "repeated_trained_state_exact_count": sum(
            row["trained_state_exact"] for row in repeated_checkpoints
        ),
        "repeated_whole_zip_exact_count": sum(
            row["whole_zip_exact"] for row in repeated_checkpoints
        ),
        "repeated_checkpoints": repeated_checkpoints,
        "reference_checkpoints": reference_checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        environment: compare_environment(
            environment, args.run_a, args.run_b, args.reference_root
        )
        for environment in ENVIRONMENTS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {}
    passed = True
    for environment, values in report.items():
        expected = values["run_a_checkpoint_count"]
        repeated_exact = values["repeated_trained_state_exact_count"]
        reference_exact = all(
            row["run_a_trained_state_exact"] and row["run_b_trained_state_exact"]
            for row in values["reference_checkpoints"]
        )
        environment_passed = (
            values["checkpoint_sets_equal"]
            and repeated_exact == expected
            and reference_exact
        )
        passed &= environment_passed
        summary[environment] = {
            "verdict": "PASS" if environment_passed else "FAIL",
            "repeated_trained_states": f"{repeated_exact}/{expected}",
            "reference_trained_states_exact": reference_exact,
        }
    print(json.dumps(summary))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
