"""Exact comparison of reproduced evaluation trials with published trials."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

ENVIRONMENT_NAMES = {
    "Pendulum-v1": "pendulum",
    "CartpoleSwingupEnvLong-v0": "cartpole",
    "UnderwaterDrone-v0": "auv",
    "RobotNavigationConstSpeedCatch-v0": "robot",
}

FIELD_PAIRS = (
    ("episode_return", "return"),
    ("goal_reached", "ggr_success"),
    ("episode_length", "episode_length"),
    ("base_policy_actions", "base_policy_calls"),
    ("fallback_policy_actions", "fallback_calls"),
    ("critic_evaluations", "critic_calls"),
    ("policy_sequence_sha256", "selection_sha256"),
)


def _canonical_boolean(value: str) -> str:
    return {"0": "false", "1": "true"}.get(value.lower(), value.lower())


def _task_id(row: dict[str, str]) -> str:
    environment = ENVIRONMENT_NAMES[row["environment"]]
    mode = "lagrangian" if row["mode"].endswith("-lagrangian") else row["mode"]
    return f"{environment}-{row['checkpoint_stage'].lower()}-{mode}"


def verify_trials(
    generated_directory: Path,
    reference_path: Path,
    environments: set[str],
) -> dict[str, object]:
    with reference_path.open(newline="") as stream:
        reference_rows = [
            row
            for row in csv.DictReader(stream)
            if ENVIRONMENT_NAMES[row["environment"]] in environments
        ]

    expected: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reference_rows:
        expected[_task_id(row)].append(row)

    mismatches: list[dict[str, object]] = []
    verified_rows = 0
    for task_id, task_reference in sorted(expected.items()):
        path = generated_directory / f"{task_id}.csv"
        if not path.is_file():
            mismatches.append({"task": task_id, "error": "missing generated CSV"})
            continue
        with path.open(newline="") as stream:
            generated = list(csv.DictReader(stream))
        if len(generated) != len(task_reference):
            mismatches.append(
                {
                    "task": task_id,
                    "error": "row count",
                    "generated": len(generated),
                    "reference": len(task_reference),
                }
            )
            continue

        for trial, (actual, reference) in enumerate(zip(generated, task_reference, strict=True)):
            lagrangian_seed = 20260801 if task_id.startswith("cartpole-") else 42
            first_seed = lagrangian_seed if task_id.endswith("-lagrangian") else 20260801
            expected_seed = str(first_seed + trial)
            if reference["seed"] != expected_seed:
                mismatches.append(
                    {
                        "task": task_id,
                        "trial": trial,
                        "field": "seed",
                        "generated": expected_seed,
                        "reference": reference["seed"],
                    }
                )
            for generated_field, reference_field in FIELD_PAIRS:
                expected_value = reference.get(reference_field, "")
                if expected_value == "":
                    continue
                actual_value = actual[generated_field]
                if generated_field == "goal_reached":
                    actual_value = _canonical_boolean(actual_value)
                    expected_value = _canonical_boolean(expected_value)
                if actual_value != expected_value:
                    mismatches.append(
                        {
                            "task": task_id,
                            "trial": trial,
                            "field": generated_field,
                            "generated": actual_value,
                            "reference": expected_value,
                        }
                    )
            verified_rows += 1

    return {
        "status": "passed" if not mismatches else "failed",
        "comparison": "exact string equality for every published nonempty field",
        "tasks": len(expected),
        "trials": verified_rows,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def write_report(report: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")


def combine_reports(*reports: dict[str, object]) -> dict[str, object]:
    mismatches = [
        mismatch
        for report in reports
        for mismatch in cast(list[dict[str, object]], report["mismatches"])
    ]
    return {
        "status": "passed" if not mismatches else "failed",
        "comparison": "exact string equality for every published nonempty field",
        "tasks": sum(cast(int, report["tasks"]) for report in reports),
        "trials": sum(cast(int, report["trials"]) for report in reports),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
