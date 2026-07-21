#!/usr/bin/env python3
"""Prepare reproducible sensitivity, gate-ablation, and calibrated full tasks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.run_checkpoint_matrix import Task, prepare_tasks, write_tasks


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PROTOCOL = REPO_ROOT / "experiments" / "checkpoint-sweep-v1.json"
DEFAULT_NU_PROTOCOL = REPO_ROOT / "experiments" / "nu-ablation-v1.json"


def read_calibrations(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key in ("training_seed", "checkpoint_step", "horizon"):
            row[key] = int(row[key])
        for key in (
            "n",
            "goal_local_variation",
            "fallback_value_range",
            "range_increment",
            "nu",
        ):
            row[key] = float(row[key])
    return rows


def n_label(n: float) -> str:
    return f"n{n:g}".replace(".", "p")


def sensitivity_tasks(
    rows: list[dict[str, Any]],
    *,
    matrix_id: str,
    evaluation_seed: int,
    trials: int,
) -> list[Task]:
    tasks = []
    for row in rows:
        label = n_label(row["n"])
        rule = row["rule_variant"]
        preset = row["preset"]
        tasks.append(
            Task(
                task_id=(
                    f"{preset}__s{row['training_seed']}__t{row['checkpoint_step']}"
                    f"__conservative__{rule}__{label}"
                ),
                matrix_id=matrix_id,
                environment=preset,
                preset=preset,
                env_id=row["environment"],
                algorithm=row["algorithm"],
                training_seed=row["training_seed"],
                checkpoint_step=row["checkpoint_step"],
                checkpoint_path=row["checkpoint_path"],
                mode=f"conservative_{rule}_{label}",
                eval_mode="calf_wrapper",
                calf_mode="conservative",
                evaluation_seed=evaluation_seed,
                n_envs=trials,
                calf_change_rate=row["nu"],
                nu_calibration_n=row["n"],
                nu_calibration_rule=rule,
            )
        )
    return tasks


def full_tasks(
    rows: list[dict[str, Any]],
    checkpoint_protocol: dict[str, Any],
    *,
    artifacts_root: Path,
    matrix_id: str,
    selected_n: float,
    selected_rule: str,
    evaluation_seed: int,
    trials: int,
    modes: list[str],
) -> list[Task]:
    chosen = {
        (row["preset"], row["training_seed"], row["checkpoint_step"]): row["nu"]
        for row in rows
        if row["n"] == selected_n and row["rule_variant"] == selected_rule
    }
    protocol = json.loads(json.dumps(checkpoint_protocol))
    protocol["evaluation"]["evaluation_seed"] = evaluation_seed
    protocol["evaluation"]["trials"] = trials
    tasks = prepare_tasks(
        protocol,
        artifacts_root,
        matrix_id,
        list(protocol["environments"]),
        modes,
    )
    calibrated = []
    for task in tasks:
        if task.eval_mode != "calf_wrapper":
            calibrated.append(task)
            continue
        key = (task.preset, task.training_seed, task.checkpoint_step)
        if key not in chosen:
            raise RuntimeError(f"missing calibrated nu for {key}")
        calibrated.append(
            replace(
                task,
                calf_change_rate=chosen[key],
                nu_calibration_n=selected_n,
                nu_calibration_rule=selected_rule,
            )
        )
    return calibrated


def infinity_ablation_tasks(
    rows: list[dict[str, Any]],
    *,
    matrix_id: str,
    preset: str,
    training_seed: int,
    checkpoint_step: int,
    selected_n: float,
    selected_rule: str,
    evaluation_seed: int,
    trials: int,
) -> list[Task]:
    matches = [
        row
        for row in rows
        if row["preset"] == preset
        and row["training_seed"] == training_seed
        and row["checkpoint_step"] == checkpoint_step
        and row["n"] == selected_n
        and row["rule_variant"] == selected_rule
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one calibrated row, found {len(matches)}")
    row = matches[0]
    common = {
        "matrix_id": matrix_id,
        "environment": preset,
        "preset": preset,
        "env_id": row["environment"],
        "algorithm": row["algorithm"],
        "training_seed": training_seed,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": row["checkpoint_path"],
        "evaluation_seed": evaluation_seed,
        "n_envs": trials,
    }
    tasks = [
        Task(
            task_id=f"{preset}__fallback",
            mode="fallback",
            eval_mode="fallback",
            calf_mode="custom",
            calf_change_rate=None,
            nu_calibration_n=None,
            nu_calibration_rule=None,
            **common,
        ),
        Task(
            task_id=f"{preset}__s{training_seed}__t{checkpoint_step}__base",
            mode="base",
            eval_mode="base",
            calf_mode="custom",
            calf_change_rate=None,
            nu_calibration_n=None,
            nu_calibration_rule=None,
            **common,
        ),
    ]
    for mode in ("conservative", "guarded", "moderate"):
        for threshold_label, threshold, n in (
            ("calibrated", row["nu"], selected_n),
            ("infinity", float("inf"), None),
        ):
            tasks.append(
                Task(
                    task_id=(
                        f"{preset}__s{training_seed}__t{checkpoint_step}__{mode}"
                        f"__{threshold_label}"
                    ),
                    mode=f"{mode}_{threshold_label}",
                    eval_mode="calf_wrapper",
                    calf_mode=mode,
                    calf_change_rate=threshold,
                    nu_calibration_n=n,
                    nu_calibration_rule=(selected_rule if n is not None else None),
                    **common,
                )
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sensitivity", "full", "infinity-ablation"))
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument(
        "--checkpoint-protocol", type=Path, default=DEFAULT_CHECKPOINT_PROTOCOL
    )
    parser.add_argument("--nu-protocol", type=Path, default=DEFAULT_NU_PROTOCOL)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--selected-n", type=float)
    parser.add_argument("--selected-rule")
    parser.add_argument(
        "--modes",
        default="fallback,base,conservative,guarded,moderate,balanced,high,almost_open",
    )
    parser.add_argument("--preset")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--checkpoint-step", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_calibrations(args.calibration_csv)
    nu_protocol = json.loads(args.nu_protocol.read_text())
    development = nu_protocol["development_evaluation"]
    evaluation_seed = int(development["evaluation_seed"])
    trials = int(development["trials"])
    if args.command == "sensitivity":
        tasks = sensitivity_tasks(
            rows,
            matrix_id=args.matrix_id,
            evaluation_seed=evaluation_seed,
            trials=trials,
        )
    elif args.command == "full":
        if (
            args.artifacts_root is None
            or args.selected_n is None
            or args.selected_rule is None
        ):
            raise ValueError(
                "full requires --artifacts-root, --selected-n, and --selected-rule"
            )
        tasks = full_tasks(
            rows,
            json.loads(args.checkpoint_protocol.read_text()),
            artifacts_root=args.artifacts_root,
            matrix_id=args.matrix_id,
            selected_n=args.selected_n,
            selected_rule=args.selected_rule,
            evaluation_seed=evaluation_seed,
            trials=trials,
            modes=[item.strip() for item in args.modes.split(",") if item.strip()],
        )
    else:
        if (
            args.preset is None
            or args.training_seed is None
            or args.checkpoint_step is None
            or args.selected_n is None
            or args.selected_rule is None
        ):
            raise ValueError(
                "infinity-ablation requires --preset, --training-seed, "
                "--checkpoint-step, --selected-n, and --selected-rule"
            )
        tasks = infinity_ablation_tasks(
            rows,
            matrix_id=args.matrix_id,
            preset=args.preset,
            training_seed=args.training_seed,
            checkpoint_step=args.checkpoint_step,
            selected_n=args.selected_n,
            selected_rule=args.selected_rule,
            evaluation_seed=evaluation_seed,
            trials=trials,
        )
    write_tasks(tasks, args.output)
    print(
        json.dumps(
            {
                "matrix_id": args.matrix_id,
                "tasks": len(tasks),
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
