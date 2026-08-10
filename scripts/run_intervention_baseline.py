#!/usr/bin/env python3
"""Train and evaluate the advantage-based intervention baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "run" / "artifacts" / "intervention-baseline"
ENVIRONMENTS = (
    "pendulum",
    "cartpole",
    "underwater-drone",
    "robot-navigation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        default="all",
        help="Comma-separated presets or all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-anchors", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=30)
    parser.add_argument("--thresholds", default="0.0,0.01,0.025,0.05")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--critic-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--tracking-uri")
    parser.add_argument(
        "--experiment-prefix", default="calf-wrapper/advantage-intervention"
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--include-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_environments(value: str) -> list[str]:
    environments = list(ENVIRONMENTS) if value == "all" else value.split(",")
    unknown = sorted(set(environments) - set(ENVIRONMENTS))
    if unknown:
        raise ValueError(f"Unknown environments: {', '.join(unknown)}")
    return environments


def render(command: list[str]) -> str:
    return " ".join(command)


def execute(command: list[str], dry_run: bool) -> None:
    print(render(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    environments = selected_environments(args.environment)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    for environment in environments:
        output_dir = args.output_root / environment
        critic_path = output_dir / "goal_cost_critic.pt"
        if not args.skip_training:
            execute(
                [
                    sys.executable,
                    str(REPO_ROOT / "run" / "train_intervention_critic.py"),
                    environment,
                    "--output-dir",
                    str(output_dir),
                    "--n-anchors",
                    str(args.n_anchors),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.critic_seed),
                ],
                args.dry_run,
            )
        if args.include_controls:
            for control in ("base", "fallback"):
                command = [
                    sys.executable,
                    str(REPO_ROOT / "run" / "eval.py"),
                    environment,
                    "--eval-mode",
                    control,
                    "--device",
                    args.device,
                    "--seed",
                    str(args.evaluation_seed),
                    "--n-envs",
                    str(args.n_envs),
                    "--no-save-episode-data",
                    "--result-path",
                    str(output_dir / f"evaluation_{control}.json"),
                    "--mlflow.experiment-name",
                    f"{args.experiment_prefix}/{environment}",
                    "--mlflow.run-name",
                    control,
                ]
                if args.tracking_uri is not None:
                    command.extend(["--mlflow.tracking-uri", args.tracking_uri])
                execute(command, args.dry_run)
        for threshold in thresholds:
            threshold_name = str(threshold).replace("-", "m").replace(".", "p")
            command = [
                sys.executable,
                str(REPO_ROOT / "run" / "eval.py"),
                environment,
                "--eval-mode",
                "advantage_intervention",
                "--intervention.critic-path",
                str(critic_path),
                "--intervention.threshold",
                str(threshold),
                "--intervention.device",
                args.device,
                "--device",
                args.device,
                "--n-envs",
                str(args.n_envs),
                "--seed",
                str(args.evaluation_seed),
                "--no-save-episode-data",
                "--result-path",
                str(output_dir / f"evaluation_eta_{threshold_name}.json"),
                "--mlflow.experiment-name",
                f"{args.experiment_prefix}/{environment}",
                "--mlflow.run-name",
                f"advantage_intervention_eta_{threshold_name}",
            ]
            if args.tracking_uri is not None:
                command.extend(["--mlflow.tracking-uri", args.tracking_uri])
            execute(command, args.dry_run)


if __name__ == "__main__":
    main()
