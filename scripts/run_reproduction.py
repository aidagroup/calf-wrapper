#!/usr/bin/env python3
"""Run the documented PPO training and evaluation matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ENVIRONMENTS = {
    "pendulum": {
        "seed": 9,
        "eval_seed": 42,
        "steps": 200,
        "stages": {"early": 30_000, "mid": 36_000, "late": 102_000},
        "artifact_dir": "ppo_Pendulum-v1_9",
    },
    "cartpole": {
        "seed": 42,
        "eval_seed": 42,
        "steps": 1_000,
        "stages": {"early": 99_000, "mid": 108_000, "late": 270_000},
        "artifact_dir": "ppo_CartpoleSwingupEnv-v0_42",
    },
}
WRAPPER_MODES = {"conservative": 0.0, "balanced": 0.5, "brave": 0.95}


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require_pushed_clean_commit() -> None:
    if git_output("status", "--porcelain"):
        raise SystemExit("Refusing to run experiments from a dirty checkout.")
    commit = git_output("rev-parse", "HEAD")
    remote_branches = git_output("branch", "-r", "--contains", commit)
    if not remote_branches:
        raise SystemExit(f"Refusing to run unpushed commit {commit}.")


def commands_for_environment(
    env_name: str,
    tracking_uri: str,
    artifact_root: Path,
    smoke: bool,
    skip_training: bool,
) -> list[list[str]]:
    config = ENVIRONMENTS[env_name]
    commands: list[list[str]] = []
    train_steps = 3_000 if smoke else config["stages"]["late"]
    if not skip_training:
        commands.append(
            [
                sys.executable,
                "run/train_ppo.py",
                env_name,
                "--mlflow.tracking-uri",
                tracking_uri,
                "--mlflow.experiment-name",
                f"calf-wrapper/reproduction/train/{env_name}",
                "--mlflow.run-name",
                f"train_{env_name}_seed_{config['seed']}",
                "--local-artifacts-path",
                str(artifact_root),
                "--total-timesteps",
                str(train_steps),
                "--save-model-every-steps",
                "3000",
            ]
        )

    stages = {"late": train_steps} if smoke else config["stages"]
    n_envs = 3 if smoke else 30
    n_steps = 20 if smoke else config["steps"]
    common = [
        "--mlflow.tracking-uri",
        tracking_uri,
        "--mlflow.experiment-name",
        f"calf-wrapper/reproduction/eval/{env_name}",
        "--n-envs",
        str(n_envs),
        "--n-steps",
        str(n_steps),
        "--seed",
        str(config["eval_seed"]),
    ]
    commands.append(
        [
            sys.executable,
            "run/eval.py",
            env_name,
            "--eval-mode",
            "fallback",
            "--checkpoint-stage",
            "none",
            "--mlflow.run-name",
            "fallback",
            *common,
        ]
    )
    for stage, step in stages.items():
        checkpoint = (
            artifact_root
            / config["artifact_dir"]
            / "checkpoints"
            / f"ppo_checkpoint_{step}_steps.zip"
        )
        commands.append(
            [
                sys.executable,
                "run/eval.py",
                env_name,
                "--eval-mode",
                "base",
                "--checkpoint-stage",
                stage,
                "--model-path",
                str(checkpoint),
                "--mlflow.run-name",
                f"base_{stage}",
                *common,
            ]
        )
        for mode, relaxprob in WRAPPER_MODES.items():
            commands.append(
                [
                    sys.executable,
                    "run/eval.py",
                    env_name,
                    "--eval-mode",
                    "calf_wrapper",
                    "--checkpoint-stage",
                    stage,
                    "--model-path",
                    str(checkpoint),
                    "--calf.relaxprob-init",
                    str(relaxprob),
                    "--mlflow.run-name",
                    f"calf_wrapper_{mode}_{stage}",
                    *common,
                ]
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/reproduction")
    )
    parser.add_argument("--environment", choices=["all", *ENVIRONMENTS], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unpushed", action="store_true")
    args = parser.parse_args()

    if not args.allow_unpushed and not args.dry_run:
        require_pushed_clean_commit()
    selected = ENVIRONMENTS if args.environment == "all" else [args.environment]
    commands = [
        command
        for env_name in selected
        for command in commands_for_environment(
            env_name,
            args.tracking_uri,
            args.artifact_root,
            args.smoke,
            args.skip_training,
        )
    ]
    for index, command in enumerate(commands, 1):
        rendered = shlex.join(command)
        print(f"[{index}/{len(commands)}] {rendered}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
