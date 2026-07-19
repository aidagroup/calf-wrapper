#!/usr/bin/env python3
"""Launch the two-environment TD3 seed matrix in per-GPU tmux queues."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.run_reproduction import require_pushed_clean_commit

ENVIRONMENTS = {
    "underwater-drone": {
        "env_id": "UnderwaterDrone-v0",
        "default_seeds": range(10),
    },
    "robot-navigation": {
        "env_id": "RobotNavigationConstSpeedCatch-v0",
        "default_seeds": range(1, 11),
    },
}


def training_command(
    *,
    environment: str,
    seed: int,
    gpu: int,
    tracking_uri: str,
    experiment_prefix: str,
    smoke: bool,
) -> list[str]:
    config = ENVIRONMENTS[environment]
    total_timesteps = 1_000 if smoke else 3_000_000
    learning_starts = 25_000
    return [
        sys.executable,
        "run/train_td3.py",
        environment,
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu}",
        "--total-timesteps",
        str(total_timesteps),
        "--learning-starts",
        str(learning_starts),
        "--tracking-uri",
        tracking_uri,
        "--experiment-name",
        f"{experiment_prefix}/{environment}",
        "--run-name",
        f"td3_{config['env_id']}_seed_{seed}",
    ]


def parse_seeds(value: str | None, environment: str) -> list[int]:
    if value is None:
        return list(ENVIRONMENTS[environment]["default_seeds"])
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-prefix", default="calf-wrapper/td3")
    parser.add_argument("--environment", choices=["all", *ENVIRONMENTS], default="all")
    parser.add_argument("--seeds", help="Comma-separated override applied to each env")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated CUDA ids")
    parser.add_argument("--session-prefix", default="calf-wrapper-td3")
    parser.add_argument("--log-dir", type=Path, default=Path("run/logs"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unpushed", action="store_true")
    args = parser.parse_args()

    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one CUDA id")
    if not args.allow_unpushed and not args.dry_run:
        require_pushed_clean_commit()

    selected = list(ENVIRONMENTS) if args.environment == "all" else [args.environment]
    jobs = [
        (environment, seed)
        for environment in selected
        for seed in parse_seeds(args.seeds, environment)
    ]
    repo_root = Path(__file__).resolve().parent.parent
    args.log_dir.mkdir(parents=True, exist_ok=True)

    gpu_queues: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    for index, (environment, seed) in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        short_env = environment.replace("-", "_")
        job_name = f"{args.session_prefix}-{short_env}-s{seed}-g{gpu}"
        command = training_command(
            environment=environment,
            seed=seed,
            gpu=gpu,
            tracking_uri=args.tracking_uri,
            experiment_prefix=args.experiment_prefix,
            smoke=args.smoke,
        )
        log_path = args.log_dir / f"{job_name}.log"
        rendered = shlex.join(command)
        print(f"[{index + 1}/{len(jobs)}] {job_name}: {rendered}", flush=True)
        gpu_queues[gpu].append(f"{rendered} 2>&1 | tee {shlex.quote(str(log_path))}")

    if args.dry_run:
        return

    for gpu, queued_commands in gpu_queues.items():
        if not queued_commands:
            continue
        session = f"{args.session_prefix}-queue-g{gpu}"
        queue_command = "set -o pipefail; " + " && ".join(queued_commands)
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(repo_root),
                f"bash -lc {shlex.quote(queue_command)}",
            ],
            check=True,
        )
        print(
            f"started {session} with {len(queued_commands)} sequential jobs",
            flush=True,
        )


if __name__ == "__main__":
    main()
