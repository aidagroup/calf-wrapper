#!/usr/bin/env python3
"""Launch one concurrent tmux session per TD3 environment/seed pair."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from mlflow import MlflowClient

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
    checkpoint_every: int = 30_000,
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
        "--checkpoint-every",
        str(checkpoint_every),
    ]


def parse_seeds(value: str | None, environment: str) -> list[int]:
    if value is None:
        return list(ENVIRONMENTS[environment]["default_seeds"])
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def start_tmux_sessions(
    rendered_jobs: list[tuple[str, str, Path]], repo_root: Path
) -> None:
    """Start every rendered job immediately in its own tmux session."""

    existing_sessions = []
    for session, _, _ in rendered_jobs:
        existing_session = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if existing_session.returncode == 0:
            existing_sessions.append(session)
    if existing_sessions:
        raise SystemExit(
            "tmux session(s) already exist: " + ", ".join(existing_sessions)
        )

    for session, rendered, log_path in rendered_jobs:
        session_command = (
            f"set -o pipefail; {rendered} 2>&1 | tee {shlex.quote(str(log_path))}"
        )
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(repo_root),
                f"bash -lc {shlex.quote(session_command)}",
            ],
            check=True,
        )
        print(f"started concurrent session {session}", flush=True)


def ensure_experiments(
    tracking_uri: str,
    experiment_prefix: str,
    environments: list[str],
    *,
    client: MlflowClient | None = None,
) -> dict[str, str]:
    """Create all experiments before concurrent workers call set_experiment."""

    tracking_client = client or MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = {}
    for environment in environments:
        experiment_name = f"{experiment_prefix}/{environment}"
        experiment = tracking_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = tracking_client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        experiment_ids[experiment_name] = experiment_id
        print(f"ready MLflow experiment {experiment_name} ({experiment_id})")
    return experiment_ids


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
    parser.add_argument("--checkpoint-every", type=int, default=30_000)
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

    rendered_jobs: list[tuple[str, str, Path]] = []
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
            checkpoint_every=args.checkpoint_every,
        )
        log_path = args.log_dir / f"{job_name}.log"
        rendered = shlex.join(command)
        print(f"[{index + 1}/{len(jobs)}] {job_name}: {rendered}", flush=True)
        rendered_jobs.append((job_name, rendered, log_path))

    if args.dry_run:
        return

    ensure_experiments(args.tracking_uri, args.experiment_prefix, selected)
    start_tmux_sessions(rendered_jobs, repo_root)
    print(f"started {len(rendered_jobs)} concurrent TD3 runs", flush=True)


if __name__ == "__main__":
    main()
