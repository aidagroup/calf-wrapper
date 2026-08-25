"""Run the copied and independently locked CALF-Enhance CleanRL TD3 trainer."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ENHANCE_ROOT = REPO_ROOT / "vendor" / "calf-enhance-td3"
ENHANCE_LOCK_SHA256 = "26812bc65b4f091bf16da07e10b7d67c9ae21ccc9d4432704795da6850055f40"
ENVIRONMENTS = {
    "underwater-drone": "UnderwaterDrone-v0",
    "robot-navigation": "RobotNavigationConstSpeedCatch-v0",
}


def require_pinned_runtime() -> None:
    if not (ENHANCE_ROOT / "uv.lock").exists():
        raise SystemExit(f"Vendored CALF-Enhance runtime is missing: {ENHANCE_ROOT}")
    lock_digest = hashlib.sha256((ENHANCE_ROOT / "uv.lock").read_bytes()).hexdigest()
    if lock_digest != ENHANCE_LOCK_SHA256:
        raise SystemExit(
            "Vendored CALF-Enhance lock does not match the pinned source: "
            f"expected {ENHANCE_LOCK_SHA256}, found {lock_digest}"
        )


def enhance_command(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(ENHANCE_ROOT),
        "--frozen",
        "python",
        str(ENHANCE_ROOT / "run" / "train_td3.py"),
        "--env-id",
        ENVIRONMENTS[args.environment],
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--total-timesteps",
        str(args.total_timesteps),
        "--learning-rate",
        "0.0003",
        "--num-envs",
        "1",
        "--buffer-size",
        "1000000",
        "--gamma",
        "0.99",
        "--tau",
        "0.005",
        "--batch-size",
        "256",
        "--policy-noise",
        "0.2",
        "--exploration-noise",
        "0.1",
        "--learning-starts",
        str(args.learning_starts),
        "--policy-frequency",
        "2",
        "--noise-clip",
        "0.5",
        "--rolling-average-window",
        "20",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--torch-deterministic",
        "--no-capture-video",
        "--mlflow.tracking-uri",
        args.tracking_uri,
        "--mlflow.experiment-name",
        args.experiment_name,
        "--mlflow.run-name",
        args.run_name,
    ]


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "MLFLOW_DISABLE_GIT": "1",
        "MINIO_PORT": "9030",
        "MINIO_CONSOLE_PORT": "9031",
        "MLFLOW_PORT": "5001",
        "EXPERIMENT_TRACKING_HOST": "127.0.0.1",
        "AWS_ACCESS_KEY_ID": "unused-for-file-tracking",
        "AWS_SECRET_ACCESS_KEY": "unused-for-file-tracking",
        "AWS_DEFAULT_REGION": "us-east-1",
        "ARTIFACT_UPLOAD_POLL_INTERVAL": "30",
        "LOG_ARTEFACTS_UPLOAD_PATH": "/tmp/calf-enhance-artifact-staging",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=30_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir is None:
        env_id = ENVIRONMENTS[args.environment]
        args.checkpoint_dir = (
            REPO_ROOT / "run" / "artifacts" / f"td3_{env_id}_{args.seed}" / "checkpoints"
        )
    require_pinned_runtime()
    command = enhance_command(args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=runtime_environment(),
            check=True,
        )


if __name__ == "__main__":
    main()
