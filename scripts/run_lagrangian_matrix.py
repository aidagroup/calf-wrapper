#!/usr/bin/env python3
"""Render or execute the frozen PPO/TD3-Lagrangian experiment matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.run_reproduction import require_pushed_clean_commit

MATRIX = {
    "pendulum": {
        "trainer": "run/train_ppo_lagrangian.py",
        "preset": "pendulum",
        "seeds": [9],
    },
    "cartpole": {
        "trainer": "run/train_ppo_lagrangian.py",
        "preset": "cartpole",
        "seeds": [42],
    },
    "underwater-drone": {
        "trainer": "run/train_td3_lagrangian.py",
        "preset": "underwater-drone",
        "seeds": list(range(10)),
    },
    "robot-navigation": {
        "trainer": "run/train_td3_lagrangian.py",
        "preset": "robot-navigation",
        "seeds": list(range(1, 11)),
    },
}


def command_for(
    environment: str,
    seed: int,
    device: str,
    output_root: Path,
    smoke: bool,
) -> list[str]:
    config = MATRIX[environment]
    command = [
        sys.executable,
        config["trainer"],
        config["preset"],
        "--seed",
        str(seed),
        "--device",
        device,
        "--evaluation-seed",
        "10000",
        "--evaluation-episodes",
        "1" if smoke else "200",
        "--paired-evaluation-seed",
        "42",
        "--paired-evaluation-episodes",
        "1" if smoke else "30",
        "--output-dir",
        str(output_root / environment / f"seed-{seed}"),
    ]
    if not smoke:
        return command
    if environment == "pendulum":
        command += [
            "--total-timesteps",
            "200",
            "--num-steps",
            "200",
            "--num-minibatches",
            "4",
            "--update-epochs",
            "1",
            "--lambda-update-episodes",
            "1",
        ]
    elif environment == "cartpole":
        command += [
            "--total-timesteps",
            "1000",
            "--num-steps",
            "1000",
            "--num-minibatches",
            "4",
            "--update-epochs",
            "1",
            "--lambda-update-episodes",
            "1",
        ]
    elif environment == "underwater-drone":
        command += [
            "--total-timesteps",
            "1505",
            "--learning-starts",
            "1500",
            "--buffer-size",
            "2000",
            "--batch-size",
            "4",
            "--lambda-update-episodes",
            "1",
            "--checkpoint-every",
            "0",
        ]
    else:
        command += [
            "--total-timesteps",
            "1005",
            "--learning-starts",
            "1000",
            "--buffer-size",
            "1500",
            "--batch-size",
            "4",
            "--lambda-update-episodes",
            "1",
            "--checkpoint-every",
            "0",
        ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "environments",
        nargs="*",
        choices=sorted(MATRIX),
        default=None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root", type=Path, default=Path("run/artifacts/lagrangian")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--seeds",
        help="Comma-separated training-seed override; requires one environment.",
    )
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Only for smoke runs; full runs always require a clean pushed commit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute and (not args.smoke or not args.allow_dirty):
        require_pushed_clean_commit()
    if args.allow_dirty and not args.smoke:
        raise SystemExit("--allow-dirty is restricted to smoke runs")

    commands: list[list[str]] = []
    environments = args.environments or list(MATRIX)
    if args.seeds and len(environments) != 1:
        raise SystemExit("--seeds requires exactly one environment")
    seed_override = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    if seed_override == []:
        raise SystemExit("--seeds must contain at least one integer")
    for environment in environments:
        seeds = seed_override or MATRIX[environment]["seeds"]
        for seed in seeds:
            command = command_for(
                environment,
                seed,
                args.device,
                args.output_root,
                args.smoke,
            )
            output_dir = Path(command[command.index("--output-dir") + 1])
            result_name = (
                f"ppo_lagrangian_{environment}_seed{seed}.json"
                if environment in {"pendulum", "cartpole"}
                else f"td3_lagrangian_{environment}_seed{seed}.json"
            )
            if (
                args.execute
                and args.skip_completed
                and (output_dir / result_name).exists()
            ):
                print(f"skipping completed output: {output_dir}", flush=True)
                continue
            if args.execute and output_dir.exists():
                raise SystemExit(f"refusing to overwrite existing output: {output_dir}")
            commands.append(command)

    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            environment = Path(command[command.index("--output-dir") + 1]).parent.name
            seed = command[command.index("--seed") + 1]
            log_path = args.output_root / "logs" / f"{environment}-seed-{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )


if __name__ == "__main__":
    main()
