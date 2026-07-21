#!/usr/bin/env python3
"""Prepare, launch, resume, and aggregate the fixed checkpoint evaluation sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from scripts.run_reproduction import require_pushed_clean_commit
from src.utils.verified_artifacts import log_verified_artifact_batch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = REPO_ROOT / "experiments" / "checkpoint-sweep-v1.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "run" / "artifacts"
CALF_MODES = (
    "conservative",
    "guarded",
    "moderate",
    "balanced",
    "high",
    "almost_open",
)
DEFAULT_TASK_SHUFFLE_SEED = 20260720


@dataclass(frozen=True)
class Task:
    task_id: str
    matrix_id: str
    environment: str
    preset: str
    env_id: str
    algorithm: str
    training_seed: int | None
    checkpoint_step: int | None
    checkpoint_path: str
    mode: str
    eval_mode: str
    calf_mode: str
    evaluation_seed: int
    n_envs: int
    calf_change_rate: float | None = None
    nu_calibration_n: float | None = None
    nu_calibration_rule: str | None = None


def checkpoint_step(path: Path) -> int:
    return int(path.name.split("_")[-2])


def training_seed(directory: Path) -> int:
    return int(directory.name.rsplit("_", 1)[-1])


def discover_checkpoints(
    config: dict[str, Any], artifacts_root: Path
) -> list[tuple[int, int, Path]]:
    discovered = []
    for directory in sorted(artifacts_root.glob(config["checkpoint_directory_glob"])):
        if not directory.is_dir():
            continue
        seed = training_seed(directory)
        for checkpoint in (directory / "checkpoints").glob(
            config["checkpoint_filename_glob"]
        ):
            step = checkpoint_step(checkpoint)
            if step <= int(config["training_horizon"]):
                discovered.append((seed, step, checkpoint.resolve()))
    return sorted(discovered, key=lambda item: (item[0], item[1]))


def prepare_tasks(
    protocol: dict[str, Any],
    artifacts_root: Path,
    matrix_id: str,
    environments: list[str],
    modes: list[str],
    *,
    smoke: bool = False,
) -> list[Task]:
    evaluation = protocol["evaluation"]
    n_envs = 2 if smoke else int(evaluation["trials"])
    eval_seed = int(evaluation["evaluation_seed"])
    tasks = []
    for environment in environments:
        config = protocol["environments"][environment]
        checkpoints = discover_checkpoints(config, artifacts_root)
        if not checkpoints:
            raise RuntimeError(f"no eligible checkpoints found for {environment}")
        if smoke:
            checkpoints = [checkpoints[-1]]

        fallback_checkpoint = checkpoints[-1][2]
        if "fallback" in modes:
            tasks.append(
                Task(
                    task_id=f"{environment}__fallback",
                    matrix_id=matrix_id,
                    environment=environment,
                    preset=environment,
                    env_id=config["env_id"],
                    algorithm=config["algorithm"],
                    training_seed=None,
                    checkpoint_step=None,
                    checkpoint_path=str(fallback_checkpoint),
                    mode="fallback",
                    eval_mode="fallback",
                    calf_mode="custom",
                    evaluation_seed=eval_seed,
                    n_envs=n_envs,
                )
            )

        for seed, step, path in checkpoints:
            for mode in modes:
                if mode == "fallback":
                    continue
                if mode == "base":
                    eval_mode, calf_mode = "base", "custom"
                elif mode in CALF_MODES:
                    eval_mode, calf_mode = "calf_wrapper", mode
                else:
                    raise ValueError(f"unknown evaluation mode: {mode}")
                tasks.append(
                    Task(
                        task_id=f"{environment}__s{seed}__t{step}__{mode}",
                        matrix_id=matrix_id,
                        environment=environment,
                        preset=environment,
                        env_id=config["env_id"],
                        algorithm=config["algorithm"],
                        training_seed=seed,
                        checkpoint_step=step,
                        checkpoint_path=str(path),
                        mode=mode,
                        eval_mode=eval_mode,
                        calf_mode=calf_mode,
                        evaluation_seed=eval_seed,
                        n_envs=n_envs,
                    )
                )
    return tasks


def write_tasks(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(tasks[0])))
        writer.writeheader()
        writer.writerows(asdict(task) for task in tasks)


def read_tasks(path: Path) -> list[Task]:
    tasks = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            tasks.append(
                Task(
                    **{
                        **row,
                        "training_seed": (
                            int(row["training_seed"]) if row["training_seed"] else None
                        ),
                        "checkpoint_step": (
                            int(row["checkpoint_step"])
                            if row["checkpoint_step"]
                            else None
                        ),
                        "evaluation_seed": int(row["evaluation_seed"]),
                        "n_envs": int(row["n_envs"]),
                        "calf_change_rate": (
                            float(row["calf_change_rate"])
                            if row.get("calf_change_rate")
                            else None
                        ),
                        "nu_calibration_n": (
                            float(row["nu_calibration_n"])
                            if row.get("nu_calibration_n")
                            else None
                        ),
                        "nu_calibration_rule": row.get("nu_calibration_rule") or None,
                    }
                )
            )
    return tasks


def evaluation_command(
    task: Task,
    *,
    tracking_uri: str,
    experiment_prefix: str,
    device: str,
    result_path: Path,
) -> list[str]:
    run_name = task.task_id.replace("__", "-")
    command = [
        sys.executable,
        str(REPO_ROOT / "run" / "eval.py"),
        task.preset,
        "--eval-mode",
        task.eval_mode,
        "--model-path",
        task.checkpoint_path,
        "--device",
        device,
        "--seed",
        str(task.evaluation_seed),
        "--n-envs",
        str(task.n_envs),
        "--no-save-episode-data",
        "--mlflow.tracking-uri",
        tracking_uri,
        "--mlflow.experiment-name",
        f"{experiment_prefix}/{task.environment}",
        "--mlflow.run-name",
        run_name,
        "--matrix-id",
        task.matrix_id,
        "--task-id",
        task.task_id,
        "--result-path",
        str(result_path),
    ]
    if task.training_seed is not None:
        command.extend(["--training-seed", str(task.training_seed)])
    if task.checkpoint_step is not None:
        command.extend(["--checkpoint-step", str(task.checkpoint_step)])
    if task.calf_mode != "custom":
        command.extend(["--calf.mode", task.calf_mode])
    if task.calf_change_rate is not None:
        command.extend(["--calf.calf-change-rate", str(task.calf_change_rate)])
    if task.nu_calibration_n is not None:
        command.extend(["--nu-calibration-n", str(task.nu_calibration_n)])
    if task.nu_calibration_rule is not None:
        command.extend(["--nu-calibration-rule", task.nu_calibration_rule])
    return command


def shuffled_task_shard(
    tasks: list[Task], worker_index: int, worker_count: int, shuffle_seed: int
) -> list[Task]:
    """Return one deterministic shard from a globally shuffled task order."""

    shuffled = list(tasks)
    random.Random(shuffle_seed).shuffle(shuffled)
    return shuffled[worker_index::worker_count]


def run_worker(args: argparse.Namespace) -> int:
    tasks = read_tasks(args.tasks)
    assigned = shuffled_task_shard(
        tasks, args.worker_index, args.worker_count, args.shuffle_seed
    )
    results_dir = args.matrix_dir / "results"
    failures_dir = args.matrix_dir / "failures"
    results_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    succeeded = failed = skipped = 0
    for index, task in enumerate(assigned, start=1):
        result_path = results_dir / f"{task.task_id}.json"
        failure_path = failures_dir / f"{task.task_id}.json"
        if result_path.exists():
            failure_path.unlink(missing_ok=True)
            skipped += 1
            continue
        command = evaluation_command(
            task,
            tracking_uri=args.tracking_uri,
            experiment_prefix=args.experiment_prefix,
            device=args.device,
            result_path=result_path,
        )
        print(
            f"worker {args.worker_index} [{index}/{len(assigned)}] {task.task_id}",
            flush=True,
        )
        last_error = None
        for attempt in range(1, args.retries + 2):
            try:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                failure_path.unlink(missing_ok=True)
                succeeded += 1
                last_error = None
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                print(
                    f"task {task.task_id} failed on attempt {attempt}: {error}",
                    flush=True,
                )
        if last_error is not None:
            failure_path.write_text(
                json.dumps(
                    {
                        "task": asdict(task),
                        "returncode": last_error.returncode,
                        "command": command,
                    },
                    indent=2,
                )
                + "\n"
            )
            failed += 1
    print(
        json.dumps(
            {
                "worker": args.worker_index,
                "assigned": len(assigned),
                "succeeded": succeeded,
                "failed": failed,
                "skipped": skipped,
            }
        ),
        flush=True,
    )
    return 1 if failed else 0


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: result.get(key)
        for key in (
            "task_id",
            "matrix_id",
            "environment",
            "algorithm",
            "training_seed",
            "checkpoint_step",
            "nu_calibration_n",
            "nu_calibration_rule",
            "eval_mode",
            "checkpoint_stage",
            "evaluation_seed",
            "horizon",
            "model_path",
            "mlflow_run_id",
        )
    }
    row["mode"] = result.get("mode")
    for key, value in result.get("calf", {}).items():
        row[f"calf_{key}"] = value
    row.update(result.get("metrics", {}))
    return row


def aggregate_results(matrix_dir: Path) -> tuple[Path, int, int]:
    rows = []
    result_paths = sorted((matrix_dir / "results").glob("*.json"))
    for result_path in result_paths:
        rows.append(flatten_result(json.loads(result_path.read_text())))
    completed_task_ids = {path.stem for path in result_paths}
    failures = [
        path
        for path in sorted((matrix_dir / "failures").glob("*.json"))
        if path.stem not in completed_task_ids
    ]
    output_path = matrix_dir / "checkpoint_mode_results.csv"
    fields = sorted({key for row in rows for key in row})
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as output:
        if fields:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, output_path)
    return output_path, len(rows), len(failures)


def run_monitor(args: argparse.Namespace) -> int:
    total = len(read_tasks(args.tasks))
    while True:
        table, completed, failed = aggregate_results(args.matrix_dir)
        print(
            f"matrix progress: completed={completed} failed={failed} total={total}",
            flush=True,
        )
        if completed + failed >= total:
            break
        time.sleep(args.poll_interval)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(f"{args.experiment_prefix}/_matrix")
    with mlflow.start_run(run_name=args.matrix_id):
        mlflow.log_params(
            {
                "matrix_id": args.matrix_id,
                "task_count": total,
                "completed_count": completed,
                "failed_count": failed,
            }
        )
        with tempfile.TemporaryDirectory(prefix="checkpoint_matrix_") as tmp:
            root = Path(tmp)
            (root / "protocol.json").write_bytes(args.protocol.read_bytes())
            (root / "tasks.csv").write_bytes(args.tasks.read_bytes())
            (root / table.name).write_bytes(table.read_bytes())
            if failed:
                failure_dir = root / "failures"
                failure_dir.mkdir()
                for path in (args.matrix_dir / "failures").glob("*.json"):
                    (failure_dir / path.name).write_bytes(path.read_bytes())
            log_verified_artifact_batch(root)
    return 1 if failed else 0


def start_tmux(
    *,
    tasks_path: Path,
    matrix_dir: Path,
    protocol: Path,
    matrix_id: str,
    tracking_uri: str,
    experiment_prefix: str,
    gpus: list[int],
    workers_per_gpu: int,
    retries: int,
    shuffle_seed: int,
) -> None:
    worker_count = len(gpus) * workers_per_gpu
    log_dir = matrix_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for worker_index in range(worker_count):
        gpu = gpus[worker_index % len(gpus)]
        session = f"calf-eval-{matrix_id}-w{worker_index}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--tasks",
            str(tasks_path),
            "--matrix-dir",
            str(matrix_dir),
            "--tracking-uri",
            tracking_uri,
            "--experiment-prefix",
            experiment_prefix,
            "--device",
            f"cuda:{gpu}",
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(worker_count),
            "--shuffle-seed",
            str(shuffle_seed),
            "--retries",
            str(retries),
        ]
        rendered = shlex.join(command)
        log_path = log_dir / f"worker-{worker_index}.log"
        shell_command = (
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
                str(REPO_ROOT),
                f"bash -lc {shlex.quote(shell_command)}",
            ],
            check=True,
        )
        print(f"started {session} on cuda:{gpu}")

    monitor_session = f"calf-eval-{matrix_id}-monitor"
    monitor_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "monitor",
        "--tasks",
        str(tasks_path),
        "--matrix-dir",
        str(matrix_dir),
        "--protocol",
        str(protocol),
        "--matrix-id",
        matrix_id,
        "--tracking-uri",
        tracking_uri,
        "--experiment-prefix",
        experiment_prefix,
    ]
    monitor_log = log_dir / "monitor.log"
    rendered = shlex.join(monitor_command)
    shell_command = (
        f"set -o pipefail; {rendered} 2>&1 | tee {shlex.quote(str(monitor_log))}"
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            monitor_session,
            "-c",
            str(REPO_ROOT),
            f"bash -lc {shlex.quote(shell_command)}",
        ],
        check=True,
    )
    print(f"started {monitor_session}")


def ensure_experiments(
    tracking_uri: str,
    experiment_prefix: str,
    environments: list[str],
) -> None:
    """Create experiments before concurrent workers enter MLflow."""

    client = MlflowClient(tracking_uri=tracking_uri)
    for suffix in [*environments, "_matrix"]:
        name = f"{experiment_prefix}/{suffix}"
        if client.get_experiment_by_name(name) is None:
            client.create_experiment(name)
        print(f"ready MLflow experiment {name}")


def parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-prefix", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    launch.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    launch.add_argument("--tracking-uri", required=True)
    launch.add_argument(
        "--experiment-prefix", default="calf-wrapper/checkpoint-sweep-v1"
    )
    launch.add_argument("--matrix-id")
    launch.add_argument("--environment", default="all")
    launch.add_argument(
        "--modes",
        default="fallback,base,conservative,guarded,moderate,balanced,high,almost_open",
    )
    launch.add_argument("--gpus", default="0,1")
    launch.add_argument("--workers-per-gpu", type=int, default=1)
    launch.add_argument("--retries", type=int, default=2)
    launch.add_argument("--shuffle-seed", type=int, default=DEFAULT_TASK_SHUFFLE_SEED)
    launch.add_argument("--smoke", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--allow-unpushed", action="store_true")

    worker = subparsers.add_parser("worker")
    parse_common(worker)
    worker.add_argument("--device", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--shuffle-seed", type=int, default=DEFAULT_TASK_SHUFFLE_SEED)
    worker.add_argument("--retries", type=int, default=2)

    monitor = subparsers.add_parser("monitor")
    parse_common(monitor)
    monitor.add_argument("--protocol", type=Path, required=True)
    monitor.add_argument("--matrix-id", required=True)
    monitor.add_argument("--poll-interval", type=float, default=60.0)

    prepared = subparsers.add_parser("launch-prepared")
    parse_common(prepared)
    prepared.add_argument("--protocol", type=Path, required=True)
    prepared.add_argument("--matrix-id", required=True)
    prepared.add_argument("--gpus", default="0,1")
    prepared.add_argument("--workers-per-gpu", type=int, default=1)
    prepared.add_argument("--retries", type=int, default=2)
    prepared.add_argument("--shuffle-seed", type=int, default=DEFAULT_TASK_SHUFFLE_SEED)
    prepared.add_argument("--allow-unpushed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "worker":
        return run_worker(args)
    if args.command == "monitor":
        return run_monitor(args)
    if args.command == "launch-prepared":
        tasks = read_tasks(args.tasks)
        environments = sorted({task.environment for task in tasks})
        if not args.allow_unpushed:
            require_pushed_clean_commit()
        ensure_experiments(args.tracking_uri, args.experiment_prefix, environments)
        gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
        start_tmux(
            tasks_path=args.tasks,
            matrix_dir=args.matrix_dir,
            protocol=args.protocol,
            matrix_id=args.matrix_id,
            tracking_uri=args.tracking_uri,
            experiment_prefix=args.experiment_prefix,
            gpus=gpus,
            workers_per_gpu=args.workers_per_gpu,
            retries=args.retries,
            shuffle_seed=args.shuffle_seed,
        )
        return 0

    protocol = json.loads(args.protocol.read_text())
    environments = (
        list(protocol["environments"])
        if args.environment == "all"
        else [item.strip() for item in args.environment.split(",") if item.strip()]
    )
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    matrix_id = args.matrix_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    matrix_dir = DEFAULT_ARTIFACTS / "checkpoint-sweeps" / matrix_id
    tasks = prepare_tasks(
        protocol,
        args.artifacts_root,
        matrix_id,
        environments,
        modes,
        smoke=args.smoke,
    )
    tasks_path = matrix_dir / "tasks.csv"
    write_tasks(tasks, tasks_path)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.environment] = counts.get(task.environment, 0) + 1
    print(json.dumps({"matrix_id": matrix_id, "tasks": len(tasks), "by_env": counts}))
    if args.dry_run:
        for task in tasks[: min(6, len(tasks))]:
            print(
                shlex.join(
                    evaluation_command(
                        task,
                        tracking_uri=args.tracking_uri,
                        experiment_prefix=args.experiment_prefix,
                        device="cuda:0",
                        result_path=matrix_dir / "results" / f"{task.task_id}.json",
                    )
                )
            )
        return 0
    if not args.allow_unpushed:
        require_pushed_clean_commit()
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    ensure_experiments(args.tracking_uri, args.experiment_prefix, environments)
    start_tmux(
        tasks_path=tasks_path,
        matrix_dir=matrix_dir,
        protocol=args.protocol,
        matrix_id=matrix_id,
        tracking_uri=args.tracking_uri,
        experiment_prefix=args.experiment_prefix,
        gpus=gpus,
        workers_per_gpu=args.workers_per_gpu,
        retries=args.retries,
        shuffle_seed=args.shuffle_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
