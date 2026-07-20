"""Run a deterministic, resumable SOOPER screening matrix across workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def task_id(task: dict[str, Any]) -> str:
    canonical = json.dumps(task, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{task['environment']}-seed{task['seed']}-{digest}"


def shuffled_shard(tasks, worker_index, worker_count, shuffle_seed):
    import random

    ordered = list(tasks)
    random.Random(shuffle_seed).shuffle(ordered)
    return ordered[worker_index::worker_count]


def command(task, output_dir, device, model_root, resume=None):
    model_path = Path(task["model_path"])
    if not model_path.is_absolute():
        model_path = (model_root / model_path).resolve()
    cmd = [
        sys.executable,
        "run/train_sooper.py",
        task["environment"],
        "--algorithm",
        task["algorithm"],
        "--model-path",
        str(model_path),
        "--seed",
        str(task["seed"]),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--tracking-uri",
        task["tracking_uri"],
        "--experiment-name",
        task["experiment_name"],
        "--run-name",
        task.get("run_name", task_id(task)),
    ]
    reserved = {
        "environment",
        "algorithm",
        "model_path",
        "seed",
        "tracking_uri",
        "experiment_name",
        "run_name",
    }
    for key, value in sorted(task.items()):
        if key in reserved:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(option)
        else:
            cmd.extend((option, str(value)))
    if resume is not None:
        cmd.extend(("--resume", str(resume)))
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve relative checkpoint paths in the matrix",
    )
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20260720)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    tasks = matrix["tasks"] if isinstance(matrix, dict) else matrix
    selected = shuffled_shard(
        tasks, args.worker_index, args.worker_count, args.shuffle_seed
    )
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]
    args.result_root.mkdir(parents=True, exist_ok=True)
    failures = args.result_root / "failures"
    failures.mkdir(exist_ok=True)
    for task in selected:
        identifier = task_id(task)
        output_dir = args.result_root / "runs" / identifier
        summary = output_dir / "summary.json"
        if summary.exists():
            continue
        checkpoints = sorted((output_dir / "checkpoints").glob("*.pt"))
        resume = checkpoints[-1] if checkpoints else None
        cmd = command(task, output_dir, args.device, args.model_root, resume)
        if args.dry_run:
            print(json.dumps({"task_id": identifier, "command": cmd}))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "worker.log"
        with log_path.open("a") as log:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        failure_path = failures / f"{identifier}.json"
        if completed.returncode:
            failure_path.write_text(
                json.dumps(
                    {
                        "task_id": identifier,
                        "returncode": completed.returncode,
                        "command": cmd,
                        "log": str(log_path),
                    },
                    indent=2,
                )
                + "\n"
            )
        elif failure_path.exists():
            failure_path.unlink()


if __name__ == "__main__":
    main()
