"""Launch committed SOOPER matrix workers in persistent tmux sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_assignment(value: str) -> tuple[int, str]:
    index, separator, device = value.partition("=")
    if not separator or not index.isdigit() or not device:
        raise argparse.ArgumentTypeError("assignment must be WORKER_INDEX=DEVICE")
    return int(index), device


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument(
        "--assignment",
        action="append",
        type=parse_assignment,
        required=True,
        help="Repeat as WORKER_INDEX=DEVICE, for example 0=cuda:0",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260720)
    parser.add_argument("--sharding", choices=["weighted", "strided"], default="weighted")
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    dirty = git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("Refusing to launch from a dirty tracked working tree")
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    matrix = args.matrix.resolve()
    args.result_root.mkdir(parents=True, exist_ok=True)
    logs = args.result_root / "logs"
    logs.mkdir(exist_ok=True)
    launches = []
    for worker_index, device in args.assignment:
        if not 0 <= worker_index < args.worker_count:
            raise SystemExit(f"Worker index {worker_index} is outside worker count")
        session = f"{args.session_prefix}-{worker_index}"
        exists = subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True
        )
        if exists.returncode == 0:
            raise SystemExit(f"tmux session already exists: {session}")
        command = [
            sys.executable,
            "scripts/run_sooper_matrix.py",
            "--matrix",
            str(matrix),
            "--result-root",
            str(args.result_root.resolve()),
            "--model-root",
            str(args.model_root.resolve()),
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(args.worker_count),
            "--shuffle-seed",
            str(args.shuffle_seed),
            "--sharding",
            args.sharding,
            "--device",
            device,
        ]
        if args.max_tasks is not None:
            command.extend(("--max-tasks", str(args.max_tasks)))
        log = (logs / f"worker-{worker_index}.log").resolve()
        shell_command = f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "sh", "-lc", shell_command],
            cwd=REPO_ROOT,
            check=True,
        )
        launches.append(
            {
                "worker_index": worker_index,
                "device": device,
                "session": session,
                "log": str(log),
                "command": command,
            }
        )
    manifest = {
        "format": "calf-wrapper-sooper-worker-launch-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_commit": commit,
        "matrix": str(matrix),
        "matrix_sha256": file_sha256(matrix),
        "worker_count": args.worker_count,
        "shuffle_seed": args.shuffle_seed,
        "sharding": args.sharding,
        "launches": launches,
    }
    manifest_path = args.result_root / f"launch-{commit[:8]}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
