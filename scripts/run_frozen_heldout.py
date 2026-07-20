"""Execute the frozen four-way held-out protocol without retuning.

The command consumes the immutable protocol emitted by
``select_sooper_scenario.py``.  It evaluates every selected SOOPER training
seed and the paired bare-backbone, fallback, and CALF controls on exactly the
same held-out initial-state seeds.  Existing successful summaries are skipped,
which makes an interrupted evaluation safely resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def command_record(command: list[str], output_dir: Path) -> dict:
    return {"command": command, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--underwater-intrusion-penalty", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("retuning_after_held_out_observation") is not False:
        raise SystemExit("Protocol does not prohibit held-out retuning")
    seeds = ",".join(map(str, protocol["held_out_seeds"]))
    base_checkpoint = args.model_root / protocol["base_checkpoint"]
    if not base_checkpoint.is_file():
        raise SystemExit(f"Missing base checkpoint: {base_checkpoint}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for run in protocol["sooper"]["runs"]:
        run_dir = args.screening_root / run["run_dir"]
        checkpoints = sorted((run_dir / "checkpoints").glob("sooper_checkpoint_*.pt"))
        if not checkpoints:
            raise SystemExit(f"Missing SOOPER checkpoint in {run_dir}")
        output = args.output_root / f"sooper-seed-{run['sooper_training_seed']}"
        command = [
            "uv",
            "run",
            "python",
            "run/eval_sooper.py",
            "--checkpoint",
            str(checkpoints[-1]),
            "--seeds",
            seeds,
            "--device",
            args.device,
            "--underwater-intrusion-penalty",
            str(args.underwater_intrusion_penalty),
            "--output-dir",
            str(output),
            "--tracking-uri",
            args.tracking_uri,
            "--experiment-name",
            args.experiment_name,
            "--run-name",
            f"sooper-seed-{run['sooper_training_seed']}-held-out",
        ]
        records.append(command_record(command, output))

    common = [
        "underwater-drone",
        "--env-id",
        protocol["env_id"],
        "--algorithm",
        protocol["algorithm"],
        "--model-path",
        str(base_checkpoint),
        "--seeds",
        seeds,
        "--device",
        args.device,
        "--horizon",
        str(protocol["horizon"]),
        "--gamma",
        str(protocol["gamma"]),
        "--cost-budget",
        str(protocol["cost_budget"]),
        "--tracking-uri",
        args.tracking_uri,
        "--experiment-name",
        args.experiment_name,
        "--underwater-intrusion-penalty",
        str(args.underwater_intrusion_penalty),
    ]
    for method in ("base", "fallback", "calf"):
        output = args.output_root / method
        command = [
            "uv",
            "run",
            "python",
            "run/eval_comparison_controls.py",
            *common,
            "--method",
            method,
            "--output-dir",
            str(output),
            "--run-name",
            f"{method}-held-out",
        ]
        if method == "calf":
            command.extend(
                [
                    "--relaxprob-init",
                    str(protocol["calf"]["p0"]),
                    "--relaxprob-factor",
                    str(protocol["calf"]["lambda"]),
                    "--calf-change-rate",
                    str(protocol["calf"]["change_rate"]),
                ]
            )
        records.append(command_record(command, output))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "format": "calf-wrapper-sooper-held-out-command-manifest-v1",
        "source_commit": commit,
        "protocol": str(args.protocol),
        "protocol_payload": protocol,
        "underwater_intrusion_penalty": args.underwater_intrusion_penalty,
        "commands": records,
    }
    (args.output_root / "command_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout:
        raise SystemExit("Refusing held-out evaluation from a dirty working tree")
    for record in records:
        output = Path(record["output_dir"])
        if (output / "summary.json").is_file():
            continue
        subprocess.run(record["command"], check=True)


if __name__ == "__main__":
    main()
