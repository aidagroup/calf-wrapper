#!/usr/bin/env python3
"""Audit concurrent TD3 runs and their remotely verified artifact batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ENVIRONMENTS = {
    "underwater-drone": {
        "env_id": "UnderwaterDrone-v0",
        "default_seeds": [0, 1, 2],
    },
    "robot-navigation": {
        "env_id": "RobotNavigationConstSpeedCatch-v0",
        "default_seeds": [1, 2, 3],
    },
}


def parse_seeds(value: str, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def artifact_paths(client: MlflowClient, run_id: str, root: str = "") -> set[str]:
    paths = set()
    for artifact in client.list_artifacts(run_id, root):
        if artifact.is_dir:
            paths.update(artifact_paths(client, run_id, artifact.path))
        else:
            paths.add(artifact.path)
    return paths


def latest_run(client: MlflowClient, experiment_id: str, run_name: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def audit_run(
    client: MlflowClient,
    experiment_id: str,
    environment: str,
    seed: int,
    checkpoint_every: int,
    total_timesteps: int,
    minimum_checkpoint_step: int,
    require_finished: bool,
) -> dict:
    env_id = ENVIRONMENTS[environment]["env_id"]
    run_name = f"td3_{env_id}_seed_{seed}"
    run = latest_run(client, experiment_id, run_name)
    if run is None:
        return {"run_name": run_name, "ok": False, "errors": ["run not found"]}

    errors = []
    if require_finished and run.info.status != "FINISHED":
        errors.append(f"status is {run.info.status}, expected FINISHED")
    elif run.info.status not in {"RUNNING", "FINISHED"}:
        errors.append(f"unexpected status {run.info.status}")

    expected_params = {
        "env_id": env_id,
        "seed": str(seed),
        "total_timesteps": str(total_timesteps),
        "checkpoint_every": str(checkpoint_every),
    }
    for key, expected_value in expected_params.items():
        actual_value = run.data.params.get(key)
        if actual_value != expected_value:
            errors.append(f"parameter {key}={actual_value}, expected {expected_value}")

    if not run.data.metrics:
        errors.append("no metrics logged")

    remote_paths = artifact_paths(client, run.info.run_id)
    if not any(path.startswith("_batch_manifests/") for path in remote_paths):
        errors.append("no remote batch manifest")

    required_steps = []
    if minimum_checkpoint_step:
        required_steps.append(minimum_checkpoint_step)
    if require_finished:
        required_steps = list(
            range(checkpoint_every, total_timesteps + 1, checkpoint_every)
        )
    for step in required_steps:
        stem = f"checkpoints/td3_checkpoint_{step}_steps"
        if f"{stem}.pt" not in remote_paths:
            errors.append(f"missing remote checkpoint {step}")
        if f"{stem}.json" not in remote_paths:
            errors.append(f"missing remote checkpoint metadata {step}")

    verified_batches = int(run.data.tags.get("artifact_batches_verified", "0"))
    if verified_batches < 1:
        errors.append("no remotely verified artifact batch")
    upload_status = run.data.tags.get("artifact_upload_status")
    expected_upload_statuses = (
        {"verified"} if require_finished else {"running", "verified"}
    )
    if upload_status not in expected_upload_statuses:
        errors.append(f"artifact_upload_status is {upload_status}")

    return {
        "run_name": run_name,
        "run_id": run.info.run_id,
        "status": run.info.status,
        "metric_count": len(run.data.metrics),
        "artifact_count": len(remote_paths),
        "verified_batches": verified_batches,
        "artifact_upload_status": upload_status,
        "ok": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--drone-seeds", default="0,1,2")
    parser.add_argument("--robot-seeds", default="1,2,3")
    parser.add_argument("--checkpoint-every", type=int, default=30_000)
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--minimum-checkpoint-step", type=int, default=0)
    parser.add_argument("--require-finished", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    requested_seeds = {
        "underwater-drone": parse_seeds(
            args.drone_seeds, ENVIRONMENTS["underwater-drone"]["default_seeds"]
        ),
        "robot-navigation": parse_seeds(
            args.robot_seeds, ENVIRONMENTS["robot-navigation"]["default_seeds"]
        ),
    }

    runs = []
    for environment, seeds in requested_seeds.items():
        experiment_name = f"{args.experiment_prefix}/{environment}"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            for seed in seeds:
                runs.append(
                    {
                        "run_name": f"td3_{ENVIRONMENTS[environment]['env_id']}_seed_{seed}",
                        "ok": False,
                        "errors": [f"experiment not found: {experiment_name}"],
                    }
                )
            continue
        for seed in seeds:
            runs.append(
                audit_run(
                    client,
                    experiment.experiment_id,
                    environment,
                    seed,
                    args.checkpoint_every,
                    args.total_timesteps,
                    args.minimum_checkpoint_step,
                    args.require_finished,
                )
            )

    result = {"all_ok": all(run["ok"] for run in runs), "runs": runs}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not result["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
