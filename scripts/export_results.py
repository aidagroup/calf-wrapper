#!/usr/bin/env python3
"""Export completed reproduction runs to raw CSV and aggregate JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow


STAGES = {
    "pendulum": {"early": 30_000, "mid": 36_000, "late": 102_000},
    "cartpole": {"early": 99_000, "mid": 108_000, "late": 270_000},
}
METHODS = ("conservative", "balanced", "brave", "base")
METRICS = ("mean_reward", "std_reward", "goal_reaching_rate")


def completed_runs(experiment_name: str) -> list[Any]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment does not exist: {experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    return [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and run.data.tags.get("repro.run_status") == "COMPLETED"
    ]


def select_latest_by_name(runs: list[Any]) -> dict[str, Any]:
    selected = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        selected.setdefault(name, run)
    return selected


def aggregate_environment(
    env_name: str, runs: dict[str, Any], reference: dict | None
) -> dict:
    result: dict[str, Any] = {"stages": STAGES[env_name]}
    fallback = runs.get("fallback")
    if fallback is None:
        raise RuntimeError(f"Missing completed fallback run for {env_name}")
    result["fallback"] = {
        metric: float(fallback.data.metrics[metric]) for metric in METRICS
    }
    for stage in STAGES[env_name]:
        result[stage] = {}
        for method in METHODS:
            run_name = (
                f"base_{stage}"
                if method == "base"
                else f"calf_wrapper_{method}_{stage}"
            )
            run = runs.get(run_name)
            if run is None:
                raise RuntimeError(f"Missing completed run {run_name} for {env_name}")
            result[stage][method] = {
                metric: float(run.data.metrics[metric]) for metric in METRICS
            }
        if reference and "residual" in reference[stage]:
            result[stage]["residual"] = reference[stage]["residual"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--experiment-prefix", default="calf-wrapper/reproduction")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    trial_rows = []
    for env_name in STAGES:
        experiment_name = f"{args.experiment_prefix}/eval/{env_name}"
        runs = completed_runs(experiment_name)
        selected = select_latest_by_name(runs)
        reference = None
        if args.reference_dir:
            reference = json.loads(
                (args.reference_dir / f"{env_name}.json").read_text()
            )
        aggregate = aggregate_environment(env_name, selected, reference)
        (args.output_dir / f"{env_name}.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        for run_name, run in selected.items():
            row = {
                "environment": env_name,
                "run_name": run_name,
                "run_id": run.info.run_id,
                "status": run.info.status,
                "artifact_uri": run.info.artifact_uri,
                "git_commit": run.data.tags.get("repro.git_commit"),
                "hostname": run.data.tags.get("repro.hostname"),
            }
            row.update(
                {f"metric.{key}": value for key, value in run.data.metrics.items()}
            )
            raw_rows.append(row)
            with tempfile.TemporaryDirectory() as temp_dir:
                trial_path = client.download_artifacts(
                    run.info.run_id, "raw/trial_metrics.csv", temp_dir
                )
                with Path(trial_path).open(newline="", encoding="utf-8") as source:
                    for trial in csv.DictReader(source):
                        trial_rows.append(
                            {
                                "environment": env_name,
                                "run_name": run_name,
                                "run_id": run.info.run_id,
                                "trial": int(trial["trial"]),
                                "reward": float(trial["reward"]),
                                "goal_reached": trial["goal_reached"].lower() == "true",
                            }
                        )

    columns = sorted({column for row in raw_rows for column in row})
    with (args.output_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(raw_rows)

    with (args.output_dir / "trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "environment",
                "run_name",
                "run_id",
                "trial",
                "reward",
                "goal_reached",
            ],
        )
        writer.writeheader()
        writer.writerows(trial_rows)

    provenance = {
        "experiment_prefix": args.experiment_prefix,
        "tracking_uri": args.tracking_uri,
        "selected_run_policy": "latest completed run per MLflow run name",
        "reference_only_methods": {},
    }
    if args.reference_dir:
        provenance["reference_only_methods"] = {
            "residual": (
                "Copied from the archived plotting JSON. CALF-Wrapper does not "
                "contain the Residual RL implementation or checkpoints."
            )
        }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
