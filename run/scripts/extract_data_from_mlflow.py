#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path

import mlflow
import pandas as pd
import tyro
from loguru import logger

from src import run_path


def get_experiment_by_name(
    experiment_name: str,
) -> Optional[mlflow.entities.Experiment]:
    """Get MLflow experiment by name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment


def get_runs_data(experiment_id: str) -> List[Dict[str, Any]]:
    """Extract runs data from the specified experiment."""
    runs_data = []

    # Get all runs from the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    for _, run_row in runs.iterrows():
        run_id = run_row["run_id"]
        run = mlflow.get_run(run_id)

        # Get run name or use run_id if name is not available
        run_name = run.data.tags.get("mlflow.runName", run_id)

        # Get metrics and artifacts URI
        metrics = run.data.metrics
        artifacts_uri = run.info.artifact_uri

        runs_data.append(
            {
                "run_name": run_name,
                "metrics": metrics,
                "artifacts_uri": artifacts_uri,
            }
        )

    return runs_data


@dataclass
class Args:
    """Extract data from MLflow experiments."""

    experiment_name: str
    """MLflow experiment name"""

    tracking_uri: str = f"file://{str(run_path / 'mlruns')}"
    """MLflow tracking URI"""

    output: Optional[Path] = None
    """Output file path (CSV format)"""

    def __post_init__(self):
        if self.output is None:
            self.output = (
                run_path
                / "artifacts"
                / "eval"
                / self.experiment_name.replace("/", "_")
                / "checkpoint_performances.csv"
            )


def main():
    args = tyro.cli(Args)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)
    logger.info(f"Set MLflow tracking URI to: {args.tracking_uri}")

    # Get experiment
    experiment = get_experiment_by_name(args.experiment_name)
    if not experiment:
        logger.error(f"Experiment '{args.experiment_name}' not found")
        return

    logger.info(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")

    # Get runs data
    runs_data = get_runs_data(experiment.experiment_id)
    logger.info(f"Retrieved {len(runs_data)} runs")

    if not runs_data:
        logger.warning("No runs found in the experiment")
        return

    # Create DataFrame for display
    rows = []
    all_metric_keys = set()

    for run_data in runs_data:
        all_metric_keys.update(run_data["metrics"].keys())

    for run_data in runs_data:
        row = {
            "run_name": run_data["run_name"],
            "artifacts_uri": run_data["artifacts_uri"],
        }
        for metric_key in all_metric_keys:
            row[metric_key] = run_data["metrics"].get(metric_key, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Display the table
    print("\nRuns and Metrics:")
    print(df.to_string(index=False))

    # Save to CSV if output path is provided
    os.makedirs(args.output.parent, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
