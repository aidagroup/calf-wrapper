import os
import csv
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import tyro
from multiprocessing import Pool
import subprocess
import pandas as pd
import mlflow
from loguru import logger
from collections import defaultdict
from typing import Optional, Literal
from src import run_path

eval_path = run_path / "eval.py"


@dataclass
class Config:
    max_jobs: int = 10
    output_dir: Optional[Path] = None
    n_envs: int = 100
    n_checkpoints: Optional[int] = None
    max_checkpoint: Optional[int] = None
    mlflow_experiment_name: Optional[str] = None
    subcommand: Literal["cartpole", "pendulum"] = "cartpole"
    mlflow_tracking_uri: str = f"file://{str(run_path / 'mlruns')}"
    seed: Optional[int] = None

    def __post_init__(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        if self.mlflow_experiment_name is None:
            self.mlflow_experiment_name = f"eval/{self.subcommand}/checkpoints"

        if self.output_dir is None:
            self.output_dir = (
                run_path / "artifacts" / "eval" / self.mlflow_experiment_name
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


def run_evaluation(args):
    checkpoint, config, seed_name, step_id = args

    logger.info(f"Evaluating checkpoint at step {step_id} for seed {seed_name}")

    mlflow_run_name = f"{seed_name}_step_{step_id}"

    cmd = [
        "uv",
        "run",
        str(eval_path),
        config.subcommand,
        "--eval-mode=checkpoint",
        f"--mlflow.experiment-name={config.mlflow_experiment_name}",
        f"--mlflow.run-name={mlflow_run_name}",
        f"--model-path={checkpoint}",
        f"--n_envs={config.n_envs}",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        return {
            "seed": seed_name,
            "step": step_id,
            "success": True,
            "run_name": mlflow_run_name,
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running evaluation: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return {
            "seed": seed_name,
            "step": step_id,
            "success": False,
            "run_name": mlflow_run_name,
        }


def get_performance_from_mlflow(config, evaluation_results):
    logger.info("Retrieving performance metrics from MLflow")

    performance_data = []
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    if not experiment:
        logger.warning(
            f"MLflow experiment '{config.mlflow_experiment_name}' not found, creating it"
        )
        experiment_id = mlflow.create_experiment(config.mlflow_experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
        if not experiment:
            logger.error(
                f"Failed to create MLflow experiment '{config.mlflow_experiment_name}'"
            )
            return pd.DataFrame()

    # Find all successful runs
    for result in evaluation_results:
        if not result["success"]:
            continue

        # Get the run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{result['run_name']}'",
        )

        if len(runs) == 0:
            logger.warning(f"No MLflow run found for {result['run_name']}")
            continue

        run = runs.iloc[0]

        # Extract performance metrics - modify these based on what metrics are actually tracked
        metrics = {
            "seed": result["seed"],
            "step": result["step"],
            "mean_reward": run.get("metrics.mean_reward", np.nan),
            "std_reward": run.get("metrics.std_reward", np.nan),
            "min_reward": run.get("metrics.min_reward", np.nan),
            "max_reward": run.get("metrics.max_reward", np.nan),
            "median_reward": run.get("metrics.median_reward", np.nan),
        }

        performance_data.append(metrics)

    return pd.DataFrame(performance_data)


def select_linear_checkpoints(checkpoints, n_checkpoints):
    """Select n_checkpoints linearly spaced checkpoints."""
    if not checkpoints:
        return []

    # Sort checkpoints by step ID
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x[3]))
    if n_checkpoints is None or n_checkpoints >= len(sorted_checkpoints):
        return sorted_checkpoints

    # Select linearly spaced indices
    indices = np.linspace(0, len(sorted_checkpoints) - 1, n_checkpoints, dtype=int)
    return [sorted_checkpoints[i] for i in indices]


def main(config: Config):
    env_id = (
        "CartpoleSwingupEnv-v0" if config.subcommand == "cartpole" else "Pendulum-v1"
    )
    logger.info(f"Collecting performances for environment: {env_id}")

    # Collect all checkpoints for the specified environment grouped by seed
    all_checkpoints = defaultdict(list)
    artifacts_path = run_path / "artifacts"

    for seed_data in artifacts_path.glob("*"):
        if seed_data.is_dir() and env_id in seed_data.name:
            seed_name = seed_data.name

            if config.seed is not None and str(config.seed) != seed_name.split("_")[-1]:
                continue

            for checkpoint_path in seed_data.glob("checkpoints/*"):
                if checkpoint_path.is_file() and checkpoint_path.name.endswith(".zip"):
                    step_id = checkpoint_path.stem.split("_")[-2]
                    if (
                        config.max_checkpoint is not None
                        and int(step_id) > config.max_checkpoint
                    ):
                        continue
                    all_checkpoints[seed_name].append(
                        (str(checkpoint_path), config, seed_name, step_id)
                    )

    # Select linearly spaced checkpoints for each seed if n_checkpoints is specified
    tasks = []
    for seed_name, checkpoints in all_checkpoints.items():
        if config.n_checkpoints:
            selected_checkpoints = select_linear_checkpoints(
                checkpoints, config.n_checkpoints
            )
            logger.info(
                f"Seed {seed_name}: Selected {len(selected_checkpoints)}/{len(checkpoints)} checkpoints"
            )
            tasks.extend(selected_checkpoints)
        else:
            tasks.extend(checkpoints)

    logger.info(f"Found {len(tasks)} checkpoint evaluation tasks")

    # Run evaluations in parallel
    with Pool(processes=min(config.max_jobs, len(tasks))) as pool:
        pool.map(run_evaluation, tasks)


presets = {
    "cartpole": (
        "Evaluate cartpole checkpoints",
        Config(
            subcommand="cartpole",
            max_jobs=10,
            n_envs=30,
            n_checkpoints=40,
            max_checkpoint=102000,
        ),
    ),
    "pendulum": (
        "Evaluate pendulum checkpoints",
        Config(
            subcommand="pendulum",
            max_jobs=10,
            n_envs=30,
            n_checkpoints=40,
            max_checkpoint=102000,
        ),
    ),
}

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
