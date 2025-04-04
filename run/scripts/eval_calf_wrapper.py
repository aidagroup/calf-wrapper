import os
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import tyro
from multiprocessing import Pool
import subprocess
from typing import Literal, Optional
from src import run_path

eval_path = run_path / "eval.py"


@dataclass
class Config:

    subcommand: Literal["cartpole", "pendulum"]
    n_envs: int

    n_processes: int
    n_relax_probs: int
    max_checkpoint_step: int
    n_checkpoints: int
    mlflow_experiment_name: Optional[str] = None
    checkpoint_dir: Path = run_path / "artifacts"

    seed: Optional[int] = None

    def __post_init__(self):
        if self.mlflow_experiment_name is None:
            self.mlflow_experiment_name = f"eval/{self.subcommand}/calf"


def run_evaluation(args):
    checkpoint, relax_prob, seed, config = args
    steps = checkpoint.split("_")[-2]
    relax_prob_formatted = f"{relax_prob:.4f}"

    print(
        f"Evaluating checkpoint at step {steps} with relax_prob={relax_prob_formatted}"
    )
    print(f"Checkpoint path: {checkpoint}")

    cmd = [
        "python",
        str(eval_path),
        config.subcommand,
        "--eval-mode=calf_wrapper",
        f"--mlflow.experiment-name={config.mlflow_experiment_name}",
        f"--mlflow.run-name=calf_{steps}_seed_{seed}_relax_{relax_prob_formatted}",
        f"--model-path={checkpoint}",
        f"--calf.relaxprob_init={relax_prob_formatted}",
        f"--n_envs={config.n_envs}",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"Stderr: {e.stderr}")
        return False


def main(config: Config):
    # Get all checkpoint files and sort them
    checkpoint_files = []
    for folder_dir in config.checkpoint_dir.iterdir():
        if not folder_dir.is_dir() or not folder_dir.name.startswith("ppo"):
            continue

        if (
            config.seed is not None
            and str(config.seed) != folder_dir.name.split("_")[-1]
        ):
            continue

        if config.subcommand in folder_dir.name.lower():
            local_checkpoint_files = sorted(
                glob.glob(
                    os.path.join(
                        folder_dir / "checkpoints", "ppo_checkpoint_*_steps.zip"
                    )
                ),
                key=lambda x: int(x.split("_")[-2]),
            )
            seed = folder_dir.name.split("_")[-1]
            local_checkpoint_files = np.array(
                [
                    (checkpoint_file, seed)
                    for checkpoint_file in local_checkpoint_files
                    if int(checkpoint_file.split("_")[-2]) <= config.max_checkpoint_step
                ]
            )
            local_checkpoint_files = local_checkpoint_files[
                np.unique(
                    np.linspace(
                        0, len(local_checkpoint_files) - 1, config.n_checkpoints
                    ).astype(int)
                )
            ]
            checkpoint_files.extend(local_checkpoint_files.tolist())

    print(f"Found {len(checkpoint_files)} checkpoints")

    # Create all evaluation tasks
    tasks = []
    for relax_prob in [0.0, 0.5, 0.95]:
        for checkpoint, seed in checkpoint_files:
            tasks.append((checkpoint, relax_prob, seed, config))

    print(f"Created {len(tasks)} evaluation tasks")
    print(f"Running with {config.n_processes} parallel processes")

    # Run evaluations in parallel
    with Pool(processes=config.n_processes) as pool:
        results = pool.map(run_evaluation, tasks)

    # Report results
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    print(f"\nEvaluation complete:")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {failed}")


presets = {
    "cartpole": (
        "Run cartpole with Calf wrapper",
        Config(
            subcommand="cartpole",
            n_envs=30,
            n_processes=10,
            n_relax_probs=21,
            max_checkpoint_step=102000,
            n_checkpoints=40,
        ),
    ),
    "pendulum": (
        "Run pendulum with Calf wrapper",
        Config(
            subcommand="pendulum",
            n_envs=30,
            n_processes=10,
            n_relax_probs=21,
            max_checkpoint_step=102000,
            n_checkpoints=40,
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
