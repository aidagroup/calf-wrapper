import os
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import tyro
from multiprocessing import Pool
import subprocess

from src import run_path

eval_path = run_path / "eval.py"

@dataclass
class Config:
    checkpoint_dir: Path
    subcommand: str
    n_envs: int 
    mlflow_experiment_name: str 
    n_processes: int  
    n_relax_probs: int  
    step_size: int 

def run_evaluation(args):
    checkpoint, relax_prob, config = args
    steps = checkpoint.split('_')[-2]
    relax_prob_formatted = f"{relax_prob:.4f}"
    
    print(f"Evaluating checkpoint at step {steps} with relax_prob={relax_prob_formatted}")
    print(f"Checkpoint path: {checkpoint}")
    
    cmd = [
        "python",
        str(eval_path),
        config.subcommand,
        "--eval-mode=calf_wrapper",
        f"--mlflow.experiment-name={config.mlflow_experiment_name}",
        f"--mlflow.run-name=calf_{steps}_relax_{relax_prob_formatted}",
        f"--model-path={checkpoint}",
        f"--calf.relaxprob_init={relax_prob_formatted}",
        f"--n_envs={config.n_envs}"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
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
    checkpoint_files = sorted(
        glob.glob(os.path.join(config.checkpoint_dir, "ppo_checkpoint_*_steps.zip")),
        key=lambda x: int(x.split('_')[-2])
    )
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    total_checkpoints = len(checkpoint_files)
    
    # Create all evaluation tasks
    tasks = []
    for relax_prob in np.linspace(0.0, 1.0, config.n_relax_probs):
        for i, checkpoint in enumerate(checkpoint_files):
            if i % config.step_size != 0 and i != total_checkpoints - 1:
                continue
            tasks.append((checkpoint, relax_prob, config))
    
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
            checkpoint_dir=run_path
            / "artifacts/ppo_CartpoleSwingupEnv-v0_42/checkpoints",
            subcommand="cartpole",
            n_envs=100,
            mlflow_experiment_name="eval/cartpole/calf",
            n_processes=10,
            n_relax_probs=21,
            step_size=5,
        ),
    ),
    "pendulum": (
        "Run pendulum with Calf wrapper",
        Config(
            checkpoint_dir=run_path / "artifacts/ppo_Pendulum-v1_42/checkpoints",
            subcommand="pendulum",
            n_envs=100,
            mlflow_experiment_name="eval/pendulum/calf",
            n_processes=10,
            n_relax_probs=21,
            step_size=5,
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
