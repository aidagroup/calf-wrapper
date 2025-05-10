# Main training script

import os
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from src.utils.mlflow import MlflowConfig, mlflow_monitoring, create_mlflow_logger
from src import run_path

current_dir = Path(__file__).parent

@dataclass
class ExperimentConfig:
    """Configuration for PPO training experiment."""

    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri="file://" + os.path.join(str(run_path), "mlruns"),
            experiment_name=run_path.name,
        )
    )
    """MLflow configuration for experiment tracking"""

    local_artifacts_path: Path = run_path / "artifacts"
    """Path to store local training artifacts like model checkpoints and logs"""

    env_id: str = "Pendulum-v1"
    """Gym environment ID to train on"""

    n_envs: int = 1
    """Number of environments for vectorized training. Higher values increase training throughput"""

    gamma: float = 0.98
    """Discount factor for future rewards. Range [0,1]. Higher values prioritize long-term rewards"""

    use_sde: bool = True
    """Whether to use State Dependent Exploration for action sampling"""

    sde_sample_freq: int = 4
    """How often to sample a new noise matrix for State Dependent Exploration"""

    learning_rate: float = 1e-3
    """Learning rate for the optimizer. Controls step size during gradient updates"""

    verbose: int = 1
    """Verbosity level: 0=no output, 1=info, 2=debug"""

    seed: int = 42
    """Random seed for reproducibility across training runs"""

    total_timesteps: int = 300_000
    """Total number of timesteps to train for"""

    n_steps: int = 3000
    """Number of steps to run for each environment per policy rollout. Affects sample efficiency"""

    save_model_every_steps: int = 3000
    """Frequency of saving model checkpoints during training"""

    device: str = "cuda:0"
    """Device to run training on: 'cpu' or 'cuda:n' for GPU"""


presets = {
    # Preset configurations for different environments
    # Usage: python train_ppo.py pendulum  # For pendulum preset
    #        python train_ppo.py cartpole  # For cartpole preset
    #        python train_ppo.py --help    # Show all available options
    #
    # Each preset is a tuple of (description, config) where:
    # - description: Brief explanation of the training setup
    # - config: ExperimentConfig with environment-specific hyperparameters
    "pendulum": (
        "Training of PPO on Pendulum-v1",
        ExperimentConfig(
            env_id="Pendulum-v1",
            total_timesteps=102_000,
            n_steps=3000,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            verbose=1,
            seed=9,
            device="cpu",  # For devices that do not have a GPU
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
                experiment_name="ppo_pendulum_train",
            ),
        ),
    ),
    "cartpole": (
        "Training of PPO on CartPole-v1",
        ExperimentConfig(
            env_id="CartpoleSwingupEnv-v0",
            total_timesteps=300_000,
            n_steps=3000,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            verbose=1,
            seed=42,
            device="cpu",  # For devices that do not have a GPU
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
                experiment_name="ppo_cartpole_train",
            ),
        ),
    ),
}


@mlflow_monitoring()
def main(config: ExperimentConfig):
    # Create the environment
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)
    local_artifacts_path = (
        config.local_artifacts_path / f"ppo_{config.env_id}_{config.seed}"
    )
    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        gamma=config.gamma,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        learning_rate=config.learning_rate,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    model.set_logger(create_mlflow_logger())

    print("Model initialized successfully.")
    # Set up a checkpoint callback to save the model every 'save_freq' steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=local_artifacts_path / "checkpoints",  # Directory to save the model
        name_prefix=f"ppo_checkpoint",
    )

    mlflow_checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=os.path.join(
            mlflow.get_artifact_uri()[len("file://") :], "checkpoints"
        ),  # Directory to save the model
        name_prefix=f"ppo_{config.env_id}",
    )

    # Combine both callbacks using CallbackList
    callback = CallbackList([checkpoint_callback, mlflow_checkpoint_callback])

    print("Starting training ...")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
