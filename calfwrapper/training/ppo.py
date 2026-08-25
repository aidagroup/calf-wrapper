# Main training script

import os
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from calfwrapper.paths import TRAINING_OUTPUT
from calfwrapper.training.tracking import MlflowConfig, create_mlflow_logger, mlflow_monitoring


@dataclass
class ExperimentConfig:
    """Configuration for PPO training experiment."""

    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
            experiment_name="calfwrapper-training",
        )
    )
    """MLflow configuration for experiment tracking"""

    local_artifacts_path: Path = TRAINING_OUTPUT
    """Path to store local training artifacts like model checkpoints and logs"""

    env_id: str = "Pendulum-v1"
    """Gym environment ID to train on"""

    position_termination_threshold: float | None = None
    """Optional CartPole absolute position termination threshold"""

    velocity_termination_threshold: float | None = None
    """Optional CartPole absolute cart-velocity termination threshold"""

    angular_velocity_termination_threshold: float | None = None
    """Optional CartPole absolute angular-velocity termination threshold"""

    reward_position_clip: float | None = None
    """Optional CartPole position magnitude used in the reward"""

    terminate_on_out_of_bounds: bool | None = None
    """Optional CartPole termination behavior"""

    saturate_state_on_out_of_bounds: bool | None = None
    """Optional CartPole state saturation at the configured bounds"""

    n_envs: int = 1
    """Number of environments for vectorized training. Higher values increase training throughput"""

    gamma: float = 0.98
    """Discount factor for future rewards."""

    use_sde: bool = True
    """Whether to use State Dependent Exploration for action sampling"""

    sde_sample_freq: int = 4
    """How often to sample a new noise matrix for State Dependent Exploration"""

    learning_rate: float = 1e-3
    """Learning rate for the optimizer. Controls step size during gradient updates"""

    final_learning_rate: float | None = None
    """Optional final learning rate for a linear schedule over the full run"""

    verbose: int = 1
    """Verbosity level: 0=no output, 1=info, 2=debug"""

    seed: int = 42
    """Random seed for reproducibility across training runs"""

    total_timesteps: int = 300_000
    """Total number of timesteps to train for"""

    n_steps: int = 2048
    """Number of steps per PPO rollout (the historical SB3 default used by the reference runs)"""

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
            n_steps=2048,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            verbose=1,
            seed=9,
            device="cpu",  # For devices that do not have a GPU
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
                experiment_name="ppo_pendulum_train",
                run_name="ppo_pendulum_seed_9",
            ),
        ),
    ),
    "cartpole": (
        "Training of PPO on CartPole-v1",
        ExperimentConfig(
            env_id="CALFWrapper/CartPoleSwingUp-v0",
            total_timesteps=300_000,
            n_steps=2048,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            verbose=1,
            seed=42,
            device="cpu",  # For devices that do not have a GPU
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
                experiment_name="ppo_cartpole_train",
                run_name="ppo_cartpole_seed_42",
            ),
        ),
    ),
    "cartpole-long-wide": (
        "Training of PPO on the 1000-step CartPole task with 1.5x bounds",
        ExperimentConfig(
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            position_termination_threshold=7.5,
            velocity_termination_threshold=12.0,
            angular_velocity_termination_threshold=15.0,
            reward_position_clip=5.0,
            total_timesteps=1_000_000,
            n_steps=2048,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            verbose=1,
            seed=42,
            device="cuda:0",
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
                experiment_name="ppo_cartpole_long_wide_train",
                run_name="ppo_cartpole_long_wide_seed_42",
            ),
        ),
    ),
    "cartpole-long-wide-600k-annealed": (
        "Training of PPO on wide-bound CartPole with a stabilizing LR schedule",
        ExperimentConfig(
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            position_termination_threshold=7.5,
            velocity_termination_threshold=12.0,
            angular_velocity_termination_threshold=15.0,
            reward_position_clip=5.0,
            total_timesteps=600_000,
            n_steps=2048,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            verbose=1,
            seed=42,
            device="cuda:0",
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
                experiment_name="ppo_cartpole_long_wide_600k_annealed_train",
                run_name="ppo_cartpole_long_wide_600k_annealed_seed_42",
            ),
        ),
    ),
    "cartpole-long-saturated-600k-annealed": (
        "Training PPO on nonterminating CartPole with saturated state bounds",
        ExperimentConfig(
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            position_termination_threshold=7.5,
            velocity_termination_threshold=12.0,
            angular_velocity_termination_threshold=15.0,
            reward_position_clip=5.0,
            terminate_on_out_of_bounds=False,
            saturate_state_on_out_of_bounds=True,
            total_timesteps=600_000,
            n_steps=2048,
            n_envs=1,
            use_sde=True,
            sde_sample_freq=4,
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            verbose=1,
            seed=42,
            device="cuda:0",
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(TRAINING_OUTPUT), "mlruns"),
                experiment_name="ppo_cartpole_long_saturated_600k_annealed_train",
                run_name="ppo_cartpole_long_saturated_600k_annealed_seed_42",
            ),
        ),
    ),
}


def linear_learning_rate(initial: float, final: float):
    """Return an SB3 schedule parameterized by remaining training progress."""

    if initial <= 0 or final <= 0:
        raise ValueError("learning rates must be positive")
    if final > initial:
        raise ValueError("final learning rate must not exceed initial learning rate")

    def schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return schedule


@mlflow_monitoring()
def main(config: ExperimentConfig):
    # Create the environment
    optional_env_kwargs = {
        "position_termination_threshold": config.position_termination_threshold,
        "velocity_termination_threshold": config.velocity_termination_threshold,
        "angular_velocity_termination_threshold": (config.angular_velocity_termination_threshold),
        "reward_position_clip": config.reward_position_clip,
        "terminate_on_out_of_bounds": config.terminate_on_out_of_bounds,
        "saturate_state_on_out_of_bounds": (config.saturate_state_on_out_of_bounds),
    }
    env_kwargs = {key: value for key, value in optional_env_kwargs.items() if value is not None}
    env = make_vec_env(
        config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        env_kwargs=env_kwargs or None,
    )
    local_artifacts_path = config.local_artifacts_path / f"ppo_{config.env_id}_{config.seed}"
    # Instantiate the agent
    learning_rate = config.learning_rate
    if config.final_learning_rate is not None:
        learning_rate = linear_learning_rate(
            config.learning_rate,
            config.final_learning_rate,
        )

    model = PPO(
        "MlpPolicy",
        env,
        gamma=config.gamma,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        learning_rate=learning_rate,
        n_steps=config.n_steps,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    model.set_logger(create_mlflow_logger())

    print("Model initialized successfully.")
    checkpoint_dir = local_artifacts_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class CheckpointAndArtifactCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % config.save_model_every_steps:
                return True
            checkpoint = checkpoint_dir / f"ppo_checkpoint_{self.num_timesteps}_steps"
            self.model.save(checkpoint)
            mlflow.log_artifact(f"{checkpoint}.zip", artifact_path="checkpoints")
            mlflow.log_metric("checkpoint_step", self.num_timesteps)
            return True

    callback = CheckpointAndArtifactCallback()

    print("Starting training ...")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    mlflow.log_metric("training_completed", 1.0)
    mlflow.log_metric("training_actual_timesteps", model.num_timesteps)
    env.close()


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
