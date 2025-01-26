from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import tyro
from dataclasses import dataclass, field
from src.utilities.mlflow_logger import MlflowConfig
from src.utilities import mlflow_logger
import os
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.callback.plotting_callback import PlottingCallback
import mlflow
import gymnasium as gym

current_dir = Path(__file__).parent

gym.register(
    id="InvertedPendulum-swingup",
    entry_point="src.mygym.inverted_pendulum_upswing:InvertedPendulumSwingupEnv",
    max_episode_steps=200,
)


@dataclass
class ExperimentConfig:
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
            experiment_name=current_dir.name,
        )
    )
    local_artifacts_path: Path = current_dir / "artifacts"
    local_logs_path: Path = current_dir / "logs"
    env_id: str = "Pendulum-v1"
    n_envs: int = 1
    gamma: float = 0.98
    use_sde: bool = True
    sde_sample_freq: int = 4
    learning_rate: float = 1e-3
    verbose: int = 1
    seed: int = 42
    total_timesteps: int = 300_000
    n_steps: int = 2048
    save_model_every_steps: int = 2048


@mlflow_logger.mlflow_monitoring()
def main(config: ExperimentConfig):
    # Create the environment
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)

    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        gamma=config.gamma,
        # Using https://proceedings.mlr.press/v164/raffin22a.html
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        learning_rate=config.learning_rate,
        verbose=config.verbose,
    )

    model.set_logger(mlflow_logger.create_logger())

    print("Model initialized successfully.")
    # Set up a checkpoint callback to save the model every 'save_freq' steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=config.local_artifacts_path
        / "checkpoints"
        / f"ppo_{config.env_id}_{config.seed}",  # Directory to save the model
        name_prefix=f"ppo_checkpoint",
    )

    mlflow_checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=os.path.join(
            mlflow.get_artifact_uri()[len("file://") :], "checkpoints"
        ),  # Directory to save the model
        name_prefix=f"ppo_{config.env_id}",
    )

    # Instantiate a plotting callback for the live learning curve
    path = (
        config.local_logs_path
        / f"episode_rewards_ppo_{config.env_id}_{config.seed}.csv"
    )
    episode_reward_callback = PlottingCallback(
        save_path=path,
        is_console_mode=True,
    )
    os.makedirs(str(path.parent), exist_ok=True)

    # Combine both callbacks using CallbackList
    callback = CallbackList(
        [checkpoint_callback, mlflow_checkpoint_callback, episode_reward_callback]
    )

    print("Starting training ...")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)


if __name__ == "__main__":
    config = tyro.cli(ExperimentConfig)
    main(config)
