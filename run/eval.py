# Main evaluation script

from dataclasses import dataclass
from typing import Callable, Any, Literal
from pathlib import Path
import os
import numpy as np
import gymnasium as gym
import mlflow
import tempfile
import json
import tyro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from src.calf_wrapper import CALFWrapper
from src.utils.mlflow import mlflow_monitoring, MlflowConfig
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.controller import Controller
from src.utils import NumpyEncoder
from src import run_path


@dataclass
class CalfConfig:
    """Configuration for CALF-Wrapper.

    CALF is a framework that combines learned policy with a stabilizing controller
    for safe reinforcement learning.
    """

    calf_change_rate: float = 0.01
    """Minimal improvement threshold for update of the best viewed critic"""

    relaxprob_init: float = 0.5
    """Initial probability of using the learned policy vs stabilizing controller"""

    relaxprob_factor: float = 0.9999
    """Decay factor for relaxation probability"""

    optimize_relaxprob: bool = False
    """Whether to optimize the relaxation probability during evaluation. Needed only to get the best possible performance for CALF-Wrapper"""


@dataclass
class EvalConfig:
    """Configuration for evaluating trained policies with optional CALF integration."""

    env_id: str
    """Gym environment identifier for evaluation"""

    mlflow: MlflowConfig
    """MLflow configuration for experiment tracking"""

    model_path: Path
    """Path to the trained model checkpoint"""

    device: str
    """Hardware device for evaluation: 'cpu' or 'cuda:n'"""

    deterministic: bool
    """Whether to use deterministic action selection during evaluation"""

    stabilizing_policy: Controller
    """Stabilizing controller configuration for safety-critical control"""

    calf: CalfConfig
    """CALF framework configuration"""

    n_steps: int
    """Number of evaluation steps to run"""

    eval_mode: Literal["fallback", "base", "calf_wrapper"] = "fallback"
    """Evaluation mode:
    - fallback: Use CALF with fallback to stabilizing controller
    - base: Evaluate pure learned policy
    - calf_wrapper: Use full CALF framework with probability-based switching
    """

    n_envs: int = 1
    """Number of parallel environments for evaluation"""

    seed: int = 42
    """Random seed for reproducibility"""


presets = {
    # Preset configurations for different environments
    # Usage: python eval.py pendulum  # For pendulum preset
    #        python eval.py cartpole  # For cartpole preset
    #        python eval.py --help    # Show all available options
    #
    # Each preset provides:
    # - Environment-specific model checkpoint
    # - Tuned stabilizing controller parameters
    # - CALF configuration for safe evaluation
    # - MLflow tracking setup
    "pendulum": (
        "Pendulum configuration",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(run_path), "mlruns"),
                experiment_name="eval/pendulum",
            ),
            env_id="Pendulum-v1",
            model_path=run_path
            / "artifacts"
            / "ppo_Pendulum-v1_9"
            / "checkpoints"
            / "ppo_checkpoint_102000_steps.zip",
            device="cpu",
            deterministic=True,
            stabilizing_policy=EnergyBasedStabilizingPolicy(
                gain=0.6,
                action_min=-2,
                action_max=2,
                switch_loc=np.cos(np.pi / 10),
                switch_vel_loc=0.2,
                pd_coeffs=[12, 4],
            ),
            eval_mode="fallback",
            calf=CalfConfig(),
            seed=42,
            n_envs=30,
            n_steps=200,
        ),
    ),
    "cartpole": (
        "Cartpole configuration",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(run_path), "mlruns"),
                experiment_name="eval/cartpole",
            ),
            env_id="CartpoleSwingupEnvLong-v0",
            n_steps=1000,
            model_path=run_path
            / "artifacts"
            / "ppo_CartpoleSwingupEnv-v0_42"
            / "checkpoints"
            / "ppo_checkpoint_270000_steps.zip",
            device="cpu",
            deterministic=True,
            stabilizing_policy=CartpoleEnergyBasedStabilizingPolicy(
                pd_coefs=[70, 10.0, 20.0, 10.0],
                gain=0.3,
                gain_pos_vel=0.5,
                action_min=-10.0,
                action_max=10.0,
            ),
            eval_mode="fallback",
            seed=42,
            n_envs=30,
            calf=CalfConfig(),
        ),
    ),
}


def run_episode(
    get_action: Callable[[np.ndarray], np.ndarray],
    env: gym.Env,
    n_steps: int,
) -> list[dict[str, Any]]:
    obs = env.reset()
    data = []
    for step in range(n_steps):
        action = get_action(obs)
        next_obs, reward, is_done, info = env.step(action)
        data.append(
            {
                "step": step,
                "obs": obs,
                "action": action,
                "reward": reward,
                "is_done": is_done,
                "info": info,
            }
        )
        obs = next_obs
    return data


def goal_reaching_rate(env_id: str, latest_obs: np.ndarray) -> float:
    if env_id == "Pendulum-v1":
        return (
            np.prod(
                np.abs(latest_obs - np.array([[1, 0, 0]]))
                < np.array([[0.05, 0.05, 0.3]]),
                axis=1,
            ).mean()
            * 100
        )
    elif env_id == "CartpoleSwingupEnvLong-v0":
        return (
            np.prod(
                np.abs(latest_obs - np.array([[0, 0, 1, 0, 0]]))
                < np.array([[0.3, 0.3, 0.05, 0.05, 0.05]]),
                axis=1,
            ).mean()
            * 100
        )
    else:
        raise ValueError(f"Unknown environment: {env_id}")


@mlflow_monitoring()
def main(config: EvalConfig):
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)

    if config.eval_mode == "fallback":
        data = run_episode(config.stabilizing_policy.get_action, env, config.n_steps)
    elif config.eval_mode == "base":
        model = PPO.load(config.model_path, device=config.device, seed=config.seed)
        data = run_episode(
            lambda obs: model.predict(obs, deterministic=config.deterministic)[0],
            env,
            config.n_steps,
        )
    elif config.eval_mode == "calf_wrapper":
        model = PPO.load(config.model_path, device=config.device, seed=config.seed)
        env = CALFWrapper(
            env,
            model,
            config.stabilizing_policy,
            calf_change_rate=config.calf.calf_change_rate,
            relaxprob_init=config.calf.relaxprob_init,
            relaxprob_factor=config.calf.relaxprob_factor,
            seed=config.seed,
        )
        data = run_episode(
            lambda obs: model.predict(obs, deterministic=config.deterministic)[0],
            env,
            config.n_steps,
        )
    else:
        raise ValueError(f"Unknown eval mode: {config.eval_mode}")
    env.close()

    # Log results
    final_rewards = np.vstack([item["reward"] for item in data]).sum(axis=0)
    for i, reward in enumerate(final_rewards):
        mlflow.log_metric("reward", reward, step=i)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = Path(tmp_dir) / f"episode_data.json"
        with open(tmp_filepath, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
        mlflow.log_artifact(str(tmp_filepath), "episode_data")
    metrics = {
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "goal_reaching_rate": float(
            goal_reaching_rate(env_id=config.env_id, latest_obs=data[-1]["obs"])
        ),
    }
    mlflow.log_metrics(metrics)

    from pprint import pprint

    print("\nEvaluation Metrics:")
    print("------------------")
    pprint(metrics, indent=2, width=60, sort_dicts=False)
    print()


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
