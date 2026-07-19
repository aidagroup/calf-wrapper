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
import csv
import tyro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import PPO

from gymnasium.wrappers.record_video import RecordVideo
from src.calf_wrapper import CALFWrapper
from src.utils.mlflow import mlflow_monitoring, MlflowConfig
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.controller import Controller
from src.utils import NumpyEncoder
from src import run_path
from src.controllers.robot_navigation import RobotNavigationConstSpeedGoalController
from src.controllers.underwaterdrone import UnderwaterDroneNominalController
from src.envs.underwaterdrone import HOLE_WIDTH, TOP_Y
from src.models.cleanrl_td3 import CleanRLTD3


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

    algorithm: Literal["ppo", "cleanrl_td3"]
    """Base-policy algorithm used to load the checkpoint"""

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

    record_video: bool = False
    """Whether to save a video of the evaluation"""

    video_folder: Path = run_path / "artifacts" / "videos"
    """Path to the folder where the video will be saved"""

    checkpoint_stage: str = "none"
    """Semantic checkpoint stage: none, early, mid, or late"""


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
            algorithm="ppo",
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
            algorithm="ppo",
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
    "underwater-drone": (
        "Underwater-drone TD3 configuration",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(run_path), "mlruns"),
                experiment_name="eval/underwater-drone",
            ),
            env_id="UnderwaterDrone-v0",
            model_path=run_path
            / "artifacts"
            / "td3_UnderwaterDrone-v0_0"
            / "checkpoints"
            / "td3_checkpoint_3000000_steps.pt",
            device="cpu",
            deterministic=True,
            algorithm="cleanrl_td3",
            stabilizing_policy=UnderwaterDroneNominalController(),
            eval_mode="fallback",
            calf=CalfConfig(),
            seed=42,
            n_envs=30,
            n_steps=1500,
        ),
    ),
    "robot-navigation": (
        "Robot-navigation TD3 configuration",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(run_path), "mlruns"),
                experiment_name="eval/robot-navigation",
            ),
            env_id="RobotNavigationConstSpeedCatch-v0",
            model_path=run_path
            / "artifacts"
            / "td3_RobotNavigationConstSpeedCatch-v0_1"
            / "checkpoints"
            / "td3_checkpoint_3000000_steps.pt",
            device="cpu",
            deterministic=True,
            algorithm="cleanrl_td3",
            stabilizing_policy=RobotNavigationConstSpeedGoalController(),
            eval_mode="fallback",
            calf=CalfConfig(),
            seed=42,
            n_envs=30,
            n_steps=1000,
        ),
    ),
}


def run_episode(
    get_action: Callable[[np.ndarray], np.ndarray],
    env: VecEnv,
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
    env.close()
    return data


def goal_reaching_mask(env_id: str, latest_obs: np.ndarray) -> np.ndarray:
    if env_id == "Pendulum-v1":
        return np.prod(
            np.abs(latest_obs - np.array([[1, 0, 0]])) < np.array([[0.05, 0.05, 0.3]]),
            axis=1,
        )
    elif env_id == "CartpoleSwingupEnvLong-v0":
        return np.prod(
            np.abs(latest_obs - np.array([[0, 0, 1, 0, 0]]))
            < np.array([[0.3, 0.3, 0.05, 0.05, 0.05]]),
            axis=1,
        )
    elif env_id == "UnderwaterDrone-v0":
        return (latest_obs[:, 1] >= TOP_Y) & (
            np.abs(latest_obs[:, 0]) <= HOLE_WIDTH / 2.0
        )
    elif env_id == "RobotNavigationConstSpeedCatch-v0":
        return np.linalg.norm(latest_obs[:, 0:2] - latest_obs[:, 4:6], axis=1) <= 0.05
    else:
        raise ValueError(f"Unknown environment: {env_id}")


def goal_reaching_rate(env_id: str, latest_obs: np.ndarray) -> float:
    return float(np.mean(goal_reaching_mask(env_id, latest_obs)) * 100)


def trial_successes(env_id: str, data: list[dict[str, Any]]) -> np.ndarray:
    """Return whether each fixed-horizon trial reached its goal at least once."""

    successes = goal_reaching_mask(env_id, data[-1]["obs"])
    info_key = {
        "UnderwaterDrone-v0": "is_in_hole",
        "RobotNavigationConstSpeedCatch-v0": "goal_reached",
    }.get(env_id)
    if info_key is None:
        return successes

    for item in data:
        for trial, info in enumerate(item["info"]):
            successes[trial] |= bool(info.get(info_key, False))
    return successes


def make_env(
    env_id: str,
    rank: int,
    seed: int,
    wrapper_class: Callable[[gym.Env], gym.Env],
    wrapper_kwargs: dict[str, Any],
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(seed + rank)
        if wrapper_class is not None:
            env = wrapper_class(env, **wrapper_kwargs)
        return env

    return _init


def load_model(config: EvalConfig):
    if config.algorithm == "ppo":
        return PPO.load(config.model_path, device=config.device, seed=config.seed)
    return CleanRLTD3.load(config.model_path, device=config.device)


@mlflow_monitoring()
def main(config: EvalConfig):
    if config.record_video:
        video_folder = config.video_folder / (
            config.mlflow.experiment_name
            + "_"
            + (
                config.mlflow.run_name
                if config.mlflow.run_name is not None
                else config.env_id + "_" + config.eval_mode + "_" + str(config.seed)
            )
        )

    env = DummyVecEnv(
        [
            make_env(
                config.env_id,
                rank,
                config.seed,
                wrapper_class=RecordVideo if config.record_video else None,
                wrapper_kwargs=(
                    {"video_folder": video_folder / f"env_{rank:03d}"}
                    if config.record_video
                    else None
                ),
            )
            for rank in range(config.n_envs)
        ]
    )
    env.seed(config.seed)

    if config.eval_mode == "fallback":
        data = run_episode(config.stabilizing_policy.get_action, env, config.n_steps)
    elif config.eval_mode == "base":
        model = load_model(config)
        data = run_episode(
            lambda obs: model.predict(obs, deterministic=config.deterministic)[0],
            env,
            config.n_steps,
        )
    elif config.eval_mode == "calf_wrapper":
        model = load_model(config)
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
    successes = trial_successes(config.env_id, data)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = Path(tmp_dir) / f"episode_data.json"
        with open(tmp_filepath, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
        mlflow.log_artifact(str(tmp_filepath), "episode_data")
        trial_filepath = Path(tmp_dir) / "trial_metrics.csv"
        with open(trial_filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial", "reward", "goal_reached"])
            writer.writeheader()
            for trial, (reward, success) in enumerate(zip(final_rewards, successes)):
                writer.writerow(
                    {
                        "trial": trial,
                        "reward": float(reward),
                        "goal_reached": bool(success),
                    }
                )
        mlflow.log_artifact(str(trial_filepath), "raw")

    if config.record_video:
        mlflow.log_artifact(video_folder, "video")

    confidence_interval = float(
        1.96 * np.std(final_rewards) / np.sqrt(len(final_rewards))
    )
    metrics = {
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "reward_ci95_half_width": confidence_interval,
        "goal_reaching_rate": float(np.mean(successes) * 100),
        "n_trials": float(len(final_rewards)),
    }
    calf_infos = [
        info
        for item in data
        for info in item["info"]
        if "calf.base_action_applied" in info
    ]
    if calf_infos:
        base_fraction = float(
            np.mean([info["calf.base_action_applied"] for info in calf_infos])
        )
        metrics["base_action_fraction"] = base_fraction
        metrics["fallback_action_fraction"] = 1.0 - base_fraction
    mlflow.log_metrics(metrics)

    from pprint import pprint

    print("\nEvaluation Metrics:")
    print("------------------")
    pprint(metrics, indent=2, width=60, sort_dicts=False)
    print()


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(presets)
    main(config)
