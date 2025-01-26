from dataclasses import dataclass, field
from typing import Callable, Any, Literal
from pathlib import Path
import os
import numpy as np
import gymnasium as gym
import mlflow
import tempfile
import json
import tyro
from typing import Optional
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from src.calf_wrapper import CALFWrapper
from src.utils.mlflow import mlflow_monitoring, MlflowConfig
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.controller import Controller
from src.utils import NumpyEncoder

current_dir = Path(__file__).parent


@dataclass
class CalfConfig:
    calf_change_rate: float = 0.01
    relaxprob_init: float = 0.5
    relaxprob_factor: float = 0.9999


@dataclass
class EvalConfig:
    env_id: str

    mlflow: MlflowConfig

    model_path: Path
    device: str
    deterministic: bool
    stabilizing_policy: Controller

    calf: CalfConfig
    n_steps: int
    eval_mode: Literal["fallback", "checkpoint", "calf_wrapper"] = "fallback"

    n_envs: int = 1
    seed: int = 42


presets = {
    "pendulum": (
        "Evaluation of pendulum with PPO",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
                experiment_name="eval/pendulum",
            ),
            env_id="Pendulum-v1",
            model_path=current_dir
            / "artifacts"
            / "ppo_Pendulum-v1_42"
            / "checkpoints"
            / "ppo_checkpoint_301056_steps.zip",
            device="cpu",
            deterministic=True,
            stabilizing_policy=EnergyBasedStabilizingPolicy(
                gain=1.0,
                action_min=-2,
                action_max=2,
                switch_loc=np.cos(np.pi / 10),
                switch_vel_loc=0.2,
                pd_coeffs=[12, 4],
            ),
            eval_mode="fallback",
            calf=CalfConfig(),
            seed=42,
            n_envs=1,
            n_steps=200,
        ),
    ),
    "cartpole": (
        "Evaluation of cartpole with PPO",
        EvalConfig(
            mlflow=MlflowConfig(
                tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
                experiment_name="eval/cartpole",
            ),
            env_id="CartpoleSwingupEnvLong-v0",
            n_steps=200,
            model_path=current_dir
            / "artifacts"
            / "ppo_CartpoleSwingupEnv-v0_42"
            / "checkpoints"
            / "ppo_checkpoint_53248_steps.zip",
            device="cpu",
            deterministic=True,
            stabilizing_policy=CartpoleEnergyBasedStabilizingPolicy(
                pd_coefs=[70, 10.0, 20.0, 10.0],
                gain=0.5,
                gain_pos_vel=0.5,
                action_min=-10.0,
                action_max=10.0,
            ),
            eval_mode="fallback",
            seed=42,
            n_envs=1,
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


@mlflow_monitoring()
def main(config: EvalConfig):
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)

    if config.eval_mode == "fallback":
        data = run_episode(config.stabilizing_policy.get_action, env, config.n_steps)
    elif config.eval_mode == "checkpoint":
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
    final_rewards = np.vstack([item["reward"] for item in data]).sum(axis=0)
    for i, reward in enumerate(final_rewards):
        mlflow.log_metric("reward", reward, step=i)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = Path(tmp_dir) / f"episode_data.json"
        with open(tmp_filepath, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
        mlflow.log_artifact(str(tmp_filepath), "episode_data")
    metrics = {
        "mean_reward": np.mean(final_rewards),
        "std_reward": np.std(final_rewards),
        "median_reward": np.median(final_rewards),
        "q1_reward": np.quantile(final_rewards, 0.25),
        "q3_reward": np.quantile(final_rewards, 0.75),
        "min_reward": np.min(final_rewards),
        "max_reward": np.max(final_rewards),
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
