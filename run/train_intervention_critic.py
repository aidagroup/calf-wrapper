"""Train a finite-horizon fallback goal-cost critic for intervention evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from stable_baselines3 import PPO
import tyro

from src import run_path
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.controller import Controller
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.robot_navigation import RobotNavigationConstSpeedGoalController
from src.controllers.underwaterdrone import UnderwaterDroneNominalController
from src.intervention_training import (
    collect_goal_cost_dataset,
    fit_goal_cost_critic,
)
from src.models.cleanrl_td3 import CleanRLTD3


@dataclass
class CriticTrainingConfig:
    env_id: str
    model_path: Path
    algorithm: Literal["ppo", "cleanrl_td3"]
    fallback_policy: Controller
    horizon: int
    output_dir: Path
    device: str = "cpu"
    deterministic: bool = True
    n_anchors: int = 2000
    prefix_fallback_probability: float = 0.5
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    seed: int = 42


presets = {
    "pendulum": (
        "Pendulum goal-cost critic",
        CriticTrainingConfig(
            env_id="Pendulum-v1",
            model_path=run_path
            / "artifacts/ppo_Pendulum-v1_9/checkpoints/ppo_checkpoint_102000_steps.zip",
            algorithm="ppo",
            fallback_policy=EnergyBasedStabilizingPolicy(
                gain=0.6,
                action_min=-2,
                action_max=2,
                switch_loc=np.cos(np.pi / 10),
                switch_vel_loc=0.2,
                pd_coeffs=[12, 4],
            ),
            horizon=200,
            output_dir=run_path / "artifacts/intervention/pendulum",
        ),
    ),
    "cartpole": (
        "CartPole goal-cost critic",
        CriticTrainingConfig(
            env_id="CartpoleSwingupEnvLong-v0",
            model_path=run_path
            / "artifacts/ppo_CartpoleSwingupEnv-v0_42/checkpoints/ppo_checkpoint_270000_steps.zip",
            algorithm="ppo",
            fallback_policy=CartpoleEnergyBasedStabilizingPolicy(
                pd_coefs=[77.76, 8.07, 20.72, 11.18],
                gain=2.5,
                gain_pos_vel=0.6,
                gain_pos=0.8,
                swing_position_reference_gain=4.10,
                switch_loc=0.82,
                blend_width=0.05,
                action_min=-10.0,
                action_max=10.0,
            ),
            horizon=1000,
            output_dir=run_path / "artifacts/intervention/cartpole",
        ),
    ),
    "underwater-drone": (
        "AUV goal-cost critic",
        CriticTrainingConfig(
            env_id="UnderwaterDrone-v0",
            model_path=run_path
            / "artifacts/td3_UnderwaterDrone-v0_0/checkpoints/td3_checkpoint_3000000_steps.pt",
            algorithm="cleanrl_td3",
            fallback_policy=UnderwaterDroneNominalController(),
            horizon=1500,
            output_dir=run_path / "artifacts/intervention/underwater-drone",
        ),
    ),
    "robot-navigation": (
        "Treasure robot goal-cost critic",
        CriticTrainingConfig(
            env_id="RobotNavigationConstSpeedCatch-v0",
            model_path=run_path
            / "artifacts/td3_RobotNavigationConstSpeedCatch-v0_1/checkpoints/td3_checkpoint_3000000_steps.pt",
            algorithm="cleanrl_td3",
            fallback_policy=RobotNavigationConstSpeedGoalController(),
            horizon=1000,
            output_dir=run_path / "artifacts/intervention/robot-navigation",
        ),
    ),
}


def load_model(config: CriticTrainingConfig):
    if config.algorithm == "ppo":
        return PPO.load(config.model_path, device=config.device, seed=config.seed)
    return CleanRLTD3.load(config.model_path, device=config.device)


def main(config: CriticTrainingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(config)
    last_reported = -1

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        percent = 100 * completed // total
        bucket = percent // 10
        if bucket != last_reported or completed == total:
            print(f"Collected {completed}/{total} paired anchors ({percent}%)")
            last_reported = bucket

    dataset = collect_goal_cost_dataset(
        env_id=config.env_id,
        model=model,
        fallback_policy=config.fallback_policy,
        horizon=config.horizon,
        n_anchors=config.n_anchors,
        seed=config.seed,
        prefix_fallback_probability=config.prefix_fallback_probability,
        deterministic=config.deterministic,
        progress=report_progress,
    )
    dataset_path = config.output_dir / "goal_cost_dataset.npz"
    dataset.save(dataset_path)

    critic, metrics = fit_goal_cost_critic(
        dataset=dataset,
        horizon=config.horizon,
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    metadata = {
        "environment": config.env_id,
        "base_model_path": str(config.model_path),
        "algorithm": config.algorithm,
        "n_anchors": config.n_anchors,
        "prefix_fallback_probability": config.prefix_fallback_probability,
        "seed": config.seed,
        "metrics": metrics,
    }
    critic_path = config.output_dir / "goal_cost_critic.pt"
    critic.save(critic_path, metadata=metadata)
    metrics_path = config.output_dir / "training_summary.json"
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    if metrics["paired_outcomes"]["base_cost_worse_rate"] == 0.0:
        print(
            "WARNING: no collected anchor assigned a higher goal cost to the "
            "base action than to the fallback action."
        )
    print(f"Saved goal-cost critic to {critic_path}")


if __name__ == "__main__":
    main(tyro.extras.overridable_config_cli(presets))
