"""Evaluation of the Lagrangian baselines reported in the article."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import gymnasium as gym
import torch

from calfwrapper.evaluation import Trial
from calfwrapper.paths import CHECKPOINTS

EVALUATION_TRIALS = 100

CHECKPOINT_ENVIRONMENT_IDS = {
    "CartpoleSwingupEnvLong-v0": "CALFWrapper/CartPoleSwingUpLong-v0",
    "UnderwaterDrone-v0": "CALFWrapper/ContaminatedZoneAUV-v0",
    "RobotNavigationConstSpeedCatch-v0": "CALFWrapper/TreasureCollectingRobot-v0",
}


def _evaluation_seed(environment: str) -> int:
    return 20260801 if environment == "cartpole" else 42


def _ppo_trials(checkpoint: Path, device: torch.device, evaluation_seed: int) -> list[Trial]:
    from calfwrapper.training import ppo_lagrangian as ppo

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_arguments = ppo.Args(**payload["args"])
    arguments = replace(
        checkpoint_arguments,
        env_id=CHECKPOINT_ENVIRONMENT_IDS.get(
            checkpoint_arguments.env_id, checkpoint_arguments.env_id
        ),
        device=str(device),
        evaluation_episodes=EVALUATION_TRIALS,
        evaluation_seed=evaluation_seed,
    )
    environments = gym.vector.SyncVectorEnv(
        [ppo.make_env(arguments.env_id, arguments.horizon, arguments.seed, 0)]
    )
    observation_shape = cast(tuple[int, ...], environments.single_observation_space.shape)
    action_shape = cast(tuple[int, ...], environments.single_action_space.shape)
    agent = ppo.Agent(environments).to(device)
    ppo.load_agent_checkpoint(
        checkpoint,
        checkpoint_arguments,
        agent,
        observation_shape,
        action_shape,
        device,
    )
    environments.close()
    result = ppo.evaluate(agent, arguments, device, stochastic=False)
    return _trials(result["trials"])


def _td3_trials(checkpoint: Path, device: torch.device, evaluation_seed: int) -> list[Trial]:
    from calfwrapper.training import td3_lagrangian as td3

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_arguments = td3.Args(**payload["args"])
    arguments = replace(
        checkpoint_arguments,
        env_id=CHECKPOINT_ENVIRONMENT_IDS.get(
            checkpoint_arguments.env_id, checkpoint_arguments.env_id
        ),
        device=str(device),
        evaluation_episodes=EVALUATION_TRIALS,
        evaluation_seed=evaluation_seed,
    )
    environments = gym.vector.SyncVectorEnv(
        [td3.make_env(arguments.env_id, arguments.horizon, arguments.seed)]
    )
    observation_shape = cast(tuple[int, ...], environments.single_observation_space.shape)
    action_shape = cast(tuple[int, ...], environments.single_action_space.shape)
    action_space = cast(gym.spaces.Box, environments.single_action_space)
    actor = td3.Actor(
        int(observation_shape[0]),
        action_space,
    ).to(device)
    td3.load_actor_checkpoint(
        checkpoint,
        checkpoint_arguments,
        actor,
        observation_shape,
        action_shape,
        device,
    )
    environments.close()
    result = td3.evaluate(actor, arguments, device)
    return _trials(result["trials"])


def _trials(rows: object) -> list[Trial]:
    assert isinstance(rows, list)
    return [
        Trial(
            episode_return=float(row["episode_return"]),
            goal_reached=bool(row["goal_reached"]),
            episode_length=int(row["episode_length"]),
            base_policy_actions=0,
            fallback_policy_actions=0,
            critic_evaluations=0,
            policy_sequence_sha256="",
        )
        for row in rows
    ]


def evaluate_lagrangian(
    environment: str,
    stage: str,
    device_name: str,
) -> list[Trial]:
    """Evaluate one published Lagrangian checkpoint."""

    checkpoint = CHECKPOINTS / "lagrangian" / environment / f"{stage.lower()}.pt"
    device = torch.device(device_name)
    evaluation_seed = _evaluation_seed(environment)
    if environment in {"pendulum", "cartpole"}:
        return _ppo_trials(checkpoint, device, evaluation_seed)
    return _td3_trials(checkpoint, device, evaluation_seed)
