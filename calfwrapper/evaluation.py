"""Evaluation of the policies reported in the article."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from calfwrapper.environments import ENVIRONMENTS
from calfwrapper.operating_modes import OperatingModeParameters, operating_mode_parameters
from calfwrapper.paths import CHECKPOINTS
from calfwrapper.wrapper import CALFWrapper
from src.goal_reaching import goal_neighborhood_mask, goal_reaching_mask
from src.models.cleanrl_td3 import CleanRLTD3

Policy = Literal[
    "fallback",
    "base",
    "conservative",
    "guarded",
    "moderate",
    "balanced",
    "high",
    "almost_open",
]


@dataclass(frozen=True)
class Trial:
    episode_return: float
    goal_reached: bool
    episode_length: int
    base_policy_actions: int
    fallback_policy_actions: int
    critic_evaluations: int
    policy_sequence_sha256: str

    def as_row(self, number: int) -> dict[str, object]:
        return {
            "trial": number,
            "episode_return": self.episode_return,
            "goal_reached": self.goal_reached,
            "episode_length": self.episode_length,
            "base_policy_actions": self.base_policy_actions,
            "fallback_policy_actions": self.fallback_policy_actions,
            "critic_evaluations": self.critic_evaluations,
            "policy_sequence_sha256": self.policy_sequence_sha256,
        }


def _make_environment(environment_name: str, seed: int, rank: int) -> gym.Env:
    environment = ENVIRONMENTS[environment_name]
    if environment_name == "cartpole":
        instance = gym.make(
            environment.gym_id,
            terminate_on_out_of_bounds=False,
            saturate_state_on_out_of_bounds=True,
            reward_position_clip=5.0,
            position_termination_threshold=7.5,
            velocity_termination_threshold=12.0,
            angular_velocity_termination_threshold=15.0,
        )
    else:
        instance = gym.make(environment.gym_id)
    instance.action_space.seed(seed + rank)
    return instance


def _load_base_policy(environment_name: str, checkpoint: Path, device: str, seed: int):
    environment = ENVIRONMENTS[environment_name]
    if environment.algorithm == "ppo":
        return PPO.load(checkpoint, device=device, seed=seed)
    return CleanRLTD3.load(checkpoint, device=device)


def _latest_states(
    latest: np.ndarray,
    next_states: np.ndarray,
    finished: np.ndarray,
    information: list[dict],
    active: np.ndarray,
) -> None:
    next_states = np.copy(next_states)
    for trial, (is_finished, info) in enumerate(zip(finished, information, strict=True)):
        if is_finished and "terminal_observation" in info:
            next_states[trial] = info["terminal_observation"]
    latest[active] = next_states[active]


def _goal_events(
    environment_name: str,
    reached: np.ndarray,
    information: list[dict],
    active: np.ndarray,
) -> None:
    key = {
        "auv": "is_in_hole",
        "robot": "goal_reached",
    }.get(environment_name)
    if key is None:
        return
    for trial, info in enumerate(information):
        if active[trial]:
            reached[trial] |= bool(info.get(key, False))


def _run_trials(
    environment_name: str,
    environment: Any,
    action: _Action,
    horizon: int,
) -> list[Trial]:
    states = cast(np.ndarray, environment.reset())
    latest_states = np.copy(states)
    active = np.ones(environment.num_envs, dtype=bool)
    reached = np.zeros(environment.num_envs, dtype=bool)
    rewards: list[np.ndarray] = []
    episode_lengths = np.zeros(environment.num_envs, dtype=np.int64)
    base_policy_actions = np.zeros(environment.num_envs, dtype=np.int64)
    fallback_policy_actions = np.zeros(environment.num_envs, dtype=np.int64)
    critic_evaluations = np.zeros(environment.num_envs, dtype=np.int64)
    sequence_hashes = [hashlib.sha256() for _ in range(environment.num_envs)]

    for _ in range(horizon):
        selected_actions = action(states)
        next_states, reward, finished, information = environment.step(selected_actions)
        next_states = cast(np.ndarray, next_states)
        rewards.append(np.copy(reward) * active)
        episode_lengths += active
        _latest_states(latest_states, next_states, finished, information, active)
        _goal_events(environment_name, reached, information, active)

        for trial, info in enumerate(information):
            if not active[trial]:
                continue
            if "calfwrapper.base_policy_selected" in info:
                critic_evaluations[trial] += 1
                base_selected = bool(info["calfwrapper.base_policy_selected"])
            else:
                base_selected = action.is_base_policy
            if base_selected:
                base_policy_actions[trial] += 1
                sequence_hashes[trial].update(b"B")
            else:
                fallback_policy_actions[trial] += 1
                sequence_hashes[trial].update(b"F")

        active &= ~finished
        states = next_states
        if not np.any(active):
            break

    environment.close()
    reached |= goal_reaching_mask(ENVIRONMENTS[environment_name].gym_id, latest_states)
    episode_returns = np.vstack(rewards).sum(axis=0)
    return [
        Trial(
            episode_return=float(episode_returns[trial]),
            goal_reached=bool(reached[trial]),
            episode_length=int(episode_lengths[trial]),
            base_policy_actions=int(base_policy_actions[trial]),
            fallback_policy_actions=int(fallback_policy_actions[trial]),
            critic_evaluations=int(critic_evaluations[trial]),
            policy_sequence_sha256=sequence_hashes[trial].hexdigest(),
        )
        for trial in range(environment.num_envs)
    ]


class _Action:
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        *,
        is_base_policy: bool,
    ):
        self.function = function
        self.is_base_policy = is_base_policy

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return self.function(states)


def evaluate(
    environment_name: str,
    checkpoint_name: str,
    policy: Policy,
    trial_count: int,
    seed: int,
    device: str,
    parameters: OperatingModeParameters | None = None,
    nu: float | None = None,
    critic_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Trial]:
    """Evaluate one policy using one fixed set of initial conditions."""

    config = ENVIRONMENTS[environment_name]
    environment: Any = DummyVecEnv(
        [
            lambda rank=rank: _make_environment(environment_name, seed, rank)
            for rank in range(trial_count)
        ]
    )
    environment.seed(seed)

    if policy == "fallback":
        action = _Action(config.fallback_policy.get_action, is_base_policy=False)
    else:
        checkpoint = CHECKPOINTS / "base" / environment_name / checkpoint_name
        base_policy = _load_base_policy(environment_name, checkpoint, device, seed)

        def base_action(states: np.ndarray) -> np.ndarray:
            return base_policy.predict(states, deterministic=True)[0]

        action = _Action(base_action, is_base_policy=True)
        if policy != "base":
            parameters = parameters or operating_mode_parameters(policy, config.horizon)
            environment = CALFWrapper(
                environment,
                base_policy,
                config.fallback_policy,
                nu=config.nu if nu is None else nu,
                p_relax=parameters.p_relax,
                lambda_=parameters.lambda_,
                seed=seed,
                critic_upper_bound=config.critic_upper_bound,
                fallback_only=lambda states: goal_neighborhood_mask(
                    config.gym_id,
                    states,
                    config.goal_neighborhood_radius,
                ),
                critic_transform=critic_transform,
            )

    return _run_trials(environment_name, environment, action, config.horizon)
