"""CALF-Wrapper implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.controllers.controller import Controller
from src.critic_values import critic_values


class CALFWrapper:
    def __init__(
        self,
        environment: Any,
        base_policy: Any,
        fallback_policy: Controller,
        *,
        nu: float,
        p_relax: float,
        lambda_: float,
        seed: int,
        critic_upper_bound: float,
        fallback_only: Callable[[np.ndarray], np.ndarray],
        critic_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.environment = environment
        self.base_policy = base_policy
        self.fallback_policy = fallback_policy
        self.nu = nu
        self.initial_p_relax = p_relax
        self.lambda_ = lambda_
        self.critic_upper_bound = critic_upper_bound
        self.fallback_only = fallback_only
        self.critic_transform = critic_transform
        self.p_relax = p_relax
        self.random = np.random.default_rng(seed)

    def critic_value(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(critic_values(self.base_policy, states))
        if self.critic_transform is not None:
            values = self.critic_transform(values)
        values = np.minimum(values, self.critic_upper_bound)
        return np.asarray(values).reshape(-1, 1)

    @property
    def num_envs(self) -> int:
        return self.environment.num_envs

    def step(self, base_policy_actions: np.ndarray):
        values = self.critic_value(self.states)
        fallback_only = np.asarray(self.fallback_only(self.states), dtype=bool).reshape(
            values.shape
        )
        critic_condition = (values - self.reference_value >= self.nu) & ~fallback_only
        self.reference_value = np.where(
            critic_condition,
            values,
            self.reference_value,
        )
        random_acceptance_condition = (
            self.random.random(size=values.shape) < self.p_relax
        ) & ~fallback_only
        base_policy_selected = critic_condition | random_acceptance_condition
        actions = np.where(
            base_policy_selected,
            base_policy_actions,
            self.fallback_policy.get_action(self.states),
        )
        transition = list(self.environment.step(actions))
        self.states = np.copy(transition[0])
        information = transition[-1]
        for trial, info in enumerate(information):
            info.update(
                {
                    "calfwrapper.p_relax": np.copy(self.p_relax),
                    "calfwrapper.critic_condition": critic_condition[trial, 0],
                    "calfwrapper.random_acceptance_condition": (
                        random_acceptance_condition[trial, 0]
                    ),
                    "calfwrapper.base_policy_selected": base_policy_selected[trial, 0],
                    "calfwrapper.fallback_only": fallback_only[trial, 0],
                }
            )
        self.p_relax *= self.lambda_
        transition[-1] = information
        return tuple(transition)

    def reset(self, *args, **kwargs):
        self.p_relax = self.initial_p_relax
        states = self.environment.reset(*args, **kwargs)
        self.states = states[0] if isinstance(states, tuple) else states
        self.reference_value = self.critic_value(self.states)
        return np.copy(self.states)

    def close(self) -> None:
        self.environment.close()
