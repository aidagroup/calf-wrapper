from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces

from src.envs.robot_navigation import (
    RobotNavigationConfig,
    RobotNavigationEnv,
    RobotNavigationMetricsCollector,
)


@dataclass
class RobotNavigationConstSpeedConfig(RobotNavigationConfig):
    """Config for constant-speed robot navigation (1D action: angular velocity only)."""

    const_speed: Optional[float] = None


class RobotNavigationConstSpeedEnv(RobotNavigationEnv):
    """Robot navigation with constant speed. Action is 1D: angular velocity only."""

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        config: Optional[RobotNavigationConstSpeedConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if config is None:
            config = RobotNavigationConstSpeedConfig()
        super().__init__(render_mode=render_mode, config=config, seed=seed)

        self._const_speed = (
            config.const_speed if config.const_speed is not None else config.max_speed
        )

        self.action_space = spaces.Box(
            low=np.array([-self.config.max_angular_velocity], dtype=np.float32),
            high=np.array([self.config.max_angular_velocity], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        angular_velocity = float(
            np.clip(action[0], -self.config.max_angular_velocity, self.config.max_angular_velocity)
        )
        full_action = np.array([self._const_speed, angular_velocity], dtype=np.float32)
        return super().step(full_action)


class ConstSpeedGoalController:
    """Controller for const-speed env. Returns 1D action [angular_velocity]."""

    def __init__(
        self,
        turn_gain: float = 1.0,
        max_turn_rate: float = math.pi,
    ) -> None:
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = float(max_turn_rate)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        batched = obs.ndim > 1
        obs = obs if batched else obs[None, :]

        robot = obs[:, 0:2]
        heading_cos = obs[:, 2:3]
        heading_sin = obs[:, 3:4]
        current_angle = np.arctan2(heading_sin, heading_cos)
        goal = obs[:, 4:6]

        delta = goal - robot
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        desired_angle = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
        desired_angle = np.where(distance < 1e-8, current_angle, desired_angle)

        angle_error = desired_angle - current_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        angular_velocity = np.clip(
            self.turn_gain * angle_error, -self.max_turn_rate, self.max_turn_rate
        )

        actions = angular_velocity.astype(np.float32)
        return actions if batched else actions[0]

