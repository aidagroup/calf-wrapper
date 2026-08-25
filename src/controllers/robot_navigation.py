import math

import numpy as np

from src.controllers.controller import Controller
from src.envs.robot_navigation_const_speed import RobotNavigationConstSpeedConfig


class RobotNavigationConstSpeedGoalController(Controller):
    """Goal-reaching fallback for the one-dimensional steering action."""

    def __init__(
        self,
        turn_gain: float = 1.0,
        max_turn_rate: float | None = None,
    ) -> None:
        config = RobotNavigationConstSpeedConfig()
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = (
            float(config.max_angular_velocity) if max_turn_rate is None else float(max_turn_rate)
        )

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        batched = observation.ndim > 1
        obs = observation if batched else observation[None, :]

        robot = obs[:, 0:2]
        heading = np.arctan2(obs[:, 3:4], obs[:, 2:3])
        goal = obs[:, 4:6]
        delta = goal - robot
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        desired_heading = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
        desired_heading = np.where(distance < 1e-8, heading, desired_heading)
        heading_error = (desired_heading - heading + math.pi) % (2 * math.pi) - math.pi
        action = np.clip(
            self.turn_gain * heading_error,
            -self.max_turn_rate,
            self.max_turn_rate,
        ).astype(np.float32)
        return action if batched else action[0]
