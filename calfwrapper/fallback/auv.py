import numpy as np

from calfwrapper.environments.auv import (
    DRONE_MASS,
    GRAVITY,
    MAX_F_LAT,
    MAX_F_LONG,
    TOP_Y,
)
from calfwrapper.fallback.base import FallbackPolicy


class AUVFallbackPolicy(FallbackPolicy):
    """PD fallback that drives the underwater drone toward the surface opening."""

    def __init__(
        self,
        kp_y: float = 0.6,
        kd_y: float = 1.2,
        kp_x: float = 1.5,
        kd_x: float = 0.8,
        target_y_margin: float = 0.1,
    ) -> None:
        self.kp_y = kp_y
        self.kd_y = kd_y
        self.kp_x = kp_x
        self.kd_x = kd_x
        self.target_y_margin = target_y_margin

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        batched = observation.ndim > 1
        obs = observation if batched else observation[None, :]

        x = obs[:, 0:1]
        y = obs[:, 1:2]
        cos_theta = obs[:, 2:3]
        sin_theta = obs[:, 3:4]
        velocity_x = obs[:, 4:5]
        velocity_y = obs[:, 5:6]

        force_y = (
            GRAVITY * DRONE_MASS
            + self.kp_y * (TOP_Y + self.target_y_margin - y)
            - self.kd_y * velocity_y
        )
        force_x = -self.kp_x * x - self.kd_x * velocity_x

        force_longitudinal = cos_theta * force_x + sin_theta * force_y
        force_lateral = -sin_theta * force_x + cos_theta * force_y
        actions = np.hstack(
            [
                np.clip(force_longitudinal, -MAX_F_LONG, MAX_F_LONG),
                np.clip(force_lateral, -MAX_F_LAT, MAX_F_LAT),
            ]
        ).astype(np.float32)
        return actions if batched else actions[0]
