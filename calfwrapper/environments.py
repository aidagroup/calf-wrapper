"""Environment configurations used in the article."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.controller import Controller
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.robot_navigation import RobotNavigationConstSpeedGoalController
from src.controllers.underwaterdrone import UnderwaterDroneNominalController


@dataclass(frozen=True)
class Environment:
    gym_id: str
    horizon: int
    algorithm: Literal["ppo", "td3"]
    checkpoints: tuple[tuple[str, str], ...]
    nu: float
    critic_upper_bound: float
    fallback_policy: Controller
    goal_neighborhood_radius: float = 0.0


ENVIRONMENTS: dict[str, Environment] = {
    "pendulum": Environment(
        "Pendulum-v1",
        200,
        "ppo",
        (("Low", "low.zip"), ("Mid", "mid.zip"), ("High", "high.zip")),
        0.015502,
        0.0,
        EnergyBasedStabilizingPolicy(
            gain=0.6,
            action_min=-2,
            action_max=2,
            switch_loc=np.cos(np.pi / 10),
            switch_vel_loc=0.2,
            pd_coeffs=[12, 4],
        ),
    ),
    "cartpole": Environment(
        "CartpoleSwingupEnvLong-v0",
        1000,
        "ppo",
        (("Low", "low.zip"), ("Mid", "mid.zip"), ("High", "high.zip")),
        0.050876,
        0.0,
        CartpoleEnergyBasedStabilizingPolicy(
            pd_coefs=[81.648, 8.4735, 21.756, 11.739],
            gain=2.2,
            gain_pos_vel=0.6,
            gain_pos=1.0,
            swing_position_reference_gain=4.10,
            switch_loc=0.82,
            blend_width=0.05,
            velocity_brake_threshold=4.5,
            velocity_brake_position_threshold=0.75,
            action_min=-10.0,
            action_max=10.0,
            action_bias=2.3,
        ),
    ),
    "auv": Environment(
        "UnderwaterDrone-v0",
        1500,
        "td3",
        (("Low", "low.pt"), ("Mid", "mid.pt"), ("High", "high.pt")),
        0.04,
        0.0,
        UnderwaterDroneNominalController(),
        goal_neighborhood_radius=0.16,
    ),
    "robot": Environment(
        "RobotNavigationConstSpeedCatch-v0",
        1000,
        "td3",
        (("Low", "low.pt"), ("Mid", "mid.pt"), ("High", "high.pt")),
        0.05,
        50.0,
        RobotNavigationConstSpeedGoalController(),
    ),
}
