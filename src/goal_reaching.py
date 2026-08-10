"""Goal-set predicates shared by evaluation and intervention training."""

from __future__ import annotations

import numpy as np

from src.envs.underwaterdrone import HOLE_WIDTH, TOP_Y

SUPPORTED_ENVIRONMENTS = (
    "Pendulum-v1",
    "CartpoleSwingupEnvLong-v0",
    "UnderwaterDrone-v0",
    "RobotNavigationConstSpeedCatch-v0",
)


def goal_reaching_mask(env_id: str, observations: np.ndarray) -> np.ndarray:
    """Return one goal-set membership flag per observation."""

    latest_obs = np.asarray(observations)
    if latest_obs.ndim == 1:
        latest_obs = latest_obs[None, :]

    if env_id == "Pendulum-v1":
        return np.all(
            np.abs(latest_obs - np.array([[1.0, 0.0, 0.0]]))
            < np.array([[0.05, 0.05, 0.3]]),
            axis=1,
        )
    if env_id == "CartpoleSwingupEnvLong-v0":
        return np.all(
            np.abs(latest_obs - np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]))
            < np.array([[0.3, 0.3, 0.05, 0.05, 0.05]]),
            axis=1,
        )
    if env_id == "UnderwaterDrone-v0":
        return (latest_obs[:, 1] >= TOP_Y) & (
            np.abs(latest_obs[:, 0]) <= HOLE_WIDTH / 2.0
        )
    if env_id == "RobotNavigationConstSpeedCatch-v0":
        return np.linalg.norm(latest_obs[:, 0:2] - latest_obs[:, 4:6], axis=1) <= 0.05
    raise ValueError(f"Unknown environment: {env_id}")


def observation_reached_goal(env_id: str, observation: np.ndarray) -> bool:
    """Return whether one observation belongs to the environment goal set."""

    return bool(goal_reaching_mask(env_id, np.asarray(observation))[0])
