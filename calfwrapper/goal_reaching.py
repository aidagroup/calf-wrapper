"""Goal-set predicates shared by evaluation and intervention training."""

from __future__ import annotations

import numpy as np

from calfwrapper.environments.auv import HOLE_WIDTH, TOP_Y

SUPPORTED_ENVIRONMENTS = (
    "Pendulum-v1",
    "CALFWrapper/CartPoleSwingUpLong-v0",
    "CALFWrapper/ContaminatedZoneAUV-v0",
    "CALFWrapper/TreasureCollectingRobot-v0",
)


def goal_reaching_mask(env_id: str, observations: np.ndarray) -> np.ndarray:
    """Return one goal-set membership flag per observation."""

    latest_obs = np.asarray(observations)
    if latest_obs.ndim == 1:
        latest_obs = latest_obs[None, :]

    if env_id == "Pendulum-v1":
        return np.all(
            np.abs(latest_obs - np.array([[1.0, 0.0, 0.0]])) < np.array([[0.05, 0.05, 0.3]]),
            axis=1,
        )
    if env_id == "CALFWrapper/CartPoleSwingUpLong-v0":
        return np.all(
            np.abs(latest_obs - np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]))
            < np.array([[0.3, 0.3, 0.05, 0.05, 0.05]]),
            axis=1,
        )
    if env_id == "CALFWrapper/ContaminatedZoneAUV-v0":
        return (latest_obs[:, 1] >= TOP_Y) & (np.abs(latest_obs[:, 0]) <= HOLE_WIDTH / 2.0)
    if env_id == "CALFWrapper/TreasureCollectingRobot-v0":
        return np.linalg.norm(latest_obs[:, 0:2] - latest_obs[:, 4:6], axis=1) <= 0.05
    raise ValueError(f"Unknown environment: {env_id}")


def goal_neighborhood_mask(env_id: str, observations: np.ndarray, distance: float) -> np.ndarray:
    """Return whether each observation is at most ``distance`` from the goal set."""

    if distance < 0.0:
        raise ValueError("goal-neighborhood distance must be nonnegative")
    latest_obs = np.asarray(observations)
    if latest_obs.ndim == 1:
        latest_obs = latest_obs[None, :]
    if env_id == "CALFWrapper/ContaminatedZoneAUV-v0":
        horizontal = np.maximum(np.abs(latest_obs[:, 0]) - HOLE_WIDTH / 2.0, 0.0)
        vertical = np.maximum(TOP_Y - latest_obs[:, 1], 0.0)
        return np.hypot(horizontal, vertical) <= distance
    if distance == 0.0:
        return goal_reaching_mask(env_id, latest_obs)
    raise ValueError(f"a positive goal-neighborhood distance is not configured for {env_id}")


def observation_reached_goal(env_id: str, observation: np.ndarray) -> bool:
    """Return whether one observation belongs to the environment goal set."""

    return bool(goal_reaching_mask(env_id, np.asarray(observation))[0])
