"""Environment-specific CMDP costs used by the SOOPER comparison.

Costs are deliberately separate from goal-reaching metrics.  The underwater
task has a native high-cost region and is therefore the primary SOOPER study.
The remaining definitions provide consistent smoke/integration coverage and
are reported as operational constraints, not as new formal safety claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


ArrayCost = Callable[[np.ndarray], np.ndarray]
InfoCost = Callable[[dict[str, Any], np.ndarray], float]


@dataclass(frozen=True)
class CostDefinition:
    env_id: str
    name: str
    description: str
    from_observation: ArrayCost
    from_transition_info: InfoCost

    def observation_cost(self, observations: np.ndarray) -> np.ndarray:
        values = self.from_observation(np.asarray(observations, dtype=np.float32))
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def transition_cost(
        self, info: dict[str, Any], next_observation: np.ndarray
    ) -> float:
        return float(self.from_transition_info(info, next_observation))


def _batch(observations: np.ndarray) -> np.ndarray:
    return observations[None, :] if observations.ndim == 1 else observations


def _pendulum_cost(observations: np.ndarray) -> np.ndarray:
    obs = _batch(observations)
    return (np.abs(obs[:, 2]) > 6.0).astype(np.float32)


def _cartpole_cost(observations: np.ndarray) -> np.ndarray:
    obs = _batch(observations)
    return (np.abs(obs[:, 0]) > 2.4).astype(np.float32)


def _underwater_cost(observations: np.ndarray) -> np.ndarray:
    obs = _batch(observations)
    # Match UnderwaterDroneEnv._is_in_high_cost_area exactly.  The historical
    # environment uses x rather than x**2 in its first term; changing it here
    # would silently redefine the published task.
    value = obs[:, 0] / 0.9**2 + (obs[:, 1] - 2.0) ** 2 / 0.6**2
    return (value <= 1.0).astype(np.float32)


def _robot_cost(observations: np.ndarray) -> np.ndarray:
    obs = _batch(observations)
    robot = obs[:, 0:2]
    obstacle_data = obs[:, 6:]
    if obstacle_data.shape[1] < 3:
        return np.zeros(len(obs), dtype=np.float32)
    obstacles = obstacle_data.reshape(len(obs), -1, 3)
    distances = np.linalg.norm(robot[:, None, :] - obstacles[:, :, 0:2], axis=-1)
    radii = obstacles[:, :, 2]
    return np.any((radii > 0.0) & (distances <= radii), axis=1).astype(np.float32)


def _info_or_observation(key: str, fn: ArrayCost) -> InfoCost:
    def value(info: dict[str, Any], next_observation: np.ndarray) -> float:
        if key in info:
            return float(bool(info[key]))
        return float(fn(np.asarray(next_observation))[0])

    return value


_DEFINITIONS = {
    "Pendulum-v1": CostDefinition(
        env_id="Pendulum-v1",
        name="angular-rate-limit",
        description="Indicator that |angular velocity| exceeds 6 rad/s.",
        from_observation=_pendulum_cost,
        from_transition_info=_info_or_observation(
            "angular_rate_violation", _pendulum_cost
        ),
    ),
    "CartpoleSwingupEnvLong-v0": CostDefinition(
        env_id="CartpoleSwingupEnvLong-v0",
        name="cart-position-limit",
        description="Indicator that the cart leaves the |x| <= 2.4 operating region.",
        from_observation=_cartpole_cost,
        from_transition_info=_info_or_observation(
            "position_constraint", _cartpole_cost
        ),
    ),
    "UnderwaterDrone-v0": CostDefinition(
        env_id="UnderwaterDrone-v0",
        name="high-cost-region-intrusion",
        description="Indicator of occupancy of the environment's native high-cost region.",
        from_observation=_underwater_cost,
        from_transition_info=_info_or_observation(
            "is_in_high_cost_area", _underwater_cost
        ),
    ),
    "RobotNavigationConstSpeedCatch-v0": CostDefinition(
        env_id="RobotNavigationConstSpeedCatch-v0",
        name="obstacle-contact",
        description="Indicator of contact with an active circular obstacle/target.",
        from_observation=_robot_cost,
        from_transition_info=_info_or_observation("in_obstacle", _robot_cost),
    ),
}


def cost_definition(env_id: str) -> CostDefinition:
    try:
        return _DEFINITIONS[env_id]
    except KeyError as error:
        raise ValueError(f"No SOOPER cost definition for {env_id}") from error
