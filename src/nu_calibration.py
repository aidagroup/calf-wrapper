"""Deterministic calibration rule for the CALF critic-improvement threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NuCalibration:
    """Components of the pre-evaluation threshold rule."""

    n: float
    horizon: int
    goal_local_variation: float
    fallback_value_range: float
    range_increment: float
    nu: float
    goal_neighborhood_transitions: int
    fallback_transitions: int


def calibrate_nu(
    current_values: np.ndarray,
    next_values: np.ndarray,
    current_goal_mask: np.ndarray,
    next_goal_mask: np.ndarray,
    *,
    horizon: int,
    n: float,
) -> NuCalibration:
    """Compute ``nu=max(delta_goal, value_range/(n*T))``.

    ``delta_goal`` is the maximum absolute one-step critic change on fallback
    transitions that enter the goal set.  The supplied transitions must be
    first-hitting trajectories truncated immediately after goal entry.  This
    forms a data-defined one-step neighborhood and avoids a coordinate-dependent
    radius or an unbounded goal-set interior.
    The maximum (rather than a quantile) makes the rule deterministic and
    deliberately conservative with respect to local critic oscillations.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n must be a positive finite number")

    current = np.asarray(current_values, dtype=np.float64).reshape(-1)
    following = np.asarray(next_values, dtype=np.float64).reshape(-1)
    current_goal = np.asarray(current_goal_mask, dtype=bool).reshape(-1)
    next_goal = np.asarray(next_goal_mask, dtype=bool).reshape(-1)
    size = len(current)
    if not (len(following) == len(current_goal) == len(next_goal) == size and size > 0):
        raise ValueError("all transition arrays must have the same non-zero length")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(following)):
        raise ValueError("critic values must be finite")

    values = np.concatenate([current, following])
    fallback_value_range = float(np.max(values) - np.min(values))
    range_increment = fallback_value_range / (float(n) * horizon)
    goal_neighborhood = ~current_goal & next_goal
    if np.any(goal_neighborhood):
        goal_local_variation = float(
            np.max(np.abs(following[goal_neighborhood] - current[goal_neighborhood]))
        )
    else:
        raise ValueError(
            "fallback calibration contains no transition entering the goal"
        )

    return NuCalibration(
        n=float(n),
        horizon=horizon,
        goal_local_variation=goal_local_variation,
        fallback_value_range=fallback_value_range,
        range_increment=range_increment,
        nu=max(goal_local_variation, range_increment),
        goal_neighborhood_transitions=int(np.sum(goal_neighborhood)),
        fallback_transitions=size,
    )
