"""Horizon-aware operating modes for CALF-Wrapper evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CalfMode = Literal[
    "custom",
    "conservative",
    "moderate",
    "high",
    "almost_open",
]


TARGET_ACCEPTANCE_BUDGETS: dict[str, float] = {
    "conservative": 0.10,
    "moderate": 0.35,
    "high": 0.70,
    "almost_open": 0.95,
}


@dataclass(frozen=True)
class ResolvedCalfMode:
    """Concrete gate parameters associated with a semantic operating mode."""

    mode: str
    horizon: int
    target_acceptance_budget: float
    relaxprob_init: float
    relaxprob_factor: float

    @property
    def acceptance_budget_lower_bound(self) -> float:
        return normalized_acceptance_budget(
            self.relaxprob_init,
            self.relaxprob_factor,
            self.horizon,
        )


def normalized_acceptance_budget(
    relaxprob_init: float,
    relaxprob_factor: float,
    horizon: int,
) -> float:
    """Return the mean probabilistic base-policy acceptance probability.

    This is the computable lower bound called NAB in the AUV CALF study:
    ``mean_t p0 * lambda**t`` for ``t = 0, ..., horizon - 1``.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= relaxprob_init <= 1.0:
        raise ValueError("relaxprob_init must be in [0, 1]")
    if not 0.0 <= relaxprob_factor <= 1.0:
        raise ValueError("relaxprob_factor must be in [0, 1]")
    powers = np.power(relaxprob_factor, np.arange(horizon, dtype=np.float64))
    return float(relaxprob_init * np.mean(powers))


def solve_relaxprob_factor(
    target_acceptance_budget: float,
    horizon: int,
    *,
    relaxprob_init: float = 1.0,
    tolerance: float = 1e-12,
) -> float:
    """Solve for lambda so a finite episode has the requested NAB lower bound."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < relaxprob_init <= 1.0:
        raise ValueError("relaxprob_init must be in (0, 1]")
    if not 0.0 < target_acceptance_budget <= relaxprob_init:
        raise ValueError("target_acceptance_budget must be in (0, relaxprob_init]")

    minimum = relaxprob_init / horizon
    if target_acceptance_budget < minimum - tolerance:
        raise ValueError(
            "target acceptance budget is unreachable with the requested p0 and horizon"
        )
    if abs(target_acceptance_budget - relaxprob_init) <= tolerance:
        return 1.0
    if abs(target_acceptance_budget - minimum) <= tolerance:
        return 0.0

    lower, upper = 0.0, 1.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        actual = normalized_acceptance_budget(relaxprob_init, midpoint, horizon)
        if abs(actual - target_acceptance_budget) <= tolerance:
            return midpoint
        if actual < target_acceptance_budget:
            lower = midpoint
        else:
            upper = midpoint
        if upper == lower:
            break
    return (lower + upper) / 2.0


def resolve_calf_mode(mode: CalfMode, horizon: int) -> ResolvedCalfMode:
    """Resolve a named AUV-style mode into horizon-specific ``p0`` and ``lambda``."""

    if mode == "custom":
        raise ValueError("custom mode must be supplied as explicit raw parameters")
    target = TARGET_ACCEPTANCE_BUDGETS[mode]
    relaxprob_init = 1.0
    return ResolvedCalfMode(
        mode=mode,
        horizon=horizon,
        target_acceptance_budget=target,
        relaxprob_init=relaxprob_init,
        relaxprob_factor=solve_relaxprob_factor(
            target,
            horizon,
            relaxprob_init=relaxprob_init,
        ),
    )
