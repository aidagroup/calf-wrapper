"""CALF-Wrapper operating modes used in the article."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OperatingMode = Literal[
    "conservative",
    "guarded",
    "moderate",
    "balanced",
    "high",
    "almost_open",
]

OPERATING_MODES: tuple[OperatingMode, ...] = (
    "conservative",
    "guarded",
    "moderate",
    "balanced",
    "high",
    "almost_open",
)

_ACCEPTANCE_BUDGETS: dict[OperatingMode, float] = {
    "conservative": 0.00,
    "guarded": 0.10,
    "moderate": 0.35,
    "balanced": 0.50,
    "high": 0.70,
    "almost_open": 0.95,
}


@dataclass(frozen=True)
class OperatingModeParameters:
    p_relax: float
    lambda_: float


def _acceptance_budget(p_relax: float, lambda_: float, horizon: int) -> float:
    powers = np.power(lambda_, np.arange(horizon, dtype=np.float64))
    return float(p_relax * np.mean(powers))


def _solve_lambda(acceptance_budget: float, horizon: int) -> float:
    lower, upper = 0.0, 1.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        actual = _acceptance_budget(1.0, midpoint, horizon)
        if abs(actual - acceptance_budget) <= 1e-12:
            return midpoint
        if actual < acceptance_budget:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def operating_mode_parameters(
    operating_mode: OperatingMode,
    horizon: int,
) -> OperatingModeParameters:
    """Return the values of p_relax and lambda for an operating mode."""

    acceptance_budget = _ACCEPTANCE_BUDGETS[operating_mode]
    if acceptance_budget == 0.0:
        return OperatingModeParameters(p_relax=0.0, lambda_=0.0)
    return OperatingModeParameters(
        p_relax=1.0,
        lambda_=_solve_lambda(acceptance_budget, horizon),
    )


def fixed_acceptance_budget(
    acceptance_budget: float,
    p_relax: float,
    horizon: int,
) -> OperatingModeParameters:
    """Keep the acceptance budget fixed while changing p_relax."""

    if not 0.0 < acceptance_budget < p_relax <= 1.0:
        raise ValueError("acceptance_budget must be smaller than p_relax")
    return OperatingModeParameters(
        p_relax=p_relax,
        lambda_=_solve_lambda(acceptance_budget / p_relax, horizon),
    )
