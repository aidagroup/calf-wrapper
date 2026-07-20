"""SOOPER adaptation for CALF-Wrapper experiments.

The implementation preserves the operational components of Wendl et al. (2026):
online discounted-cost tracking, pessimistic switching to a conservative policy
prior, probabilistic ensemble dynamics, uncertainty bonuses, and planning-MDP
termination with the prior's terminal value.
"""

from .costs import CostDefinition, cost_definition
from .filter import SOOPERSafetyFilter
from .models import (
    Actor,
    PriorValueEnsemble,
    ProbabilisticEnsemble,
    ReplayBuffer,
    TD3Planner,
)

__all__ = [
    "Actor",
    "CostDefinition",
    "ProbabilisticEnsemble",
    "PriorValueEnsemble",
    "ReplayBuffer",
    "SOOPERSafetyFilter",
    "TD3Planner",
    "cost_definition",
]
