"""Summary data used by the article figures and tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

from calfwrapper.operating_modes import OPERATING_MODES

STAGES = ("Low", "Mid", "High")
METHODS = ("fallback", *OPERATING_MODES, "base", "lagrangian")


class Result(TypedDict):
    environment: str
    checkpoint: str
    method: str
    trials: int
    mean_episode_return: float
    return_ci95_half_width: float
    goal_reaching_rate: float
    fallback_action_percentage: float


def _read_trials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(
    trials_directory: Path,
    environments: tuple[str, ...],
) -> list[Result]:
    """Calculate the values displayed in the central results table."""

    rows: list[Result] = []
    for environment in environments:
        for stage in STAGES:
            for method in METHODS:
                checkpoint = "all" if method == "fallback" else stage.lower()
                trials = _read_trials(trials_directory / f"{environment}-{checkpoint}-{method}.csv")
                returns = [float(trial["episode_return"]) for trial in trials]
                mean_return = sum(returns) / len(returns)
                variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
                successes = [trial["goal_reached"].lower() == "true" for trial in trials]
                rows.append(
                    {
                        "environment": environment,
                        "checkpoint": stage,
                        "method": method,
                        "trials": len(trials),
                        "mean_episode_return": mean_return,
                        "return_ci95_half_width": (
                            1.96 * math.sqrt(variance) / math.sqrt(len(returns))
                        ),
                        "goal_reaching_rate": 100 * sum(successes) / len(successes),
                        "fallback_action_percentage": (
                            sum(
                                100
                                * int(trial["fallback_policy_actions"])
                                / int(trial["episode_length"])
                                for trial in trials
                            )
                            / len(trials)
                            if method != "lagrangian"
                            else 0.0
                        ),
                    }
                )
    return rows


def write_summary(rows: list[Result], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
