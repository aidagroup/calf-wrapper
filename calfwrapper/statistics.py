"""Paired return comparisons reported in the article."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from scipy.stats import wilcoxon

from calfwrapper.operating_modes import OPERATING_MODES

STAGES = ("Low", "Mid", "High")


class ReturnTest(TypedDict):
    family: str
    environment: str
    stage: str
    mode: str
    comparison: str
    test: str
    mean_paired_difference: float
    p_value: float
    holm_adjusted_p_value: float
    significant: bool


def _returns(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        return np.asarray(
            [float(row["episode_return"]) for row in csv.DictReader(stream)],
            dtype=float,
        )


def _p_value(differences: np.ndarray, alternative: str) -> float:
    if np.all(differences == 0):
        return 1.0
    return float(wilcoxon(differences, alternative=alternative, method="auto").pvalue)


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def paired_return_tests(
    trials_directory: Path,
    environments: tuple[str, ...],
) -> list[ReturnTest]:
    """Compute the tests summarized in Tables 7, 8, 10, and 11."""

    incomplete_rows: list[dict[str, object]] = []
    for stage in STAGES:
        for environment in environments:
            fallback = _returns(trials_directory / f"{environment}-all-fallback.csv")
            base = _returns(trials_directory / f"{environment}-{stage.lower()}-base.csv")
            for mode in OPERATING_MODES:
                wrapped = _returns(trials_directory / f"{environment}-{stage.lower()}-{mode}.csv")
                for reference_name, reference in (("fallback", fallback), ("base", base)):
                    differences = wrapped - reference
                    incomplete_rows.append(
                        {
                            "family": f"calfwrapper_vs_{reference_name}_{stage.lower()}",
                            "environment": environment,
                            "stage": stage,
                            "mode": mode,
                            "comparison": reference_name,
                            "test": "two-sided paired Wilcoxon signed-rank",
                            "mean_paired_difference": float(differences.mean()),
                            "p_value": _p_value(differences, "two-sided"),
                        }
                    )

                base_differences = wrapped - base
                fallback_differences = wrapped - fallback
                incomplete_rows.append(
                    {
                        "family": f"calfwrapper_above_both_{stage.lower()}",
                        "environment": environment,
                        "stage": stage,
                        "mode": mode,
                        "comparison": "base and fallback",
                        "test": "intersection-union of two one-sided paired Wilcoxon tests",
                        "mean_paired_difference": min(
                            float(base_differences.mean()),
                            float(fallback_differences.mean()),
                        ),
                        "p_value": max(
                            _p_value(base_differences, "greater"),
                            _p_value(fallback_differences, "greater"),
                        ),
                    }
                )

    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(incomplete_rows):
        families[str(row["family"])].append(index)
    for indices in families.values():
        adjusted = _holm([cast(float, incomplete_rows[index]["p_value"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            incomplete_rows[index]["holm_adjusted_p_value"] = value
            incomplete_rows[index]["significant"] = value < 0.05
    return cast(list[ReturnTest], incomplete_rows)


def write_tests(rows: list[ReturnTest], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
