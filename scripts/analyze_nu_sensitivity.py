#!/usr/bin/env python3
"""Select a frozen nu calibration rule from a completed development sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def candidate_scores(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["environment", "training_seed", "checkpoint_step"]
    candidates = ["nu_calibration_rule", "nu_calibration_n"]
    expected = results[candidates].drop_duplicates().shape[0]
    counts = results.groupby(keys).size()
    if not (counts == expected).all():
        raise ValueError("each checkpoint must contain every candidate exactly once")

    reward_min = results.groupby(keys)["mean_reward"].transform("min")
    reward_span = results.groupby(keys)["mean_reward"].transform("max") - reward_min
    results = results.copy()
    results["checkpoint_reward_score"] = np.where(
        reward_span > 0,
        (results["mean_reward"] - reward_min) / reward_span,
        0.5,
    )
    by_environment = results.groupby([*candidates, "environment"], as_index=False).agg(
        environment_reward_score=("checkpoint_reward_score", "mean"),
        environment_goal_rate=("goal_reaching_rate", "mean"),
        checkpoints=("task_id", "count"),
    )
    scores = (
        by_environment.groupby(candidates, as_index=False)
        .agg(
            reward_score=("environment_reward_score", "mean"),
            minimum_environment_goal_rate=("environment_goal_rate", "min"),
            mean_environment_goal_rate=("environment_goal_rate", "mean"),
            environments=("environment", "count"),
            checkpoints=("checkpoints", "sum"),
        )
        .sort_values(candidates)
    )
    return scores


def select_candidate(
    scores: pd.DataFrame,
    *,
    minimum_goal_rate: float,
    practical_tie: float,
) -> tuple[pd.Series, str]:
    feasible = scores[scores["minimum_environment_goal_rate"] >= minimum_goal_rate]
    if len(feasible):
        pool = feasible
        basis = f"minimum environment goal rate >= {minimum_goal_rate:g}%"
    else:
        best_goal = scores["minimum_environment_goal_rate"].max()
        pool = scores[scores["minimum_environment_goal_rate"] == best_goal]
        basis = "no candidate met the goal-rate constraint; maximized minimum goal rate"

    best_reward = pool["reward_score"].max()
    plateau = pool[pool["reward_score"] >= best_reward - practical_tie].copy()
    log_center = np.mean(np.log(scores["nu_calibration_n"].unique()))
    plateau["center_distance"] = np.abs(
        np.log(plateau["nu_calibration_n"]) - log_center
    )
    plateau["conservative_rule_priority"] = (
        plateau["nu_calibration_rule"] != "goal_guarded_max"
    ).astype(int)
    selected = plateau.sort_values(
        [
            "center_distance",
            "conservative_rule_priority",
            "nu_calibration_n",
        ]
    ).iloc[0]
    return selected, basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-goal-rate", type=float, default=95.0)
    parser.add_argument("--practical-tie", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = pd.read_csv(args.results)
    scores = candidate_scores(results)
    selected, basis = select_candidate(
        scores,
        minimum_goal_rate=args.minimum_goal_rate,
        practical_tie=args.practical_tie,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "nu_candidate_scores.csv", index=False)
    decision = {
        "selected_rule": selected["nu_calibration_rule"],
        "selected_n": float(selected["nu_calibration_n"]),
        "reward_score": float(selected["reward_score"]),
        "minimum_environment_goal_rate": float(
            selected["minimum_environment_goal_rate"]
        ),
        "mean_environment_goal_rate": float(selected["mean_environment_goal_rate"]),
        "feasibility_basis": basis,
        "minimum_goal_rate": args.minimum_goal_rate,
        "practical_tie": args.practical_tie,
        "selection_status": "frozen",
    }
    (args.output_dir / "nu_selection.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
