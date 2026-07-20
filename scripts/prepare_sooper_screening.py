"""Freeze a SOOPER development matrix from the completed CALF sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def as_rate(value: str) -> float:
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


def load_groups(path: Path):
    groups = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["training_seed"] or not row["checkpoint_step"]:
                continue
            key = (
                row["algorithm"],
                row["environment"],
                int(float(row["training_seed"])),
                int(float(row["checkpoint_step"])),
            )
            groups[key][row["mode"]] = row
    return groups


def shortlist(groups, protocol):
    spec = protocol["checkpoint_shortlist"]
    required = set(spec["required_checkpoint_methods"])
    candidates = []
    for key, modes in groups.items():
        algorithm, environment, training_seed, checkpoint_step = key
        if environment != protocol["primary_environment"] or not required <= set(modes):
            continue
        base = modes["base"]
        base_goal = as_rate(base["goal_reaching_rate"])
        if base_goal > spec["base_goal_reaching_rate_max"]:
            continue
        feasible = []
        for mode in required - {"base"}:
            row = modes[mode]
            goal = as_rate(row["goal_reaching_rate"])
            gain = float(row["mean_reward"]) - float(base["mean_reward"])
            if goal >= spec["calf_goal_reaching_rate_min"] and gain > 0.0:
                feasible.append((gain, mode, row))
        if not feasible:
            continue
        gain, mode, row = max(feasible)
        candidates.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "training_seed": training_seed,
                "checkpoint_step": checkpoint_step,
                "base_reward": float(base["mean_reward"]),
                "base_goal_reaching_rate": base_goal,
                "selected_calf_mode": mode,
                "calf_reward": float(row["mean_reward"]),
                "calf_goal_reaching_rate": as_rate(row["goal_reaching_rate"]),
                "calf_reward_gain": gain,
                "development_task_ids": {
                    name: modes[name]["task_id"] for name in sorted(required)
                },
                "development_mlflow_run_ids": {
                    name: modes[name]["mlflow_run_id"] for name in sorted(required)
                },
            }
        )
    candidates.sort(key=lambda row: row["calf_reward_gain"], reverse=True)
    return candidates[: spec["retain"]]


def task(candidate, seed, iterations, cost_budget, tracking_uri, experiment_name):
    environment = candidate["environment"]
    training_seed = candidate["training_seed"]
    checkpoint_step = candidate["checkpoint_step"]
    return {
        "environment": "underwater-drone",
        "algorithm": candidate["algorithm"],
        "model_path": (
            f"run/artifacts/td3_{environment}_{training_seed}/checkpoints/"
            f"td3_checkpoint_{checkpoint_step}_steps.pt"
        ),
        "seed": seed,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "horizon": 1500,
        "offline_prior_episodes": 5,
        "online_iterations": iterations,
        "real_episodes_per_iteration": 1,
        "evaluation_episodes": 10,
        "ensemble_size": 5,
        "model_epochs": 5,
        "model_batch_size": 256,
        "prior_value_epochs": 30,
        "prior_value_learning_rate": 0.0003,
        "actor_learning_rate": 0.0003,
        "replay_capacity": 1000000,
        "model_rollout_batch": 256,
        "model_rollout_horizon": 5,
        "policy_updates": 500,
        "batch_size": 256,
        "gamma": 0.99,
        "cost_budget": cost_budget,
        "pessimism_beta": 2.0,
        "prior_horizon": 50,
        "lambda_explore": 0.1,
        "lambda_expand": 0.1,
        "exploration_noise": 0.1,
        "checkpoint_every": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calf-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--budget-calibration", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--shortlist-output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    calibration = json.loads(args.budget_calibration.read_text())
    if calibration.get("format") != "calf-wrapper-sooper-budget-calibration-v1":
        raise SystemExit("Unsupported budget calibration file")
    cost_budget = float(calibration["cost_budget"])
    candidates = shortlist(load_groups(args.calf_results), protocol)
    if len(candidates) != protocol["checkpoint_shortlist"]["retain"]:
        raise SystemExit(
            f"Expected {protocol['checkpoint_shortlist']['retain']} candidates; "
            f"found {len(candidates)}. Is the CALF matrix complete?"
        )
    screening = protocol["sooper_screening"]
    tasks = [
        task(
            candidate,
            seed,
            iterations,
            cost_budget,
            args.tracking_uri,
            args.experiment_name,
        )
        for candidate in candidates
        for iterations in screening["online_iterations"]
        for seed in screening["training_seeds"]
    ]
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.write_text(
        json.dumps(
            {
                "format": "calf-wrapper-sooper-development-matrix-v1",
                "protocol": str(args.protocol),
                "source": str(args.calf_results),
                "budget_calibration": str(args.budget_calibration),
                "cost_budget": cost_budget,
                "tasks": tasks,
            },
            indent=2,
        )
        + "\n"
    )
    args.shortlist_output.write_text(
        json.dumps(
            {
                "format": "calf-wrapper-sooper-checkpoint-shortlist-v1",
                "protocol": str(args.protocol),
                "source": str(args.calf_results),
                "candidates": candidates,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"candidates": len(candidates), "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
