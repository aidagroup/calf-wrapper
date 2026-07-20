"""Calibrate one fixed CMDP budget from conservative-prior development trials."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np

import src  # noqa: F401
from run.eval import goal_reaching_mask
from run.train_sooper import configure_determinism, controller_for
from src.sooper import cost_definition
from src.utils.mlflow import reproducibility_tags
from src.utils.verified_artifacts import log_verified_artifact_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=["underwater-drone"])
    parser.add_argument("--env-id", default="UnderwaterDrone-v0")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop-inclusive", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--margin", type=float, default=1.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    configure_determinism(args.seed_start)
    prior = controller_for(args.environment)
    definition = cost_definition(args.env_id)
    rows = []
    for trial, seed in enumerate(range(args.seed_start, args.seed_stop_inclusive + 1)):
        env = gym.make(args.env_id)
        observation, _ = env.reset(seed=seed)
        discounted_cost = 0.0
        total_reward = 0.0
        reached = False
        for step in range(args.horizon):
            action = np.asarray(prior.get_action(observation), dtype=np.float32)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            discounted_cost += args.gamma**step * definition.transition_cost(
                info, next_observation
            )
            total_reward += float(reward)
            reached |= bool(goal_reaching_mask(args.env_id, next_observation[None])[0])
            observation = next_observation
            if terminated or truncated:
                break
        env.close()
        rows.append(
            {
                "trial": trial,
                "seed": seed,
                "episode_return": total_reward,
                "discounted_cost": discounted_cost,
                "goal_reached": reached,
                "episode_length": step + 1,
            }
        )
    costs = np.asarray([row["discounted_cost"] for row in rows])
    budget = float(costs.max() * args.margin + 1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "prior_calibration_trials.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "calf-wrapper-sooper-budget-calibration-v1",
        "environment": args.environment,
        "env_id": args.env_id,
        "cost_definition": definition.name,
        "seed_start": args.seed_start,
        "seed_stop_inclusive": args.seed_stop_inclusive,
        "n_trials": len(rows),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "margin": args.margin,
        "selection_rule": "margin_times_maximum_discounted_prior_cost",
        "maximum_discounted_prior_cost": float(costs.max()),
        "mean_discounted_prior_cost": float(costs.mean()),
        "std_discounted_prior_cost": float(costs.std()),
        "prior_goal_reaching_rate": float(np.mean([row["goal_reached"] for row in rows])),
        "cost_budget": budget,
    }
    summary_path = args.output_dir / "summary.json"
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(reproducibility_tags())
        mlflow.set_tags({"method": "fallback", "protocol": "budget-calibration"})
        mlflow.log_params(
            {
                "environment": args.environment,
                "env_id": args.env_id,
                "seed_start": args.seed_start,
                "seed_stop_inclusive": args.seed_stop_inclusive,
                "horizon": args.horizon,
                "gamma": args.gamma,
                "margin": args.margin,
                "selection_rule": summary["selection_rule"],
            }
        )
        mlflow.log_metrics(
            {
                "cost_budget": budget,
                "maximum_discounted_prior_cost": costs.max(),
                "mean_discounted_prior_cost": costs.mean(),
                "prior_goal_reaching_rate": summary["prior_goal_reaching_rate"],
            }
        )
        summary["mlflow_run_id"] = run.info.run_id
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="sooper_budget_") as temp:
            staging = Path(temp)
            (staging / "summary.json").write_bytes(summary_path.read_bytes())
            (staging / raw_path.name).write_bytes(raw_path.read_bytes())
            log_verified_artifact_batch(staging)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
