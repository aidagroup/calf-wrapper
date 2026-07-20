"""Evaluate bare, fallback, and CALF controls on explicit held-out seeds.

The evaluator deliberately shares the environment, reward, cost, horizon,
goal-reaching predicate, and CMDP budget used by ``eval_sooper.py``.  This
provides paired raw trials for the final four-way comparison without relying
on differently seeded historical summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import torch

import src  # noqa: F401 - register project environments
from run.eval import goal_reaching_mask
from run.train_sooper import configure_determinism, controller_for, load_base_policy
from src.models.cleanrl_td3 import CleanRLTD3
from src.sooper import cost_definition
from src.utils.mlflow import reproducibility_tags
from src.utils.verified_artifacts import log_verified_artifact_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def policy_value(base, observation: np.ndarray) -> float:
    tensor = torch.as_tensor(
        observation[None], dtype=torch.float32, device=base.device
    )
    if isinstance(base, CleanRLTD3):
        action = base.actor(tensor)
        q1, q2 = base.critic(tensor, action)
        return float(torch.minimum(q1, q2).item())
    return float(base.policy.predict_values(tensor).item())


def evaluate_controls(
    *,
    environment: str,
    env_id: str,
    algorithm: str,
    model_path: Path,
    device: str,
    method: str,
    seeds: list[int],
    horizon: int,
    gamma: float,
    cost_budget: float,
    relaxprob_init: float,
    relaxprob_factor: float,
    calf_change_rate: float,
) -> list[dict]:
    base = load_base_policy(algorithm, str(model_path), device, seeds[0])
    prior = controller_for(environment)
    definition = cost_definition(env_id)
    rows = []
    for trial, seed in enumerate(seeds):
        env = gym.make(env_id)
        observation, _ = env.reset(seed=seed)
        observation = np.asarray(observation, dtype=np.float32)
        rng = np.random.default_rng(seed)
        best_value = policy_value(base, observation)
        total_reward = 0.0
        discounted_cost = 0.0
        reached = False
        interventions = 0
        deterministic_accepts = 0
        probabilistic_accepts = 0
        for step in range(horizon):
            base_action = np.asarray(
                base.predict(observation, deterministic=True)[0], dtype=np.float32
            ).reshape(-1)
            if method == "base":
                action = base_action
            elif method == "fallback":
                action = np.asarray(prior.get_action(observation), dtype=np.float32)
                interventions += 1
            elif method == "calf":
                value = policy_value(base, observation)
                deterministic = value - best_value - calf_change_rate >= 0.0
                if deterministic:
                    best_value = value
                    deterministic_accepts += 1
                probabilistic = bool(
                    rng.random() < relaxprob_init * relaxprob_factor**step
                )
                probabilistic_accepts += int(probabilistic)
                accepted = deterministic or probabilistic
                action = (
                    base_action
                    if accepted
                    else np.asarray(prior.get_action(observation), dtype=np.float32)
                )
                interventions += int(not accepted)
            else:
                raise ValueError(method)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            cost = definition.transition_cost(info, next_observation)
            total_reward += float(reward)
            discounted_cost += gamma**step * cost
            reached |= bool(goal_reaching_mask(env_id, next_observation[None])[0])
            observation = next_observation
            if terminated or truncated:
                break
        env.close()
        length = step + 1
        rows.append(
            {
                "method": method,
                "trial": trial,
                "evaluation_seed": seed,
                "episode_return": total_reward,
                "discounted_cost": discounted_cost,
                "constraint_satisfied": discounted_cost < cost_budget,
                "goal_reached": reached,
                "interventions": interventions,
                "intervention_fraction": interventions / length,
                "empirical_nab": 1.0 - interventions / length,
                "deterministic_acceptance_fraction": deterministic_accepts / length,
                "probabilistic_acceptance_fraction": probabilistic_accepts / length,
                "episode_length": length,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=[
        "pendulum", "cartpole", "underwater-drone", "robot-navigation"
    ])
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algorithm", choices=["ppo", "cleanrl_td3"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--method", choices=["base", "fallback", "calf"], required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated held-out seeds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost-budget", type=float, required=True)
    parser.add_argument("--relaxprob-init", type=float, default=1.0)
    parser.add_argument("--relaxprob-factor", type=float, default=1.0)
    parser.add_argument("--calf-change-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    configure_determinism(seeds[0])
    rows = evaluate_controls(
        environment=args.environment,
        env_id=args.env_id,
        algorithm=args.algorithm,
        model_path=args.model_path,
        device=args.device,
        method=args.method,
        seeds=seeds,
        horizon=args.horizon,
        gamma=args.gamma,
        cost_budget=args.cost_budget,
        relaxprob_init=args.relaxprob_init,
        relaxprob_factor=args.relaxprob_factor,
        calf_change_rate=args.calf_change_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = args.output_dir / "held_out_trials.csv"
    with trial_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    returns = np.asarray([row["episode_return"] for row in rows])
    summary = {
        "format": "calf-wrapper-comparison-control-held-out-v1",
        "method": args.method,
        "environment": args.environment,
        "env_id": args.env_id,
        "algorithm": args.algorithm,
        "model_path": str(args.model_path),
        "model_sha256": sha256(args.model_path),
        "held_out_seeds": seeds,
        "cost_budget": args.cost_budget,
        "calf": {
            "relaxprob_init": args.relaxprob_init,
            "relaxprob_factor": args.relaxprob_factor,
            "change_rate": args.calf_change_rate,
        },
        "metrics": {
            "mean_reward": float(returns.mean()),
            "std_reward": float(returns.std()),
            "reward_ci95_half_width": float(
                1.96 * returns.std() / np.sqrt(len(returns))
            ),
            "goal_reaching_rate": float(np.mean([row["goal_reached"] for row in rows])),
            "constraint_satisfaction_rate": float(
                np.mean([row["constraint_satisfied"] for row in rows])
            ),
            "intervention_fraction": float(
                np.mean([row["intervention_fraction"] for row in rows])
            ),
            "empirical_nab": float(np.mean([row["empirical_nab"] for row in rows])),
            "n_trials": len(rows),
        },
    }
    summary_path = args.output_dir / "summary.json"
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(reproducibility_tags())
        mlflow.set_tags({"method": args.method, "protocol": "held-out-confirmation"})
        mlflow.log_params(
            {
                "environment": args.environment,
                "env_id": args.env_id,
                "algorithm": args.algorithm,
                "model_sha256": summary["model_sha256"],
                "held_out_seeds": args.seeds,
                "cost_budget": args.cost_budget,
                "relaxprob_init": args.relaxprob_init,
                "relaxprob_factor": args.relaxprob_factor,
                "calf_change_rate": args.calf_change_rate,
            }
        )
        mlflow.log_metrics(summary["metrics"])
        summary["mlflow_run_id"] = run.info.run_id
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="comparison_control_") as temp:
            staging = Path(temp)
            (staging / "summary.json").write_bytes(summary_path.read_bytes())
            (staging / "held_out_trials.csv").write_bytes(trial_path.read_bytes())
            log_verified_artifact_batch(staging)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
