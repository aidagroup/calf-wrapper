"""Evaluate a frozen SOOPER checkpoint on explicitly held-out seeds."""

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

import src  # noqa: F401
from run.train_sooper import (
    SOOPERConfig,
    configure_determinism,
    controller_for,
    evaluate,
)
from src.sooper import PriorValueEnsemble, ProbabilisticEnsemble, TD3Planner
from src.utils.mlflow import reproducibility_tags
from src.utils.verified_artifacts import log_verified_artifact_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_components(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != "calf-wrapper-sooper-v1":
        raise ValueError("Unsupported SOOPER checkpoint")
    config = SOOPERConfig(**payload["config"])
    config.device = device
    env = gym.make(config.env_id)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    model = ProbabilisticEnsemble(
        observation_dim,
        action_dim,
        ensemble_size=config.ensemble_size,
        device=device,
    )
    model.load_state_dict(payload["world_model"])
    model.is_fitted = bool(payload["world_model_is_fitted"])
    prior_value_model = PriorValueEnsemble(
        observation_dim,
        action_dim,
        ensemble_size=config.ensemble_size,
        device=device,
    )
    prior_value_model.load_state_dict(payload["prior_value_model"])
    prior_value_model.is_fitted = bool(payload["prior_value_model_is_fitted"])
    planner = TD3Planner.create(
        observation_dim,
        env.action_space.low,
        env.action_space.high,
        device,
        config.actor_learning_rate,
    )
    planner.load_state_dict(payload["planner"])
    env.close()
    return (
        config,
        model,
        prior_value_model,
        planner,
        float(payload["budget"]),
        int(payload["iteration"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated held-out seeds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    config, model, prior_value_model, planner, budget, iteration = load_components(
        args.checkpoint, args.device
    )
    configure_determinism(config.seed)
    trials = evaluate(
        planner,
        model,
        prior_value_model,
        controller_for(config.environment),
        config,
        budget,
        iteration=iteration,
        seeds=seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = args.output_dir / "held_out_trials.csv"
    with trial_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    returns = np.asarray([row["episode_return"] for row in trials])
    summary = {
        "format": "calf-wrapper-sooper-held-out-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "environment": config.environment,
        "env_id": config.env_id,
        "training_seed": config.seed,
        "held_out_seeds": seeds,
        "cost_budget": budget,
        "metrics": {
            "mean_reward": float(returns.mean()),
            "std_reward": float(returns.std()),
            "reward_ci95_half_width": float(
                1.96 * returns.std() / np.sqrt(len(returns))
            ),
            "goal_reaching_rate": float(
                np.mean([row["goal_reached"] for row in trials])
            ),
            "constraint_satisfaction_rate": float(
                np.mean([row["constraint_satisfied"] for row in trials])
            ),
            "intervention_fraction": float(
                np.mean([row["intervention_fraction"] for row in trials])
            ),
            "n_trials": len(trials),
        },
    }
    summary_path = args.output_dir / "summary.json"
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(reproducibility_tags())
        mlflow.set_tags(
            {
                "method": "SOOPER",
                "protocol": "held-out-confirmation",
                "source_checkpoint_sha256": summary["checkpoint_sha256"],
            }
        )
        mlflow.log_params(
            {
                "environment": config.environment,
                "env_id": config.env_id,
                "training_seed": config.seed,
                "held_out_seeds": args.seeds,
                "cost_budget": budget,
            }
        )
        mlflow.log_metrics(summary["metrics"])
        summary["mlflow_run_id"] = run.info.run_id
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="sooper_heldout_") as temp:
            staging = Path(temp)
            (staging / "summary.json").write_bytes(summary_path.read_bytes())
            (staging / "held_out_trials.csv").write_bytes(trial_path.read_bytes())
            log_verified_artifact_batch(staging)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
