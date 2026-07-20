"""Train the SOOPER baseline on CALF-Wrapper environments.

This portable PyTorch implementation follows Algorithms 1--2 of Wendl et al.
(ICLR 2026) while using the repository's Gymnasium environments and TD3-style
actor--critic.  See ``docs/sooper-baseline.md`` for the exact correspondence
and documented implementation-level deviations from the official JAX/Brax code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
import torch
from stable_baselines3 import PPO

import src  # noqa: F401 - register project environments
from run.eval import goal_reaching_mask
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from src.controllers.pendulum import EnergyBasedStabilizingPolicy
from src.controllers.robot_navigation import RobotNavigationConstSpeedGoalController
from src.controllers.underwaterdrone import UnderwaterDroneNominalController
from src.models.cleanrl_td3 import CleanRLTD3
from src.sooper import (
    PriorValueEnsemble,
    ProbabilisticEnsemble,
    ReplayBuffer,
    SOOPERSafetyFilter,
    TD3Planner,
    cost_definition,
)
from src.utils.mlflow import reproducibility_tags
from src.utils.verified_artifacts import log_verified_artifact_batch


ENVIRONMENTS = {
    "pendulum": ("Pendulum-v1", 200),
    "cartpole": ("CartpoleSwingupEnvLong-v0", 1000),
    "underwater-drone": ("UnderwaterDrone-v0", 1500),
    "robot-navigation": ("RobotNavigationConstSpeedCatch-v0", 1000),
}


@dataclass
class SOOPERConfig:
    environment: str
    env_id: str
    algorithm: str
    model_path: str
    seed: int
    device: str
    horizon: int
    offline_prior_episodes: int
    online_iterations: int
    real_episodes_per_iteration: int
    evaluation_episodes: int
    ensemble_size: int
    model_epochs: int
    model_batch_size: int
    model_learning_rate: float
    prior_value_epochs: int
    prior_value_learning_rate: float
    behavior_clone_epochs: int
    actor_learning_rate: float
    replay_capacity: int
    model_rollout_batch: int
    model_rollout_horizon: int
    policy_updates: int
    batch_size: int
    gamma: float
    cost_budget: float
    budget_margin: float
    pessimism_beta: float
    prior_horizon: int
    lambda_explore: float
    lambda_expand: float
    exploration_noise: float
    checkpoint_every: int
    output_dir: str
    tracking_uri: str
    experiment_name: str
    run_name: str


def controller_for(environment: str):
    if environment == "pendulum":
        return EnergyBasedStabilizingPolicy(
            gain=0.6,
            action_min=-2.0,
            action_max=2.0,
            switch_loc=np.cos(np.pi / 10),
            switch_vel_loc=0.2,
            pd_coeffs=[12, 4],
        )
    if environment == "cartpole":
        return CartpoleEnergyBasedStabilizingPolicy(
            pd_coefs=[70, 10.0, 20.0, 10.0],
            gain=0.3,
            gain_pos_vel=0.5,
            action_min=-10.0,
            action_max=10.0,
        )
    if environment == "underwater-drone":
        return UnderwaterDroneNominalController()
    if environment == "robot-navigation":
        return RobotNavigationConstSpeedGoalController()
    raise ValueError(environment)


def load_base_policy(algorithm: str, path: str, device: str, seed: int):
    if algorithm == "ppo":
        return PPO.load(path, device=device, seed=seed)
    if algorithm == "cleanrl_td3":
        return CleanRLTD3.load(path, device=device)
    raise ValueError(f"Unsupported base algorithm: {algorithm}")


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def restore_torch_rng_states(payload: dict[str, Any]) -> None:
    """Restore RNG tensors after ``map_location`` without device leakage."""
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    if payload["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])


def safe_reset(env: gym.Env, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    observation, info = env.reset(seed=seed)
    return np.asarray(observation, dtype=np.float32), info


def collect_prior_data(
    env_id: str,
    prior,
    base,
    replay: ReplayBuffer,
    *,
    episodes: int,
    horizon: int,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    definition = cost_definition(env_id)
    behavior_observations: list[np.ndarray] = []
    behavior_actions: list[np.ndarray] = []
    episode_costs: list[float] = []
    for episode in range(episodes):
        env = gym.make(env_id)
        observation, _ = safe_reset(env, seed + episode)
        discounted_cost = 0.0
        for step in range(horizon):
            action = np.asarray(prior.get_action(observation), dtype=np.float32)
            base_action = base.predict(observation, deterministic=True)[0]
            behavior_observations.append(observation.copy())
            behavior_actions.append(
                np.asarray(base_action, dtype=np.float32).reshape(-1)
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            cost = definition.transition_cost(info, next_observation)
            done = bool(terminated or truncated)
            replay.add(observation, action, next_observation, reward, cost, done)
            discounted_cost += gamma**step * cost
            observation = next_observation
            if done:
                break
        env.close()
        episode_costs.append(discounted_cost)
    return (
        np.asarray(behavior_observations, dtype=np.float32),
        np.asarray(behavior_actions, dtype=np.float32),
        episode_costs,
    )


def fit_world_model(model, prior_value_model, prior, replay, config, iteration):
    data = replay.arrays()
    metrics = model.fit(
        data["observations"],
        data["actions"],
        data["next_observations"],
        data["rewards"],
        data["costs"],
        epochs=config.model_epochs,
        batch_size=config.model_batch_size,
        learning_rate=config.model_learning_rate,
        seed=config.seed + 10_000 + iteration,
    )
    metrics.update(
        prior_value_model.fit(
            data,
            prior,
            gamma=config.gamma,
            epochs=config.prior_value_epochs,
            batch_size=config.model_batch_size,
            learning_rate=config.prior_value_learning_rate,
            seed=config.seed + 30_000 + iteration,
        )
    )
    return metrics


def model_rollouts(
    model,
    planner,
    safety_filter,
    real_replay,
    synthetic_replay,
    env,
    config,
    rng,
):
    batch = real_replay.sample(min(config.model_rollout_batch, real_replay.size))
    observations = batch["observations"].copy()
    active = np.ones(len(observations), dtype=bool)
    accumulated = np.zeros(len(observations), dtype=np.float32)
    action_low = env.action_space.low
    action_high = env.action_space.high
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    generated = 0
    mean_uncertainty = []
    for step in range(config.model_rollout_horizon):
        if not np.any(active):
            break
        actions = planner.action(observations)
        with torch.no_grad():
            obs_t = torch.as_tensor(observations, device=model.device)
            action_t = torch.as_tensor(actions, device=model.device)
            next_members, rewards_members, costs_members, _ = model.predict_members(
                obs_t, action_t
            )
            uncertainty = next_members.std(0, unbiased=False).norm(dim=-1).cpu().numpy()
            member_indices = rng.integers(
                0, model.ensemble_size, size=len(observations)
            )
            row = torch.arange(len(observations), device=model.device)
            member_tensor = torch.as_tensor(member_indices, device=model.device)
            next_observations = next_members[member_tensor, row].cpu().numpy()
            rewards = rewards_members[member_tensor, row].cpu().numpy()
            costs = costs_members[member_tensor, row].cpu().numpy()
        next_observations = np.maximum(np.minimum(next_observations, obs_high), obs_low)
        q_cost, prior_reward, _ = safety_filter.prior_values_batch(
            observations, actions
        )
        expected_cost = accumulated + config.gamma**step * q_cost
        unsafe = expected_cost >= safety_filter.budget
        bonus = (config.lambda_explore + config.lambda_expand) * uncertainty
        planning_reward = np.where(unsafe, prior_reward, rewards + bonus)
        done = unsafe | (step + 1 == config.model_rollout_horizon)
        for index in np.flatnonzero(active):
            synthetic_replay.add(
                observations[index],
                np.clip(actions[index], action_low, action_high),
                next_observations[index],
                planning_reward[index],
                costs[index],
                done[index],
            )
            generated += 1
        mean_uncertainty.extend(uncertainty[active].tolist())
        accumulated += config.gamma**step * costs
        observations = next_observations
        active &= ~done
    return {
        "model_transitions": float(generated),
        "model_uncertainty": (
            float(np.mean(mean_uncertainty)) if mean_uncertainty else 0.0
        ),
    }


def evaluate(
    planner,
    model,
    prior_value_model,
    prior,
    config,
    budget,
    *,
    iteration: int,
    seeds: list[int] | None = None,
    underwater_intrusion_penalty: float = 5.0,
) -> list[dict[str, Any]]:
    definition = cost_definition(config.env_id)
    evaluation_seeds = seeds or [
        config.seed + 1_000_000 + trial for trial in range(config.evaluation_episodes)
    ]
    env_kwargs = (
        {"high_cost_penalty": underwater_intrusion_penalty}
        if config.env_id == "UnderwaterDrone-v0"
        else {}
    )
    envs = [gym.make(config.env_id, **env_kwargs) for _ in evaluation_seeds]
    observations = np.stack(
        [safe_reset(env, seed)[0] for env, seed in zip(envs, evaluation_seeds)]
    )
    filter_ = SOOPERSafetyFilter(
        model,
        prior,
        definition,
        prior_value_model=prior_value_model,
        budget=budget,
        gamma=config.gamma,
        pessimism_beta=config.pessimism_beta,
        prior_horizon=config.prior_horizon,
        observation_low=envs[0].observation_space.low,
        observation_high=envs[0].observation_space.high,
    )
    count = len(envs)
    active = np.ones(count, dtype=bool)
    total_rewards = np.zeros(count)
    total_costs = np.zeros(count)
    intrusion_steps = np.zeros(count, dtype=int)
    interventions = np.zeros(count, dtype=int)
    reached = np.zeros(count, dtype=bool)
    lengths = np.zeros(count, dtype=int)
    predicted_cost_sums = np.zeros(count)
    expected_cost_maxima = np.full(count, -np.inf)
    uncertainty_sums = np.zeros(count)
    for _ in range(config.horizon):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        current = observations[indices]
        proposed = planner.action(current)
        prior_cost, _, uncertainty = filter_.prior_values_batch(current, proposed)
        expected = (
            total_costs[indices] + np.power(config.gamma, lengths[indices]) * prior_cost
        )
        intervene = expected >= budget
        prior_actions = np.asarray(prior.get_action(current), dtype=np.float32)
        actions = np.where(intervene[:, None], prior_actions, proposed)
        predicted_cost_sums[indices] += prior_cost
        uncertainty_sums[indices] += uncertainty
        expected_cost_maxima[indices] = np.maximum(
            expected_cost_maxima[indices], expected
        )
        interventions[indices] += intervene.astype(int)
        for local, index in enumerate(indices):
            next_observation, reward, terminated, truncated, info = envs[index].step(
                actions[local]
            )
            next_observation = np.asarray(next_observation, dtype=np.float32)
            cost = definition.transition_cost(info, next_observation)
            total_rewards[index] += float(reward)
            total_costs[index] += config.gamma ** lengths[index] * cost
            intrusion_steps[index] += int(cost)
            lengths[index] += 1
            reached[index] |= bool(
                goal_reaching_mask(config.env_id, next_observation[None])[0]
            )
            observations[index] = next_observation
            if terminated or truncated:
                active[index] = False
    for env in envs:
        env.close()
    return [
        {
            "iteration": iteration,
            "trial": trial,
            "evaluation_seed": seed,
            "episode_return": total_rewards[trial],
            "discounted_cost": total_costs[trial],
            "intrusion_steps": intrusion_steps[trial],
            "constraint_satisfied": total_costs[trial] < budget,
            "goal_reached": reached[trial],
            "interventions": interventions[trial],
            "intervention_fraction": interventions[trial] / max(lengths[trial], 1),
            "episode_length": lengths[trial],
            "mean_predicted_prior_cost": predicted_cost_sums[trial]
            / max(lengths[trial], 1),
            "max_expected_total_cost": expected_cost_maxima[trial],
            "mean_model_uncertainty": uncertainty_sums[trial] / max(lengths[trial], 1),
        }
        for trial, seed in enumerate(evaluation_seeds)
    ]


def collect_online_episode(
    planner,
    model,
    prior_value_model,
    prior,
    replay,
    config,
    budget,
    *,
    iteration,
    episode,
    rng,
):
    env = gym.make(config.env_id)
    observation, _ = safe_reset(
        env,
        config.seed
        + 100_000
        + iteration * config.real_episodes_per_iteration
        + episode,
    )
    definition = cost_definition(config.env_id)
    filter_ = SOOPERSafetyFilter(
        model,
        prior,
        definition,
        prior_value_model=prior_value_model,
        budget=budget,
        gamma=config.gamma,
        pessimism_beta=config.pessimism_beta,
        prior_horizon=config.prior_horizon,
        observation_low=env.observation_space.low,
        observation_high=env.observation_space.high,
    )
    total_reward = total_cost = 0.0
    interventions = 0
    reached = False
    predicted_prior_costs = []
    expected_total_costs = []
    model_uncertainties = []
    for step in range(config.horizon):
        proposed = planner.action(observation)
        noise = rng.normal(0.0, config.exploration_noise, size=proposed.shape)
        proposed = np.clip(
            proposed + noise, env.action_space.low, env.action_space.high
        )
        decision = filter_.decide(observation, proposed)
        predicted_prior_costs.append(decision.predicted_prior_cost)
        expected_total_costs.append(decision.expected_total_cost)
        model_uncertainties.append(decision.model_uncertainty)
        next_observation, reward, terminated, truncated, info = env.step(
            decision.action
        )
        next_observation = np.asarray(next_observation, dtype=np.float32)
        cost = definition.transition_cost(info, next_observation)
        done = bool(terminated or truncated)
        replay.add(observation, decision.action, next_observation, reward, cost, done)
        filter_.observe_cost(cost)
        total_reward += float(reward)
        total_cost += config.gamma**step * cost
        interventions += int(decision.intervention)
        reached |= bool(goal_reaching_mask(config.env_id, next_observation[None])[0])
        observation = next_observation
        if done:
            break
    env.close()
    return {
        "iteration": iteration,
        "episode": episode,
        "episode_return": total_reward,
        "discounted_cost": total_cost,
        "constraint_satisfied": total_cost < budget,
        "goal_reached": reached,
        "interventions": interventions,
        "intervention_fraction": interventions / (step + 1),
        "episode_length": step + 1,
        "mean_predicted_prior_cost": float(np.mean(predicted_prior_costs)),
        "max_expected_total_cost": float(np.max(expected_total_costs)),
        "mean_model_uncertainty": float(np.mean(model_uncertainties)),
    }


def checkpoint_payload(
    config,
    model,
    prior_value_model,
    planner,
    real_replay,
    synthetic_replay,
    iteration,
    budget,
    metrics,
    experiment_rng,
):
    return {
        "format": "calf-wrapper-sooper-v1",
        "config": asdict(config),
        "iteration": iteration,
        "budget": budget,
        "world_model": model.state_dict(),
        "world_model_is_fitted": model.is_fitted,
        "prior_value_model": prior_value_model.state_dict(),
        "prior_value_model_is_fitted": prior_value_model.is_fitted,
        "planner": planner.state_dict(),
        "real_replay": real_replay.state_dict(),
        "synthetic_replay": synthetic_replay.state_dict(),
        "metrics": metrics,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "experiment": experiment_rng.bit_generator.state,
        },
    }


def write_metrics(
    output_dir: Path, online: list[dict], evaluations: list[dict]
) -> None:
    raw = output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("online_episodes.csv", online),
        ("evaluation_trials.csv", evaluations),
    ):
        path = raw / filename
        if not rows:
            continue
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summary_for(config, budget, prior_costs, online, evaluations, elapsed, checkpoint):
    final_iteration = max(row["iteration"] for row in evaluations)
    final = [row for row in evaluations if row["iteration"] == final_iteration]
    returns = np.asarray([row["episode_return"] for row in final])
    goals = np.asarray([row["goal_reached"] for row in final], dtype=float)
    constraints = np.asarray(
        [row["constraint_satisfied"] for row in final], dtype=float
    )
    interventions = np.asarray([row["intervention_fraction"] for row in final])
    return {
        "format": "calf-wrapper-sooper-summary-v1",
        "config": asdict(config),
        "cost_definition": asdict(cost_definition(config.env_id))
        | {"from_observation": "callable", "from_transition_info": "callable"},
        "cost_budget": budget,
        "prior_calibration_costs": prior_costs,
        "completed_iterations": final_iteration + 1,
        "real_interactions": int(sum(row["episode_length"] for row in online)),
        "offline_prior_interactions": config.offline_prior_episodes * config.horizon,
        "wall_clock_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "final_metrics": {
            "mean_reward": float(returns.mean()),
            "std_reward": float(returns.std()),
            "reward_ci95_half_width": float(
                1.96 * returns.std() / np.sqrt(len(returns))
            ),
            "goal_reaching_rate": float(goals.mean()),
            "constraint_satisfaction_rate": float(constraints.mean()),
            "intervention_fraction": float(interventions.mean()),
            "n_trials": len(final),
        },
    }


def run(config: SOOPERConfig, resume: Path | None = None) -> dict[str, Any]:
    configure_determinism(config.seed)
    started = time.monotonic()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(config.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    prior = controller_for(config.environment)
    base = load_base_policy(
        config.algorithm, config.model_path, config.device, config.seed
    )
    real_replay = ReplayBuffer(obs_dim, action_dim, config.replay_capacity, config.seed)
    synthetic_replay = ReplayBuffer(
        obs_dim, action_dim, config.replay_capacity, config.seed + 1
    )
    model = ProbabilisticEnsemble(
        obs_dim, action_dim, ensemble_size=config.ensemble_size, device=config.device
    )
    prior_value_model = PriorValueEnsemble(
        obs_dim,
        action_dim,
        ensemble_size=config.ensemble_size,
        device=config.device,
    )
    planner = TD3Planner.create(
        obs_dim,
        env.action_space.low,
        env.action_space.high,
        config.device,
        config.actor_learning_rate,
    )
    env.close()
    online_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    rng = np.random.default_rng(config.seed + 2)
    start_iteration = 0
    if resume is None:
        bc_observations, bc_actions, prior_costs = collect_prior_data(
            config.env_id,
            prior,
            base,
            real_replay,
            episodes=config.offline_prior_episodes,
            horizon=config.horizon,
            gamma=config.gamma,
            seed=config.seed + 10_000,
        )
        budget = (
            config.cost_budget
            if config.cost_budget >= 0
            else max(prior_costs) * config.budget_margin + 1e-6
        )
        model_metrics = fit_world_model(
            model, prior_value_model, prior, real_replay, config, 0
        )
        if isinstance(base, CleanRLTD3):
            planner.initialize_from_cleanrl(base.actor, base.critic)
            bc_loss = 0.0
            initialization_method = "exact-cleanrl-td3-state-copy"
        else:
            bc_loss = planner.behavior_clone(
                bc_observations,
                bc_actions,
                epochs=config.behavior_clone_epochs,
                batch_size=config.batch_size,
                seed=config.seed + 20_000,
            )
            initialization_method = "behavior-cloning"
        initialization = {
            **model_metrics,
            "behavior_clone_loss": bc_loss,
            "policy_initialization": initialization_method,
        }
    else:
        payload = torch.load(resume, map_location=config.device, weights_only=False)
        if payload.get("format") != "calf-wrapper-sooper-v1":
            raise ValueError("Unsupported SOOPER checkpoint")
        model.load_state_dict(payload["world_model"])
        model.is_fitted = payload["world_model_is_fitted"]
        prior_value_model.load_state_dict(payload["prior_value_model"])
        prior_value_model.is_fitted = payload["prior_value_model_is_fitted"]
        planner.load_state_dict(payload["planner"])
        real_replay.load_state_dict(payload["real_replay"])
        synthetic_replay.load_state_dict(payload["synthetic_replay"])
        budget = float(payload["budget"])
        prior_costs = payload["metrics"]["prior_costs"]
        online_rows = payload["metrics"]["online"]
        evaluation_rows = payload["metrics"]["evaluations"]
        start_iteration = int(payload["iteration"]) + 1
        initialization = payload["metrics"].get("initialization", {})
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        restore_torch_rng_states(payload)
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng"]["experiment"]

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name) as active_run:
        # MLflow creates a random run identifier and may therefore advance the
        # process-wide Python RNG.  Establish the experiment RNG boundary only
        # after the run has been opened so independent and resumed checkpoints
        # preserve exactly the same RNG state.
        if resume is None:
            configure_determinism(config.seed)
        else:
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            restore_torch_rng_states(payload)
        mlflow.set_tags(reproducibility_tags())
        mlflow.set_tags(
            {
                "method": "SOOPER",
                "source.paper": "Wendl2026",
                "source.official_commit": "76fa2f8f576e60a4417227793dd162f031ba89be",
            }
        )
        mlflow.log_params(
            {
                k: str(v) if isinstance(v, (list, dict)) else v
                for k, v in asdict(config).items()
                if k not in {"tracking_uri"}
            }
        )
        mlflow.log_param("resolved_cost_budget", budget)
        if resume is None:
            initial_trials = evaluate(
                planner,
                model,
                prior_value_model,
                prior,
                config,
                budget,
                iteration=-1,
            )
            evaluation_rows.extend(initial_trials)
            mlflow.log_metrics(
                {
                    "eval_mean_reward": float(
                        np.mean([x["episode_return"] for x in initial_trials])
                    ),
                    "eval_goal_reaching_rate": float(
                        np.mean([x["goal_reached"] for x in initial_trials])
                    ),
                    "eval_constraint_satisfaction_rate": float(
                        np.mean([x["constraint_satisfied"] for x in initial_trials])
                    ),
                    "eval_intervention_fraction": float(
                        np.mean([x["intervention_fraction"] for x in initial_trials])
                    ),
                },
                step=0,
            )
        checkpoints = output_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        final_checkpoint = None
        for iteration in range(start_iteration, config.online_iterations):
            for episode in range(config.real_episodes_per_iteration):
                online_rows.append(
                    collect_online_episode(
                        planner,
                        model,
                        prior_value_model,
                        prior,
                        real_replay,
                        config,
                        budget,
                        iteration=iteration,
                        episode=episode,
                        rng=rng,
                    )
                )
            model_metrics = fit_world_model(
                model,
                prior_value_model,
                prior,
                real_replay,
                config,
                iteration + 1,
            )
            model_env = gym.make(config.env_id)
            rollout_metrics = model_rollouts(
                model,
                planner,
                SOOPERSafetyFilter(
                    model,
                    prior,
                    cost_definition(config.env_id),
                    prior_value_model=prior_value_model,
                    budget=budget,
                    gamma=config.gamma,
                    pessimism_beta=config.pessimism_beta,
                    prior_horizon=config.prior_horizon,
                ),
                real_replay,
                synthetic_replay,
                model_env,
                config,
                rng,
            )
            model_env.close()
            losses = {}
            source = (
                synthetic_replay
                if synthetic_replay.size >= config.batch_size
                else real_replay
            )
            for _ in range(config.policy_updates):
                losses = planner.update(
                    source.sample(min(config.batch_size, source.size))
                )
            trials = evaluate(
                planner,
                model,
                prior_value_model,
                prior,
                config,
                budget,
                iteration=iteration,
            )
            evaluation_rows.extend(trials)
            metrics = {
                **model_metrics,
                **rollout_metrics,
                **losses,
                "eval_mean_reward": float(
                    np.mean([x["episode_return"] for x in trials])
                ),
                "eval_goal_reaching_rate": float(
                    np.mean([x["goal_reached"] for x in trials])
                ),
                "eval_constraint_satisfaction_rate": float(
                    np.mean([x["constraint_satisfied"] for x in trials])
                ),
                "eval_intervention_fraction": float(
                    np.mean([x["intervention_fraction"] for x in trials])
                ),
                "real_replay_size": float(real_replay.size),
                "synthetic_replay_size": float(synthetic_replay.size),
            }
            mlflow.log_metrics(metrics, step=iteration + 1)
            if (
                (iteration + 1) % config.checkpoint_every == 0
                or iteration + 1 == config.online_iterations
            ):
                write_metrics(output_dir, online_rows, evaluation_rows)
                final_checkpoint = (
                    checkpoints / f"sooper_checkpoint_{iteration + 1:06d}.pt"
                )
                torch.save(
                    checkpoint_payload(
                        config,
                        model,
                        prior_value_model,
                        planner,
                        real_replay,
                        synthetic_replay,
                        iteration,
                        budget,
                        {
                            "prior_costs": prior_costs,
                            "initialization": initialization,
                            "online": online_rows,
                            "evaluations": evaluation_rows,
                        },
                        rng,
                    ),
                    final_checkpoint,
                )
                with tempfile.TemporaryDirectory(prefix="sooper_batch_") as temp:
                    staging = Path(temp)
                    shutil.copy2(final_checkpoint, staging / final_checkpoint.name)
                    (staging / "progress.json").write_text(
                        json.dumps(
                            {"iteration": iteration, "metrics": metrics}, indent=2
                        )
                        + "\n"
                    )
                    log_verified_artifact_batch(staging)
        assert final_checkpoint is not None
        write_metrics(output_dir, online_rows, evaluation_rows)
        summary = summary_for(
            config,
            budget,
            prior_costs,
            online_rows,
            evaluation_rows,
            time.monotonic() - started,
            final_checkpoint,
        )
        summary["mlflow_run_id"] = active_run.info.run_id
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="sooper_final_") as temp:
            staging = Path(temp)
            shutil.copy2(summary_path, staging / "summary.json")
            shutil.copytree(output_dir / "raw", staging / "raw")
            log_verified_artifact_batch(staging)
        mlflow.log_metrics(summary["final_metrics"])
        return summary


def parse_args() -> tuple[SOOPERConfig, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("--algorithm", choices=["ppo", "cleanrl_td3"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--offline-prior-episodes", type=int, default=5)
    parser.add_argument("--online-iterations", type=int, default=50)
    parser.add_argument("--real-episodes-per-iteration", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--model-epochs", type=int, default=20)
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--model-learning-rate", type=float, default=1e-3)
    parser.add_argument("--prior-value-epochs", type=int, default=20)
    parser.add_argument("--prior-value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--behavior-clone-epochs", type=int, default=50)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--model-rollout-batch", type=int, default=256)
    parser.add_argument("--model-rollout-horizon", type=int, default=5)
    parser.add_argument("--policy-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost-budget", type=float, default=-1.0)
    parser.add_argument("--budget-margin", type=float, default=1.05)
    parser.add_argument("--pessimism-beta", type=float, default=2.0)
    parser.add_argument("--prior-horizon", type=int, default=50)
    parser.add_argument("--lambda-explore", type=float, default=0.1)
    parser.add_argument("--lambda-expand", type=float, default=0.1)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    env_id, default_horizon = ENVIRONMENTS[args.environment]
    values = vars(args)
    resume = values.pop("resume")
    values["env_id"] = env_id
    values["horizon"] = args.horizon or default_horizon
    return SOOPERConfig(**values), resume


if __name__ == "__main__":
    experiment_config, resume_path = parse_args()
    result = run(experiment_config, resume_path)
    print(json.dumps(result, indent=2))
