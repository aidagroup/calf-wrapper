"""Data collection and fitting for the deployment-time goal-cost critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.advantage_intervention import GoalCostCritic
from src.controllers.controller import Controller
from src.goal_reaching import observation_reached_goal


@dataclass(frozen=True)
class GoalCostDataset:
    observations: np.ndarray
    actions: np.ndarray
    remaining_steps: np.ndarray
    goal_costs: np.ndarray
    failures: np.ndarray
    anchor_ids: np.ndarray
    action_sources: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            remaining_steps=self.remaining_steps,
            goal_costs=self.goal_costs,
            failures=self.failures,
            anchor_ids=self.anchor_ids,
            action_sources=self.action_sources,
        )


def _model_action(model, observation: np.ndarray, deterministic: bool) -> np.ndarray:
    action = model.predict(observation, deterministic=deterministic)[0]
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _fallback_action(
    fallback_policy: Controller, observation: np.ndarray
) -> np.ndarray:
    return np.asarray(
        fallback_policy.get_action(observation), dtype=np.float32
    ).reshape(-1)


def _reset(env: gym.Env, seed: int) -> np.ndarray:
    reset_output = env.reset(seed=seed)
    return np.asarray(
        reset_output[0] if isinstance(reset_output, tuple) else reset_output,
        dtype=np.float32,
    )


def _step(env: gym.Env, action: np.ndarray):
    observation, _, terminated, truncated, info = env.step(action)
    return (
        np.asarray(observation, dtype=np.float32),
        bool(terminated),
        bool(truncated),
        info,
    )


def _reached_goal(env_id: str, observation: np.ndarray, info: dict) -> bool:
    return (
        observation_reached_goal(env_id, observation)
        or bool(info.get("goal_reached", False))
        or bool(info.get("is_in_hole", False))
    )


def _candidate_outcome(
    env_id: str,
    episode_seed: int,
    prefix_actions: list[np.ndarray],
    anchor_observation: np.ndarray,
    candidate_action: np.ndarray,
    fallback_policy: Controller,
    horizon: int,
) -> tuple[float, float]:
    """Return normalized outside-goal occupancy and terminal failure."""

    env = gym.make(env_id)
    observation = _reset(env, episode_seed)
    try:
        for prefix_action in prefix_actions:
            observation, terminated, truncated, _ = _step(env, prefix_action)
            if terminated or truncated:
                raise RuntimeError(
                    "Recorded prefix terminated during deterministic replay"
                )
        if not np.allclose(observation, anchor_observation, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "Environment reset and prefix replay are not reproducible"
            )

        observation, terminated, truncated, info = _step(env, candidate_action)
        steps_left = horizon - len(prefix_actions)
        outside_goal_steps = int(not _reached_goal(env_id, observation, info))
        executed_steps = 1
        for _ in range(max(steps_left - 1, 0)):
            if terminated or truncated:
                break
            observation, terminated, truncated, info = _step(
                env, _fallback_action(fallback_policy, observation)
            )
            outside_goal_steps += int(not _reached_goal(env_id, observation, info))
            executed_steps += 1

        terminal_observation = info.get("terminal_observation")
        if terminal_observation is not None:
            observation = np.asarray(terminal_observation, dtype=np.float32)
        terminal_success = _reached_goal(env_id, observation, info)
        unexecuted_steps = max(steps_left - executed_steps, 0)
        if not terminal_success:
            outside_goal_steps += unexecuted_steps
        goal_cost = outside_goal_steps / float(steps_left)
        return float(goal_cost), float(not terminal_success)
    finally:
        env.close()


def collect_goal_cost_dataset(
    env_id: str,
    model,
    fallback_policy: Controller,
    horizon: int,
    n_anchors: int,
    seed: int,
    prefix_fallback_probability: float = 0.5,
    deterministic: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> GoalCostDataset:
    """Collect paired base/fallback targets from reproducible fork rollouts.

    Each anchor is reached by a fixed stochastic mixture of the frozen base and
    fallback policies.  Two deterministic replays then differ only in the first
    action after the anchor; both use fallback for the rest of the horizon.
    """

    if horizon <= 0 or n_anchors <= 0:
        raise ValueError("horizon and n_anchors must be positive")
    if not 0.0 <= prefix_fallback_probability <= 1.0:
        raise ValueError("prefix_fallback_probability must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    remaining_steps: list[int] = []
    goal_costs: list[float] = []
    failures: list[float] = []
    anchor_ids: list[int] = []
    action_sources: list[str] = []
    attempts = 0
    max_attempts = max(50, 20 * n_anchors)

    while len(observations) // 2 < n_anchors:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Collected only {len(observations) // 2}/{n_anchors} anchors "
                f"after {max_attempts} attempts"
            )
        episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        prefix_length = int(rng.integers(0, horizon))
        prefix_env = gym.make(env_id)
        observation = _reset(prefix_env, episode_seed)
        prefix_actions: list[np.ndarray] = []
        valid_anchor = True
        try:
            for _ in range(prefix_length):
                if rng.random() < prefix_fallback_probability:
                    action = _fallback_action(fallback_policy, observation)
                else:
                    action = _model_action(model, observation, deterministic)
                prefix_actions.append(action)
                observation, terminated, truncated, _ = _step(prefix_env, action)
                if terminated or truncated:
                    valid_anchor = False
                    break
        finally:
            prefix_env.close()
        if not valid_anchor:
            continue

        anchor_index = len(observations) // 2
        anchor_observation = np.copy(observation)
        base_action = _model_action(model, anchor_observation, deterministic)
        fallback_action = _fallback_action(fallback_policy, anchor_observation)
        steps_left = horizon - prefix_length
        for source, candidate_action in (
            ("base", base_action),
            ("fallback", fallback_action),
        ):
            goal_cost, failure = _candidate_outcome(
                env_id=env_id,
                episode_seed=episode_seed,
                prefix_actions=prefix_actions,
                anchor_observation=anchor_observation,
                candidate_action=candidate_action,
                fallback_policy=fallback_policy,
                horizon=horizon,
            )
            observations.append(anchor_observation)
            actions.append(candidate_action)
            remaining_steps.append(steps_left)
            goal_costs.append(goal_cost)
            failures.append(failure)
            anchor_ids.append(anchor_index)
            action_sources.append(source)
        if progress is not None:
            progress(anchor_index + 1, n_anchors)

    return GoalCostDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        remaining_steps=np.asarray(remaining_steps, dtype=np.float32),
        goal_costs=np.asarray(goal_costs, dtype=np.float32),
        failures=np.asarray(failures, dtype=np.float32),
        anchor_ids=np.asarray(anchor_ids, dtype=np.int64),
        action_sources=np.asarray(action_sources),
    )


def _regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "mean_absolute_error": float(np.mean(np.abs(predictions - targets))),
        "mean_squared_error": float(np.mean((predictions - targets) ** 2)),
        "mean_predicted_goal_cost": float(np.mean(predictions)),
        "mean_target_goal_cost": float(np.mean(targets)),
    }


def _paired_outcome_metrics(dataset: GoalCostDataset) -> dict:
    base_failures: list[float] = []
    fallback_failures: list[float] = []
    base_costs: list[float] = []
    fallback_costs: list[float] = []
    for anchor_id in np.unique(dataset.anchor_ids):
        anchor_mask = dataset.anchor_ids == anchor_id
        sources = dataset.action_sources[anchor_mask]
        failures = dataset.failures[anchor_mask]
        costs = dataset.goal_costs[anchor_mask]
        base_failure = failures[sources == "base"]
        fallback_failure = failures[sources == "fallback"]
        base_cost = costs[sources == "base"]
        fallback_cost = costs[sources == "fallback"]
        if any(
            len(values) != 1
            for values in (base_failure, fallback_failure, base_cost, fallback_cost)
        ):
            raise ValueError(
                "Each anchor must contain one base and one fallback target"
            )
        base_failures.append(float(base_failure[0]))
        fallback_failures.append(float(fallback_failure[0]))
        base_costs.append(float(base_cost[0]))
        fallback_costs.append(float(fallback_cost[0]))

    base_failure_array = np.asarray(base_failures)
    fallback_failure_array = np.asarray(fallback_failures)
    base_cost_array = np.asarray(base_costs)
    fallback_cost_array = np.asarray(fallback_costs)
    tolerance = 1e-8
    return {
        "base_failure_rate": float(np.mean(base_failure_array)),
        "fallback_failure_rate": float(np.mean(fallback_failure_array)),
        "base_failure_fallback_success_rate": float(
            np.mean((base_failure_array == 1.0) & (fallback_failure_array == 0.0))
        ),
        "base_success_fallback_failure_rate": float(
            np.mean((base_failure_array == 0.0) & (fallback_failure_array == 1.0))
        ),
        "equal_outcome_rate": float(
            np.mean(base_failure_array == fallback_failure_array)
        ),
        "mean_base_goal_cost": float(np.mean(base_cost_array)),
        "mean_fallback_goal_cost": float(np.mean(fallback_cost_array)),
        "mean_goal_cost_advantage": float(
            np.mean(base_cost_array - fallback_cost_array)
        ),
        "base_cost_worse_rate": float(
            np.mean(base_cost_array > fallback_cost_array + tolerance)
        ),
        "base_cost_better_rate": float(
            np.mean(base_cost_array < fallback_cost_array - tolerance)
        ),
        "equal_goal_cost_rate": float(
            np.mean(np.abs(base_cost_array - fallback_cost_array) <= tolerance)
        ),
    }


def fit_goal_cost_critic(
    dataset: GoalCostDataset,
    horizon: int,
    device: str = "cpu",
    hidden_sizes: tuple[int, ...] = (128, 128),
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[GoalCostCritic, dict]:
    """Fit a bounded goal-cost critic and report pair-wise split metrics."""

    unique_anchors = np.unique(dataset.anchor_ids)
    if len(unique_anchors) < 2:
        raise ValueError("At least two anchors are required for train/validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_anchors)
    validation_count = max(1, int(round(len(unique_anchors) * validation_fraction)))
    validation_count = min(validation_count, len(unique_anchors) - 1)
    validation_anchors = unique_anchors[:validation_count]
    validation_mask = np.isin(dataset.anchor_ids, validation_anchors)
    train_mask = ~validation_mask

    torch.manual_seed(seed)
    critic = GoalCostCritic(
        observation_dim=dataset.observations.shape[1],
        action_dim=dataset.actions.shape[1],
        horizon=horizon,
        hidden_sizes=hidden_sizes,
    ).to(device)
    critic.set_normalization(
        dataset.observations[train_mask],
        dataset.actions[train_mask],
        dataset.remaining_steps[train_mask],
    )

    train_tensors = TensorDataset(
        torch.as_tensor(dataset.observations[train_mask], dtype=torch.float32),
        torch.as_tensor(dataset.actions[train_mask], dtype=torch.float32),
        torch.as_tensor(dataset.remaining_steps[train_mask], dtype=torch.float32),
        torch.as_tensor(dataset.goal_costs[train_mask], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_tensors,
        batch_size=min(batch_size, len(train_tensors)),
        shuffle=True,
        generator=generator,
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    critic.train()
    for _ in range(epochs):
        for observations, actions, steps_left, targets in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            steps_left = steps_left.to(device)
            targets = targets.to(device)
            loss = loss_function(critic(observations, actions, steps_left), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    critic.eval()

    train_predictions = critic.goal_cost(
        dataset.observations[train_mask],
        dataset.actions[train_mask],
        dataset.remaining_steps[train_mask],
    )
    validation_predictions = critic.goal_cost(
        dataset.observations[validation_mask],
        dataset.actions[validation_mask],
        dataset.remaining_steps[validation_mask],
    )
    metrics = {
        "n_anchors": int(len(unique_anchors)),
        "n_training_anchors": int(len(unique_anchors) - validation_count),
        "n_validation_anchors": int(validation_count),
        "paired_outcomes": _paired_outcome_metrics(dataset),
        "train": _regression_metrics(train_predictions, dataset.goal_costs[train_mask]),
        "validation": _regression_metrics(
            validation_predictions, dataset.goal_costs[validation_mask]
        ),
    }
    return critic, metrics
