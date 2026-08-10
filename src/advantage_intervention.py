"""SAILR-style deployment-time intervention for fixed base policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from src.controllers.controller import Controller

CHECKPOINT_FORMAT = "calf-wrapper-goal-cost-critic-v1"


class GoalCostCritic(nn.Module):
    """Estimate fallback-continuation goal cost after one action.

    The input contains the observation, proposed action, and the fraction of the
    evaluation horizon that remains.  The target is the fraction of subsequent
    fallback-continuation states that remain outside the goal set.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        input_dim = self.observation_dim + self.action_dim + 1
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_sizes:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))

    def features(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        remaining_steps: torch.Tensor,
    ) -> torch.Tensor:
        observations = observations.reshape(-1, self.observation_dim)
        actions = actions.reshape(-1, self.action_dim)
        remaining_steps = remaining_steps.reshape(-1, 1)
        remaining_fraction = torch.clamp(
            remaining_steps / float(self.horizon), min=0.0, max=1.0
        )
        return torch.cat((observations, actions, remaining_fraction), dim=1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        remaining_steps: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(observations, actions, remaining_steps)
        normalized = (features - self.input_mean) / self.input_scale
        return torch.sigmoid(self.network(normalized).reshape(-1))

    def set_normalization(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        remaining_steps: np.ndarray,
    ) -> None:
        device = self.input_mean.device
        with torch.no_grad():
            features = self.features(
                torch.as_tensor(observations, dtype=torch.float32, device=device),
                torch.as_tensor(actions, dtype=torch.float32, device=device),
                torch.as_tensor(remaining_steps, dtype=torch.float32, device=device),
            )
            mean = features.mean(dim=0)
            scale = features.std(dim=0, unbiased=False)
            scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            self.input_mean.copy_(mean)
            self.input_scale.copy_(scale)

    def goal_cost(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        remaining_steps: int | np.ndarray,
    ) -> np.ndarray:
        device = self.input_mean.device
        observations_array = np.asarray(observations, dtype=np.float32)
        actions_array = np.asarray(actions, dtype=np.float32)
        if observations_array.ndim == 1:
            observations_array = observations_array[None, :]
        if actions_array.ndim == 1:
            actions_array = actions_array[None, :]
        remaining_array = np.broadcast_to(
            np.asarray(remaining_steps, dtype=np.float32),
            (len(observations_array),),
        ).copy()
        with torch.no_grad():
            costs = self(
                torch.as_tensor(observations_array, device=device),
                torch.as_tensor(actions_array, device=device),
                torch.as_tensor(remaining_array, device=device),
            )
            return costs.cpu().numpy()

    def save(self, path: Path | str, metadata: Optional[dict[str, Any]] = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "hidden_sizes": self.hidden_sizes,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            target,
        )

    @classmethod
    def load(
        cls, path: Path | str, device: str | torch.device = "cpu"
    ) -> tuple["GoalCostCritic", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported goal-cost critic format: {payload.get('format')}"
            )
        critic = cls(
            observation_dim=payload["observation_dim"],
            action_dim=payload["action_dim"],
            horizon=payload["horizon"],
            hidden_sizes=payload["hidden_sizes"],
        ).to(device)
        critic.load_state_dict(payload["state_dict"])
        critic.eval()
        return critic, dict(payload.get("metadata", {}))


class AdvantageInterventionWrapper(Wrapper):
    """Override base actions whose predicted goal-cost advantage is too large."""

    def __init__(
        self,
        env: VecEnv,
        goal_cost_critic: GoalCostCritic,
        fallback_policy: Controller,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.num_envs = env.num_envs
        self.goal_cost_critic = goal_cost_critic
        self.fallback_policy = fallback_policy
        self.threshold = float(threshold)
        self.elapsed_steps = 0

    def step(self, base_action: np.ndarray):
        base_action = np.asarray(base_action, dtype=np.float32)
        fallback_action = np.asarray(
            self.fallback_policy.get_action(self.obs), dtype=np.float32
        )
        remaining_steps = max(self.goal_cost_critic.horizon - self.elapsed_steps, 1)
        base_goal_cost = self.goal_cost_critic.goal_cost(
            self.obs, base_action, remaining_steps
        )
        fallback_goal_cost = self.goal_cost_critic.goal_cost(
            self.obs, fallback_action, remaining_steps
        )
        goal_cost_advantage = base_goal_cost - fallback_goal_cost
        base_action_applied = goal_cost_advantage <= self.threshold
        action = np.where(base_action_applied[:, None], base_action, fallback_action)

        env_step_output = list(self.env.step(action))
        self.obs = np.copy(env_step_output[0])
        infos = env_step_output[-1]
        if not isinstance(infos, list):
            raise TypeError(
                "AdvantageInterventionWrapper requires a vector environment"
            )
        for index, info in enumerate(infos):
            info.update(
                {
                    "intervention.base_goal_cost": float(base_goal_cost[index]),
                    "intervention.fallback_goal_cost": float(fallback_goal_cost[index]),
                    "intervention.goal_cost_advantage": float(
                        goal_cost_advantage[index]
                    ),
                    "intervention.base_action_applied": bool(
                        base_action_applied[index]
                    ),
                    "intervention.action": np.copy(action[index]),
                    "intervention.remaining_steps": int(remaining_steps),
                }
            )
        env_step_output[-1] = infos
        self.elapsed_steps += 1
        return tuple(env_step_output)

    def reset(self, *args, **kwargs):
        self.elapsed_steps = 0
        reset_output = self.env.reset(*args, **kwargs)
        self.obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        return np.copy(self.obs)
