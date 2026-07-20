"""SOOPER online cost filter and prior-value estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.controllers.controller import Controller
from .costs import CostDefinition
from .models import ProbabilisticEnsemble


@dataclass
class FilterDecision:
    action: np.ndarray
    intervention: bool
    accumulated_cost: float
    predicted_prior_cost: float
    expected_total_cost: float
    model_uncertainty: float


class SOOPERSafetyFilter:
    """Algorithm-1 policy prior switch using an ensemble-pessimistic Qc."""

    def __init__(
        self,
        model: ProbabilisticEnsemble,
        prior: Controller,
        cost: CostDefinition,
        *,
        budget: float,
        gamma: float = 0.99,
        pessimism_beta: float = 2.0,
        prior_horizon: int = 50,
        observation_low: np.ndarray | None = None,
        observation_high: np.ndarray | None = None,
    ):
        self.model = model
        self.prior = prior
        self.cost = cost
        self.budget = float(budget)
        self.gamma = float(gamma)
        self.pessimism_beta = float(pessimism_beta)
        self.prior_horizon = int(prior_horizon)
        self.observation_low = observation_low
        self.observation_high = observation_high
        self.reset()

    def reset(self) -> None:
        self.accumulated_cost = 0.0
        self.step = 0

    def observe_cost(self, cost: float) -> None:
        self.accumulated_cost += self.gamma**self.step * float(cost)
        self.step += 1

    def _clip(self, observation: torch.Tensor) -> torch.Tensor:
        if self.observation_low is None or self.observation_high is None:
            return observation
        low = torch.as_tensor(self.observation_low, device=observation.device)
        high = torch.as_tensor(self.observation_high, device=observation.device)
        finite_low = torch.where(torch.isfinite(low), low, observation)
        finite_high = torch.where(torch.isfinite(high), high, observation)
        return torch.maximum(torch.minimum(observation, finite_high), finite_low)

    @torch.no_grad()
    def prior_values_batch(
        self, observations: np.ndarray, first_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.model.device
        )
        actions = torch.as_tensor(
            first_actions, dtype=torch.float32, device=self.model.device
        )
        if obs.ndim == 1:
            obs = obs[None]
        if actions.ndim == 1:
            actions = actions[None]
        batch_size = len(obs)
        member_costs = torch.zeros(
            self.model.ensemble_size, batch_size, device=self.model.device
        )
        member_rewards = torch.zeros_like(member_costs)
        uncertainty_sum = torch.zeros(batch_size, device=self.model.device)
        member_obs = obs.unsqueeze(0).repeat(self.model.ensemble_size, 1, 1)
        for offset in range(self.prior_horizon):
            if offset > 0:
                flattened = member_obs.reshape(
                    self.model.ensemble_size * batch_size, -1
                )
                prior_actions = self.prior.get_action(flattened.detach().cpu().numpy())
                action_members = torch.as_tensor(
                    prior_actions, dtype=torch.float32, device=self.model.device
                ).reshape(self.model.ensemble_size, batch_size, -1)
            else:
                action_members = actions.unsqueeze(0).repeat(
                    self.model.ensemble_size, 1, 1
                )
            next_members = []
            rewards = []
            costs = []
            for index, member in enumerate(self.model.members):
                inputs = torch.cat((member_obs[index], action_members[index]), dim=-1)
                normalized = (inputs - self.model.input_mean) / self.model.input_std
                mean, _ = member(normalized)
                mean = mean * self.model.target_std + self.model.target_mean
                next_members.append(
                    member_obs[index] + mean[:, : self.model.observation_dim]
                )
                rewards.append(mean[:, self.model.observation_dim])
                costs.append(mean[:, self.model.observation_dim + 1].clamp(0.0, 1.0))
            member_obs = self._clip(torch.stack(next_members))
            member_rewards += self.gamma**offset * torch.stack(rewards)
            member_costs += self.gamma**offset * torch.stack(costs)
            uncertainty_sum += (
                member_obs.std(0, unbiased=False).norm(dim=-1) * self.gamma**offset
            )
        pessimistic_cost = member_costs.mean(
            0
        ) + self.pessimism_beta * member_costs.std(0, unbiased=False)
        pessimistic_reward = member_rewards.mean(
            0
        ) - self.pessimism_beta * member_rewards.std(0, unbiased=False)
        return (
            pessimistic_cost.cpu().numpy(),
            pessimistic_reward.cpu().numpy(),
            uncertainty_sum.cpu().numpy(),
        )

    def prior_values(
        self, observation: np.ndarray, first_action: np.ndarray
    ) -> tuple[float, float, float]:
        costs, rewards, uncertainties = self.prior_values_batch(
            observation, first_action
        )
        return float(costs[0]), float(rewards[0]), float(uncertainties[0])

    def decide(
        self, observation: np.ndarray, proposed_action: np.ndarray
    ) -> FilterDecision:
        if not self.model.is_fitted:
            action = np.asarray(self.prior.get_action(observation), dtype=np.float32)
            return FilterDecision(
                action,
                True,
                self.accumulated_cost,
                float("inf"),
                float("inf"),
                float("inf"),
            )
        prior_cost, _, uncertainty = self.prior_values(observation, proposed_action)
        expected = self.accumulated_cost + self.gamma**self.step * prior_cost
        intervention = expected >= self.budget
        action = self.prior.get_action(observation) if intervention else proposed_action
        return FilterDecision(
            np.asarray(action, dtype=np.float32),
            intervention,
            self.accumulated_cost,
            prior_cost,
            expected,
            uncertainty,
        )
