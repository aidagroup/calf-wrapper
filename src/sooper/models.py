"""Neural components for the portable SOOPER/MBPO implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, output_dim: int, hidden: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden), nn.SiLU()))
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _relu_mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


class EnsembleMember(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int):
        super().__init__()
        # Delta state, reward and cost are modeled jointly.  Predicting a
        # log-variance yields a heteroscedastic probabilistic member; epistemic
        # uncertainty is measured across members.
        output_dim = observation_dim + 2
        self.network = _mlp(observation_dim + action_dim, 2 * output_dim, hidden)
        self.output_dim = output_dim

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_logvar = self.network(inputs).split(self.output_dim, dim=-1)
        logvar = -10.0 + F.softplus(raw_logvar + 10.0)
        logvar = 0.5 - F.softplus(0.5 - logvar)
        return mean, logvar


class ProbabilisticEnsemble(nn.Module):
    """Bootstrap ensemble predicting delta observation, reward, and cost."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        ensemble_size: int = 5,
        hidden: int = 256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.members = nn.ModuleList(
            [
                EnsembleMember(observation_dim, action_dim, hidden)
                for _ in range(ensemble_size)
            ]
        )
        self.register_buffer("input_mean", torch.zeros(observation_dim + action_dim))
        self.register_buffer("input_std", torch.ones(observation_dim + action_dim))
        self.register_buffer("target_mean", torch.zeros(observation_dim + 2))
        self.register_buffer("target_std", torch.ones(observation_dim + 2))
        self.device = torch.device(device)
        self.to(self.device)
        self.is_fitted = False

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    def _normalized_predictions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observations, actions), dim=-1)
        normalized = (inputs - self.input_mean) / self.input_std
        means, logvars = zip(*(member(normalized) for member in self.members))
        means_t = torch.stack(means)
        logvars_t = torch.stack(logvars)
        means_t = means_t * self.target_std + self.target_mean
        logvars_t = logvars_t + 2.0 * torch.log(self.target_std)
        return means_t, logvars_t

    def predict_members(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, logvars = self._normalized_predictions(observations, actions)
        delta = means[..., : self.observation_dim]
        reward = means[..., self.observation_dim]
        cost = means[..., self.observation_dim + 1].clamp(0.0, 1.0)
        next_observation = observations.unsqueeze(0) + delta
        return next_observation, reward, cost, logvars

    def predict_mean(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_obs, reward, cost, _ = self.predict_members(observations, actions)
        uncertainty = next_obs.std(dim=0, unbiased=False).norm(dim=-1)
        return next_obs.mean(0), reward.mean(0), cost.mean(0), uncertainty

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> dict[str, float]:
        inputs = np.concatenate((observations, actions), axis=-1).astype(np.float32)
        targets = np.concatenate(
            (
                (next_observations - observations).astype(np.float32),
                rewards.reshape(-1, 1).astype(np.float32),
                costs.reshape(-1, 1).astype(np.float32),
            ),
            axis=-1,
        )
        self.input_mean.copy_(torch.as_tensor(inputs.mean(0), device=self.device))
        self.input_std.copy_(
            torch.as_tensor(inputs.std(0).clip(1e-6), device=self.device)
        )
        self.target_mean.copy_(torch.as_tensor(targets.mean(0), device=self.device))
        self.target_std.copy_(
            torch.as_tensor(targets.std(0).clip(1e-6), device=self.device)
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)
        final_loss = 0.0
        self.train()
        for _ in range(epochs):
            member_indices = [
                rng.integers(0, len(inputs), len(inputs)) for _ in self.members
            ]
            for start in range(0, len(inputs), batch_size):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for member, indices in zip(self.members, member_indices):
                    selected = indices[start : start + batch_size]
                    x = torch.as_tensor(inputs[selected], device=self.device)
                    y = torch.as_tensor(targets[selected], device=self.device)
                    x = (x - self.input_mean) / self.input_std
                    y = (y - self.target_mean) / self.target_std
                    mean, logvar = member(x)
                    losses.append(
                        ((mean - y).square() * torch.exp(-logvar) + logvar).mean()
                    )
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
        self.eval()
        self.is_fitted = True
        return {"model_loss": final_loss}


class PriorValueMember(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.network = _mlp(observation_dim + action_dim, 2, hidden)

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        values = self.network(torch.cat((observation, action), dim=-1))
        return values[..., 0], F.softplus(values[..., 1])


class PriorValueEnsemble(nn.Module):
    """Fitted policy-evaluation ensemble for the conservative prior.

    This corresponds to the learned backup ``Qr``/``Qc`` heads in the official
    implementation and avoids an expensive model rollout at every real step.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        ensemble_size: int = 5,
        hidden: int = 256,
        reward_scale: float = 100.0,
        cost_scale: float = 10.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [
                PriorValueMember(observation_dim, action_dim, hidden)
                for _ in range(ensemble_size)
            ]
        )
        self.reward_scale = float(reward_scale)
        self.cost_scale = float(cost_scale)
        self.device = torch.device(device)
        self.to(self.device)
        self.is_fitted = False

    def predict_members(self, observations: torch.Tensor, actions: torch.Tensor):
        rewards, costs = zip(
            *(member(observations, actions) for member in self.members)
        )
        return (
            torch.stack(rewards) * self.reward_scale,
            torch.stack(costs) * self.cost_scale,
        )

    def fit(
        self,
        data: dict[str, np.ndarray],
        prior,
        *,
        gamma: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> dict[str, float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)
        size = len(data["rewards"])
        final_loss = 0.0
        self.train()
        for _ in range(epochs):
            targets = copy.deepcopy(self.members).to(self.device).eval()
            bootstraps = [rng.integers(0, size, size=size) for _ in self.members]
            for start in range(0, size, batch_size):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for member, target, indices in zip(self.members, targets, bootstraps):
                    selected = indices[start : start + batch_size]
                    obs = torch.as_tensor(
                        data["observations"][selected], device=self.device
                    )
                    actions = torch.as_tensor(
                        data["actions"][selected], device=self.device
                    )
                    next_obs = torch.as_tensor(
                        data["next_observations"][selected], device=self.device
                    )
                    rewards = torch.as_tensor(
                        data["rewards"][selected], device=self.device
                    )
                    costs = torch.as_tensor(data["costs"][selected], device=self.device)
                    dones = torch.as_tensor(data["dones"][selected], device=self.device)
                    next_actions = torch.as_tensor(
                        prior.get_action(next_obs.detach().cpu().numpy()),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    with torch.no_grad():
                        next_reward, next_cost = target(next_obs, next_actions)
                        target_reward = rewards + gamma * (1.0 - dones) * (
                            next_reward * self.reward_scale
                        )
                        target_cost = costs + gamma * (1.0 - dones) * (
                            next_cost * self.cost_scale
                        )
                    predicted_reward, predicted_cost = member(obs, actions)
                    loss = F.smooth_l1_loss(
                        predicted_reward, target_reward / self.reward_scale
                    ) + F.smooth_l1_loss(predicted_cost, target_cost / self.cost_scale)
                    losses.append(loss)
                total = torch.stack(losses).mean()
                total.backward()
                optimizer.step()
                final_loss = float(total.detach().cpu())
        self.eval()
        self.is_fitted = True
        return {"prior_value_loss": final_loss}


class Actor(nn.Module):
    def __init__(
        self, observation_dim: int, action_low: np.ndarray, action_high: np.ndarray
    ):
        super().__init__()
        action_dim = len(action_low)
        self.network = _relu_mlp(observation_dim, action_dim, 256)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (
            torch.tanh(self.network(observation)) * self.action_scale + self.action_bias
        )


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.q1 = _relu_mlp(observation_dim + action_dim, 1, 256)
        self.q2 = _relu_mlp(observation_dim + action_dim, 1, 256)

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        inputs = torch.cat((observation, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class ReplayBuffer:
    def __init__(self, observation_dim: int, action_dim: int, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.costs = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, next_observation, reward, cost, done):
        i = self.position
        self.observations[i] = observation
        self.actions[i] = action
        self.next_observations[i] = next_observation
        self.rewards[i] = reward
        self.costs[i] = cost
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "next_observations": self.next_observations[indices],
            "rewards": self.rewards[indices],
            "costs": self.costs[indices],
            "dones": self.dones[indices],
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations[: self.size],
            "actions": self.actions[: self.size],
            "next_observations": self.next_observations[: self.size],
            "rewards": self.rewards[: self.size],
            "costs": self.costs[: self.size],
            "dones": self.dones[: self.size],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.arrays(),
            "capacity": self.capacity,
            "position": self.position,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        count = len(state["rewards"])
        for name in (
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "costs",
            "dones",
        ):
            getattr(self, name)[:count] = state[name]
        self.size = count
        self.position = int(state["position"])
        self.rng.bit_generator.state = state["rng"]


@dataclass
class TD3Planner:
    actor: Actor
    critic: Critic
    actor_target: Actor
    critic_target: Critic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    device: torch.device
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    updates: int = 0

    @classmethod
    def create(
        cls, observation_dim, action_low, action_high, device, learning_rate=3e-4
    ):
        target_device = torch.device(device)
        actor = Actor(observation_dim, action_low, action_high).to(target_device)
        critic = Critic(observation_dim, len(action_low)).to(target_device)
        actor_target = copy.deepcopy(actor)
        critic_target = copy.deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            actor_target=actor_target,
            critic_target=critic_target,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=learning_rate),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=learning_rate),
            device=target_device,
        )

    def action(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 1
        if single:
            tensor = tensor[None]
        with torch.no_grad():
            action = self.actor(tensor).cpu().numpy()
        return action[0] if single else action

    def behavior_clone(self, observations, actions, *, epochs, batch_size, seed):
        rng = np.random.default_rng(seed)
        final = 0.0
        self.actor.train()
        for _ in range(epochs):
            order = rng.permutation(len(observations))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                obs = torch.as_tensor(observations[indices], device=self.device)
                target = torch.as_tensor(actions[indices], device=self.device)
                loss = F.mse_loss(self.actor(obs), target)
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.actor_optimizer.step()
                final = float(loss.detach().cpu())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.eval()
        return final

    def initialize_from_cleanrl(self, base_actor, base_critic) -> None:
        """Exactly copy the vendored CleanRL TD3 actor and twin critics."""

        def copy_linear(target: nn.Linear, source: nn.Linear) -> None:
            target.weight.data.copy_(source.weight.data.to(self.device))
            target.bias.data.copy_(source.bias.data.to(self.device))

        copy_linear(self.actor.network[0], base_actor.fc1)
        copy_linear(self.actor.network[2], base_actor.fc2)
        copy_linear(self.actor.network[4], base_actor.fc_mu)
        self.actor.action_scale.copy_(base_actor.action_scale.to(self.device))
        self.actor.action_bias.copy_(base_actor.action_bias.to(self.device))
        for target_network, source_network in (
            (self.critic.q1, base_critic.qf1),
            (self.critic.q2, base_critic.qf2),
        ):
            copy_linear(target_network[0], source_network.fc1)
            copy_linear(target_network[2], source_network.fc2)
            copy_linear(target_network[4], source_network.fc3)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensors = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }
        obs, actions, next_obs = (
            tensors["observations"],
            tensors["actions"],
            tensors["next_observations"],
        )
        rewards, dones = tensors["rewards"].view(-1, 1), tensors["dones"].view(-1, 1)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = self.actor_target(next_obs) + noise
            low = self.actor.action_bias - self.actor.action_scale
            high = self.actor.action_bias + self.actor.action_scale
            next_action = torch.maximum(torch.minimum(next_action, high), low)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            target = rewards + (1.0 - dones) * self.gamma * torch.minimum(q1_t, q2_t)
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss_value = float("nan")
        if self.updates % self.policy_frequency == 0:
            actor_loss = -self.critic.q1(
                torch.cat((obs, self.actor(obs)), dim=-1)
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            with torch.no_grad():
                for source, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
                for source, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates = int(state["updates"])
