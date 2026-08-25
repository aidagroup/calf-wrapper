"""TD3 models and checkpoint loading used by evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TD3Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.zeros(action_dim))
        self.register_buffer("action_bias", torch.zeros(action_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(observation))
        hidden = F.relu(self.fc2(hidden))
        action = torch.tanh(self.fc_mu(hidden))
        return action * self.action_scale + self.action_bias


class TD3Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([observation, action], dim=1)
        hidden = F.relu(self.fc1(hidden))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)


class TwinTD3Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.qf1 = TD3Critic(observation_dim, action_dim)
        self.qf2 = TD3Critic(observation_dim, action_dim)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qf1(observation, action), self.qf2(observation, action)


class TD3Policy:
    """Small inference-only facade matching the model API used by evaluation."""

    def __init__(
        self,
        actor: TD3Actor,
        critic: TwinTD3Critic,
        device: torch.device,
        metadata: dict[str, Any],
    ):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.metadata = metadata

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> TD3Policy:
        target_device = torch.device(device)
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        supported_formats = {"calfwrapper-td3-v1", "calf-enhance-cleanrl-td3-v1"}
        if payload.get("format") not in supported_formats:
            raise ValueError(f"Unsupported TD3 checkpoint format: {payload.get('format')}")

        actor_state = payload["actor"]
        observation_dim = actor_state["fc1.weight"].shape[1]
        action_dim = actor_state["fc_mu.weight"].shape[0]
        actor = TD3Actor(observation_dim, action_dim).to(target_device)
        critic = TwinTD3Critic(observation_dim, action_dim).to(target_device)
        actor.load_state_dict(actor_state)
        critic.qf1.load_state_dict(payload["qf1"])
        critic.qf2.load_state_dict(payload["qf2"])
        actor.eval()
        critic.eval()
        metadata = {
            key: payload.get(key)
            for key in ("format", "source_commit", "completed_steps", "env_id", "seed")
        }
        return cls(actor, critic, target_device, metadata)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        del deterministic  # The deterministic actor has no exploration branch.
        is_single = observation.ndim == 1
        batch = observation.reshape(1, -1) if is_single else observation
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(tensor).cpu().numpy()
        return (actions[0] if is_single else actions), None
