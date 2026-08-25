"""Train the TD3 base policies used by the article evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim
from torch.nn import functional as F

import calfwrapper.environments  # noqa: F401
from calfwrapper.paths import TRAINING_OUTPUT

ENVIRONMENTS = {
    "auv": "CALFWrapper/ContaminatedZoneAUV-v0",
    "robot": "CALFWrapper/TreasureCollectingRobot-v0",
}


@dataclass(frozen=True)
class Settings:
    environment: str
    seed: int
    device: str
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    gamma: float
    tau: float
    batch_size: int
    policy_noise: float
    exploration_noise: float
    learning_starts: int
    policy_frequency: int
    noise_clip: float
    checkpoint_dir: Path
    checkpoint_every: int


class Actor(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.spaces.Box):
        super().__init__()
        observation_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(observation))
        hidden = F.relu(self.fc2(hidden))
        return torch.tanh(self.fc_mu(hidden)) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.spaces.Box):
        super().__init__()
        input_dim = int(np.prod(observation_space.shape) + np.prod(action_space.shape))
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((observation, action), dim=1)
        hidden = F.relu(self.fc1(hidden))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)


def make_environment(environment_id: str, seed: int):
    def create() -> gym.Env:
        environment = gym.make(environment_id, seed=seed)
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment.action_space.seed(seed)
        return environment

    return create


def save_checkpoint(
    settings: Settings,
    completed_steps: int,
    actor: Actor,
    target_actor: Actor,
    qf1: QNetwork,
    qf2: QNetwork,
    qf1_target: QNetwork,
    qf2_target: QNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    observation: np.ndarray,
    replay_buffer: ReplayBuffer,
) -> Path:
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = settings.checkpoint_dir / f"td3_checkpoint_{completed_steps}_steps.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    payload = {
        "format": "calfwrapper-td3-v1",
        "completed_steps": completed_steps,
        "env_id": ENVIRONMENTS[settings.environment],
        "seed": settings.seed,
        "algorithm": {
            "learning_rate": settings.learning_rate,
            "buffer_size": settings.buffer_size,
            "gamma": settings.gamma,
            "tau": settings.tau,
            "batch_size": settings.batch_size,
            "policy_noise": settings.policy_noise,
            "exploration_noise": settings.exploration_noise,
            "learning_starts": settings.learning_starts,
            "policy_frequency": settings.policy_frequency,
            "noise_clip": settings.noise_clip,
        },
        "actor": actor.state_dict(),
        "target_actor": target_actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "observation": np.asarray(observation).copy(),
        "replay_buffer": {
            "position": replay_buffer.pos,
            "full": replay_buffer.full,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": payload["format"],
                "completed_steps": completed_steps,
                "env_id": payload["env_id"],
                "seed": settings.seed,
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {checkpoint}", flush=True)
    return checkpoint


def train(settings: Settings) -> None:
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.backends.cudnn.deterministic = True
    if settings.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    device = torch.device(settings.device)
    environment_id = ENVIRONMENTS[settings.environment]
    environments = gym.vector.SyncVectorEnv(
        [make_environment(environment_id, settings.seed)]
    )
    observation_space = environments.single_observation_space
    action_space = environments.single_action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError("TD3 requires a continuous Box action space")

    # These constructors consume the seeded PyTorch RNG in the order used by
    # the published training runs.
    actor = Actor(observation_space, action_space).to(device)
    qf1 = QNetwork(observation_space, action_space).to(device)
    qf2 = QNetwork(observation_space, action_space).to(device)
    qf1_target = QNetwork(observation_space, action_space).to(device)
    qf2_target = QNetwork(observation_space, action_space).to(device)
    target_actor = Actor(observation_space, action_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=settings.learning_rate)
    q_optimizer = optim.Adam(
        (*qf1.parameters(), *qf2.parameters()),
        lr=settings.learning_rate,
    )
    observation_space.dtype = np.float32
    replay_buffer = ReplayBuffer(
        settings.buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        handle_timeout_termination=False,
    )

    observation, _ = environments.reset(seed=settings.seed)
    for step in range(settings.total_timesteps):
        if step < settings.learning_starts:
            action = np.asarray([action_space.sample()])
        else:
            with torch.no_grad():
                action_tensor = actor(
                    torch.as_tensor(observation, dtype=torch.float32, device=device)
                )
                action_tensor += torch.normal(
                    0,
                    actor.action_scale * settings.exploration_noise,
                )
                action = action_tensor.cpu().numpy().clip(action_space.low, action_space.high)

        action = np.asarray(action, dtype=float)
        next_observation, reward, terminated, truncated, info = environments.step(action)
        replay_next_observation = next_observation.copy()
        for index, was_truncated in enumerate(truncated):
            if was_truncated:
                replay_next_observation[index] = info["final_observation"][index]
        replay_buffer.add(
            observation,
            replay_next_observation,
            action,
            reward,
            terminated,
            info,
        )
        observation = next_observation

        if step > settings.learning_starts:
            data = replay_buffer.sample(settings.batch_size)
            with torch.no_grad():
                noise = (
                    torch.randn_like(data.actions) * settings.policy_noise
                ).clamp(-settings.noise_clip, settings.noise_clip)
                noise *= target_actor.action_scale
                next_action = (target_actor(data.next_observations) + noise).clamp(
                    float(action_space.low[0]),
                    float(action_space.high[0]),
                )
                next_q = torch.minimum(
                    qf1_target(data.next_observations, next_action),
                    qf2_target(data.next_observations, next_action),
                ).view(-1)
                target_q = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * settings.gamma * next_q

            qf1_loss = F.mse_loss(qf1(data.observations, data.actions).view(-1), target_q)
            qf2_loss = F.mse_loss(qf2(data.observations, data.actions).view(-1), target_q)
            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            q_optimizer.step()

            if step % settings.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                for source, target in (
                    (actor, target_actor),
                    (qf1, qf1_target),
                    (qf2, qf2_target),
                ):
                    for parameter, target_parameter in zip(
                        source.parameters(), target.parameters(), strict=True
                    ):
                        target_parameter.data.copy_(
                            settings.tau * parameter.data
                            + (1 - settings.tau) * target_parameter.data
                        )

        completed_steps = step + 1
        if completed_steps % settings.checkpoint_every == 0:
            save_checkpoint(
                settings,
                completed_steps,
                actor,
                target_actor,
                qf1,
                qf2,
                qf1_target,
                qf2_target,
                actor_optimizer,
                q_optimizer,
                observation,
                replay_buffer,
            )

    if settings.total_timesteps % settings.checkpoint_every:
        save_checkpoint(
            settings,
            settings.total_timesteps,
            actor,
            target_actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
            observation,
            replay_buffer,
        )
    environments.close()


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=30_000)
    arguments = parser.parse_args()
    checkpoint_dir = arguments.checkpoint_dir or (
        TRAINING_OUTPUT / f"td3-{arguments.environment}" / "checkpoints"
    )
    return Settings(
        **{key: value for key, value in vars(arguments).items() if key != "checkpoint_dir"},
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    train(parse_args())
