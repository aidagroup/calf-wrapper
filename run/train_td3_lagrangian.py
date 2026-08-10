"""Standalone CleanRL-style TD3-Lagrangian trainer and evaluator.

This file deliberately contains the complete algorithm.  It follows the pinned
CleanRL TD3 loop used by the repository and adds a terminal-failure cost critic,
a projected Lagrange multiplier, time-aware observations, and finite-horizon
evaluation.
"""

from __future__ import annotations

import json
import math
import random
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from scipy.stats import beta

import src  # noqa: F401  # Register custom environments.
from src import run_path
from src.goal_reaching import goal_reaching_mask

CHECKPOINT_FORMAT = "calf-wrapper-cleanrl-td3-lagrangian-v1"


@dataclass
class Args:
    environment: str = "underwater-drone"
    env_id: str = "UnderwaterDrone-v0"
    horizon: int = 1500
    total_timesteps: int = 3_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 25_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    reward_scale: float = 0.01
    tau: float = 0.005
    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    noise_clip: float = 0.5
    policy_frequency: int = 2
    cost_limit: float = 0.05
    lambda_init: float = 0.0
    lambda_lr: float = 0.1
    lambda_update_episodes: int = 20
    lambda_max: float = 10_000.0
    seed: int = 0
    device: str = "cpu"
    torch_deterministic: bool = True
    evaluation_episodes: int = 200
    evaluation_seed: int = 10_000
    paired_evaluation_episodes: int = 30
    paired_evaluation_seed: int = 42
    checkpoint_every: int = 30_000
    log_every: int = 1_000
    output_dir: Path = run_path / "artifacts" / "td3_lagrangian_underwater"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "CALF-Wrapper/Lagrangian-Baselines"
    mlflow_run_name: str | None = None


PRESETS = {
    "underwater-drone": (
        "TD3-Lagrangian on Contaminated-Zone AUV Navigation",
        Args(),
    ),
    "robot-navigation": (
        "TD3-Lagrangian on Treasure-Collecting Robot",
        Args(
            environment="robot-navigation",
            env_id="RobotNavigationConstSpeedCatch-v0",
            horizon=1000,
            seed=1,
            output_dir=run_path / "artifacts" / "td3_lagrangian_robot",
        ),
    ),
}


def serializable_config(args: Args) -> dict[str, object]:
    config = asdict(args)
    config["output_dir"] = str(config["output_dir"])
    return config


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def log_mlflow_metrics(record: dict[str, object], step: int, prefix: str) -> None:
    """Log the finite numeric portion of a record to the active MLflow run."""
    if mlflow.active_run() is None:
        return
    metrics: dict[str, float] = {}
    for key, value in record.items():
        if key == "step" or not isinstance(value, (bool, int, float)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            metrics[f"{prefix}/{key}"] = numeric_value
    if metrics:
        mlflow.log_metrics(metrics, step=step)


def source_metadata() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


class TimeAwareObservation(gym.Wrapper):
    """Append elapsed-time fraction to the physical observation."""

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.elapsed_steps = 0
        low = np.concatenate(
            [np.asarray(env.observation_space.low, dtype=np.float32), [0.0]]
        )
        high = np.concatenate(
            [np.asarray(env.observation_space.high, dtype=np.float32), [1.0]]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        fraction = min(self.elapsed_steps / self.horizon, 1.0)
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), [fraction]]
        ).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.elapsed_steps = 0
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        return self._augment(observation), reward, terminated, truncated, info


def make_env(env_id: str, horizon: int, seed: int):
    def thunk():
        env = gym.make(env_id, max_episode_steps=horizon, seed=seed)
        env = TimeAwareObservation(env, horizon)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def physical_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32)[..., :-1]


def reached_goal(
    env_id: str, observation: np.ndarray, info: dict | None = None
) -> bool:
    info = info or {}
    return bool(
        goal_reaching_mask(env_id, physical_observation(observation))[0]
        or info.get("goal_reached", False)
        or info.get("is_in_hole", False)
    )


def terminal_costs(
    env_id: str,
    next_observations: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    infos: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return terminal failure costs and the true final next observations."""

    episode_ends = np.logical_or(terminations, truncations)
    costs = np.zeros(len(episode_ends), dtype=np.float32)
    real_next_observations = np.asarray(next_observations).copy()
    final_observations = infos.get("final_observation")
    final_infos = infos.get("final_info")
    for index, ended in enumerate(episode_ends):
        if not ended:
            continue
        observation = next_observations[index]
        if final_observations is not None and final_observations[index] is not None:
            observation = final_observations[index]
            real_next_observations[index] = observation
        info = {}
        if final_infos is not None and final_infos[index] is not None:
            info = final_infos[index]
        costs[index] = 0.0 if reached_goal(env_id, observation, info) else 1.0
    return costs, real_next_observations


def update_lagrange_multiplier(
    value: float,
    mean_episode_cost: float,
    cost_limit: float,
    learning_rate: float,
    upper_bound: float,
) -> float:
    return float(
        np.clip(
            value + learning_rate * (mean_episode_cost - cost_limit),
            0.0,
            upper_bound,
        )
    )


def reward_bellman_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    reward_scale: float,
) -> torch.Tensor:
    return reward_scale * rewards + (1.0 - dones) * gamma * next_values


def cost_bellman_target(
    costs: torch.Tensor, dones: torch.Tensor, next_values: torch.Tensor
) -> torch.Tensor:
    """Undiscounted terminal-failure target with no timeout bootstrap."""

    return costs + (1.0 - dones) * next_values


def tensor_gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    squared_norm = sum(
        gradient.detach().pow(2).sum() for gradient in gradients if gradient is not None
    )
    return float(torch.sqrt(squared_norm).item())


def parameter_gradient_norm(parameters) -> float:
    return tensor_gradient_norm(tuple(parameter.grad for parameter in parameters))


def require_finite(name: str, *tensors: torch.Tensor) -> None:
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError(f"non-finite values detected in {name}")


def actor_lagrangian_loss(
    reward_value: torch.Tensor,
    cost_value: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    return -reward_value + lambda_value * cost_value


@torch.no_grad()
def estimate_initial_failure_probability(
    actor: nn.Module,
    cost_q: nn.Module,
    initial_observations: torch.Tensor,
) -> tuple[float, float]:
    actions = actor(initial_observations)
    raw_estimates = cost_q(initial_observations, actions).squeeze(-1)
    require_finite("initial-state failure estimates", raw_estimates)
    return (
        float(raw_estimates.mean().item()),
        float(raw_estimates.clamp(0.0, 1.0).mean().item()),
    )


class ReplayBuffer:
    """Minimal replay buffer with explicit cost and true episode-end storage."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: torch.device,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((capacity,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.next_observations[self.position] = next_observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.costs[self.position] = cost
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(
                self.observations[indices], device=self.device
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=self.device
            ),
            "actions": torch.as_tensor(self.actions[indices], device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device),
            "costs": torch.as_tensor(self.costs[indices], device=self.device),
            "dones": torch.as_tensor(self.dones[indices], device=self.device),
        }


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        return self.network(torch.cat([observation, action], dim=1))


class Actor(nn.Module):
    def __init__(self, observation_size: int, action_space: gym.spaces.Box):
        super().__init__()
        action_size = int(np.prod(action_space.shape))
        self.network = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32
            ),
        )

    def forward(self, observation: torch.Tensor):
        return self.network(observation) * self.action_scale + self.action_bias


def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - half_width, center + half_width


def clopper_pearson_failure_upper(
    failures: int, trials: int, confidence: float = 0.95
) -> float:
    """Return the one-sided exact upper bound on binomial failure probability."""

    if trials <= 0:
        return float("nan")
    if failures >= trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


@torch.no_grad()
def evaluate(actor: Actor, args: Args, device: torch.device) -> dict[str, object]:
    returns: list[float] = []
    costs: list[float] = []
    trials: list[dict[str, object]] = []
    was_training = actor.training
    actor.eval()
    for episode in range(args.evaluation_episodes):
        trial_seed = args.evaluation_seed + episode
        env = TimeAwareObservation(
            gym.make(
                args.env_id,
                max_episode_steps=args.horizon,
                seed=trial_seed,
            ),
            args.horizon,
        )
        observation, _ = env.reset(seed=trial_seed)
        episode_return = 0.0
        episode_length = 0
        terminated = truncated = False
        final_info: dict = {}
        while not (terminated or truncated):
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = actor(tensor)[0].cpu().numpy()
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
        returns.append(episode_return)
        episode_cost = (
            0.0 if reached_goal(args.env_id, observation, final_info) else 1.0
        )
        costs.append(episode_cost)
        trials.append(
            {
                "trial": episode,
                "seed": trial_seed,
                "episode_return": episode_return,
                "episode_cost": episode_cost,
                "goal_reached": not bool(episode_cost),
                "episode_length": episode_length,
            }
        )
        env.close()
    actor.train(was_training)
    returns_array = np.asarray(returns, dtype=np.float64)
    costs_array = np.asarray(costs, dtype=np.float64)
    successes = int(np.sum(1.0 - costs_array))
    failures = len(costs) - successes
    lower, upper = wilson_interval(successes, len(costs))
    failure_upper = clopper_pearson_failure_upper(failures, len(costs))
    return {
        "mean_reward": float(returns_array.mean()),
        "std_reward": float(returns_array.std()),
        "reward_ci95_half_width": float(
            1.96 * returns_array.std() / math.sqrt(len(returns_array))
        ),
        "mean_episode_cost": float(costs_array.mean()),
        "goal_reaching_probability": float(1.0 - costs_array.mean()),
        "goal_reaching_rate_percent": float(100.0 * (1.0 - costs_array.mean())),
        "goal_rate_wilson95_low": float(lower),
        "goal_rate_wilson95_high": float(upper),
        "failure_probability_upper95": failure_upper,
        "constraint_assessment_eligible": bool(len(costs) >= 200),
        "upper_bound_below_limit": bool(failure_upper <= args.cost_limit),
        "constraint_assessment_passed": bool(
            len(costs) >= 200 and failure_upper <= args.cost_limit
        ),
        "evaluation_episodes": len(costs),
        "trials": trials,
    }


def save_checkpoint(
    path: Path,
    args: Args,
    global_step: int,
    lambda_value: float,
    actor: Actor,
    target_actor: Actor,
    reward_q1: QNetwork,
    reward_q2: QNetwork,
    target_reward_q1: QNetwork,
    target_reward_q2: QNetwork,
    cost_q: QNetwork,
    target_cost_q: QNetwork,
    actor_optimizer: optim.Optimizer,
    reward_q_optimizer: optim.Optimizer,
    cost_q_optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "args": serializable_config(args),
            "runtime": source_metadata(),
            "observation_shape": replay_buffer.observations.shape[1:],
            "action_shape": replay_buffer.actions.shape[1:],
            "global_step": global_step,
            "lambda": lambda_value,
            "actor": actor.state_dict(),
            "target_actor": target_actor.state_dict(),
            "reward_q1": reward_q1.state_dict(),
            "reward_q2": reward_q2.state_dict(),
            "target_reward_q1": target_reward_q1.state_dict(),
            "target_reward_q2": target_reward_q2.state_dict(),
            "cost_q": cost_q.state_dict(),
            "target_cost_q": target_cost_q.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "reward_q_optimizer": reward_q_optimizer.state_dict(),
            "cost_q_optimizer": cost_q_optimizer.state_dict(),
            "replay": {
                "size": replay_buffer.size,
                "position": replay_buffer.position,
                "capacity": replay_buffer.capacity,
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_actor_checkpoint(
    path: Path,
    args: Args,
    actor: Actor,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported TD3-Lagrangian checkpoint: {path}")
    saved_args = payload.get("args", {})
    if (
        saved_args.get("env_id") != args.env_id
        or saved_args.get("horizon") != args.horizon
    ):
        raise ValueError("checkpoint environment or horizon is incompatible")
    if tuple(payload.get("observation_shape", ())) != tuple(observation_shape):
        raise ValueError("checkpoint observation shape is incompatible")
    if tuple(payload.get("action_shape", ())) != tuple(action_shape):
        raise ValueError("checkpoint action shape is incompatible")
    actor.load_state_dict(payload["actor"])
    return payload


def run_training(args: Args) -> None:
    supported_tasks = {
        ("UnderwaterDrone-v0", 1500),
        ("RobotNavigationConstSpeedCatch-v0", 1000),
    }
    if (args.env_id, args.horizon) not in supported_tasks:
        raise ValueError(
            "unsupported environment/horizon pair for TD3-Lagrangian: "
            f"{args.env_id}/{args.horizon}"
        )
    if not 0.0 <= args.cost_limit <= 1.0:
        raise ValueError("cost_limit must lie in [0, 1]")
    if args.horizon <= 0 or args.total_timesteps <= 0:
        raise ValueError("horizon and total_timesteps must be positive")
    if args.learning_starts >= args.total_timesteps:
        raise ValueError("learning_starts must be smaller than total_timesteps")
    if args.lambda_update_episodes <= 0:
        raise ValueError("lambda_update_episodes must be positive")
    if args.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    if args.paired_evaluation_episodes <= 0:
        raise ValueError("paired_evaluation_episodes must be positive")
    if args.batch_size <= 0 or args.buffer_size < args.batch_size:
        raise ValueError("buffer_size must be at least the positive batch_size")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.tau <= 1.0:
        raise ValueError("gamma and tau must lie in (0, 1]")
    if args.reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    if args.policy_frequency <= 0:
        raise ValueError("policy_frequency must be positive")
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    if min(args.policy_noise, args.exploration_noise, args.noise_clip) < 0.0:
        raise ValueError("noise scales must be nonnegative")
    if not 0.0 <= args.lambda_init <= args.lambda_max:
        raise ValueError("lambda_init must lie in [0, lambda_max]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(args.device)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.horizon, args.seed)])
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("TD3-Lagrangian supports continuous actions only")
    observation_size = int(np.prod(envs.single_observation_space.shape))
    action_size = int(np.prod(envs.single_action_space.shape))
    actor = Actor(observation_size, envs.single_action_space).to(device)
    target_actor = Actor(observation_size, envs.single_action_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    reward_q1 = QNetwork(observation_size, action_size).to(device)
    reward_q2 = QNetwork(observation_size, action_size).to(device)
    target_reward_q1 = QNetwork(observation_size, action_size).to(device)
    target_reward_q2 = QNetwork(observation_size, action_size).to(device)
    target_reward_q1.load_state_dict(reward_q1.state_dict())
    target_reward_q2.load_state_dict(reward_q2.state_dict())
    cost_q = QNetwork(observation_size, action_size).to(device)
    target_cost_q = QNetwork(observation_size, action_size).to(device)
    target_cost_q.load_state_dict(cost_q.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    reward_q_optimizer = optim.Adam(
        list(reward_q1.parameters()) + list(reward_q2.parameters()),
        lr=args.learning_rate,
    )
    cost_q_optimizer = optim.Adam(cost_q.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        device,
    )

    observation, _ = envs.reset(seed=args.seed)
    episode_initial_observation = observation[0].copy()
    episode_return = 0.0
    episode_length = 0
    pending_dual_initial_observations: list[np.ndarray] = []
    completed_behavior_costs: list[float] = []
    completed_episodes = 0
    lambda_value = float(args.lambda_init)
    latest_diagnostics: dict[str, float] = {}
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        args.output_dir / f"td3_lagrangian_{args.environment}_seed{args.seed}.jsonl"
    )
    metrics_path.write_text("")
    checkpoint_path = (
        args.output_dir / f"td3_lagrangian_{args.environment}_seed{args.seed}.pt"
    )

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action = np.asarray([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                action_tensor = actor(
                    torch.as_tensor(observation, dtype=torch.float32, device=device)
                )
                noise = torch.normal(
                    mean=0.0,
                    std=actor.action_scale * args.exploration_noise,
                )
                action = (action_tensor + noise).cpu().numpy()
            action = np.clip(
                action, envs.single_action_space.low, envs.single_action_space.high
            )
        if not np.isfinite(action).all():
            raise FloatingPointError("non-finite action produced during training")

        next_observation, reward, terminated, truncated, infos = envs.step(action)
        done = np.logical_or(terminated, truncated)
        cost, real_next_observation = terminal_costs(
            args.env_id, next_observation, terminated, truncated, infos
        )
        replay_buffer.add(
            observation[0],
            real_next_observation[0],
            action[0],
            float(reward[0]),
            float(cost[0]),
            bool(done[0]),
        )
        episode_return += float(reward[0])
        episode_length += 1
        observation = next_observation

        if done[0]:
            completed_episodes += 1
            completed_behavior_costs.append(float(cost[0]))
            pending_dual_initial_observations.append(episode_initial_observation)
            behavior_mean_cost = float(np.mean(completed_behavior_costs[-100:]))
            episode_record = {
                "record_type": "episode",
                "step": global_step + 1,
                "episode": completed_episodes,
                "episode_return": episode_return,
                "episode_cost": float(cost[0]),
                "rolling_behavior_cost": behavior_mean_cost,
                "lambda": lambda_value,
            }
            append_jsonl(metrics_path, episode_record)
            log_mlflow_metrics(episode_record, global_step + 1, "train/episode")
            print(json.dumps(episode_record), flush=True)
            episode_return = 0.0
            episode_length = 0
            episode_initial_observation = next_observation[0].copy()

        if (
            global_step >= args.learning_starts
            and replay_buffer.size >= args.batch_size
        ):
            data = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data["actions"]) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale
                next_action = target_actor(data["next_observations"]) + clipped_noise
                next_action = torch.max(
                    torch.min(
                        next_action,
                        target_actor.action_bias + target_actor.action_scale,
                    ),
                    target_actor.action_bias - target_actor.action_scale,
                )
                next_reward_q = torch.min(
                    target_reward_q1(data["next_observations"], next_action),
                    target_reward_q2(data["next_observations"], next_action),
                ).squeeze(-1)
                reward_target = reward_bellman_target(
                    data["rewards"],
                    data["dones"],
                    next_reward_q,
                    args.gamma,
                    args.reward_scale,
                )
                next_cost_q = target_cost_q(
                    data["next_observations"], next_action
                ).squeeze(-1)
                cost_target = cost_bellman_target(
                    data["costs"], data["dones"], next_cost_q
                )
                require_finite("TD3 targets", reward_target, cost_target, next_action)

            reward_q1_value = reward_q1(data["observations"], data["actions"]).squeeze(
                -1
            )
            reward_q2_value = reward_q2(data["observations"], data["actions"]).squeeze(
                -1
            )
            reward_q_loss = F.mse_loss(reward_q1_value, reward_target) + F.mse_loss(
                reward_q2_value, reward_target
            )
            reward_q_optimizer.zero_grad()
            reward_q_loss.backward()
            reward_q_gradient_norm = parameter_gradient_norm(
                list(reward_q1.parameters()) + list(reward_q2.parameters())
            )
            reward_q_optimizer.step()

            cost_q_value = cost_q(data["observations"], data["actions"]).squeeze(-1)
            cost_q_loss = F.mse_loss(cost_q_value, cost_target)
            cost_q_optimizer.zero_grad()
            cost_q_loss.backward()
            cost_q_gradient_norm = parameter_gradient_norm(cost_q.parameters())
            cost_q_optimizer.step()
            require_finite("critic losses", reward_q_loss, cost_q_loss)

            if global_step % args.policy_frequency == 0:
                for network in (reward_q1, cost_q):
                    for parameter in network.parameters():
                        parameter.requires_grad_(False)
                actor_action = actor(data["observations"])
                actor_reward_value = reward_q1(
                    data["observations"], actor_action
                ).mean()
                actor_cost_value = cost_q(data["observations"], actor_action).mean()
                actor_loss = actor_lagrangian_loss(
                    actor_reward_value, actor_cost_value, lambda_value
                )
                reward_actor_gradients = torch.autograd.grad(
                    -actor_reward_value,
                    tuple(actor.parameters()),
                    retain_graph=True,
                )
                cost_actor_gradients = torch.autograd.grad(
                    actor_cost_value,
                    tuple(actor.parameters()),
                    retain_graph=True,
                )
                reward_actor_gradient_norm = tensor_gradient_norm(
                    reward_actor_gradients
                )
                cost_actor_gradient_norm = tensor_gradient_norm(cost_actor_gradients)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_gradient_norm = parameter_gradient_norm(actor.parameters())
                actor_optimizer.step()
                require_finite(
                    "actor update",
                    actor_loss,
                    actor_reward_value,
                    actor_cost_value,
                    actor_action,
                )
                for network in (reward_q1, cost_q):
                    for parameter in network.parameters():
                        parameter.requires_grad_(True)

                with torch.no_grad():
                    for parameter, target_parameter in zip(
                        actor.parameters(), target_actor.parameters()
                    ):
                        target_parameter.mul_(1.0 - args.tau).add_(
                            parameter, alpha=args.tau
                        )
                    for source, target in (
                        (reward_q1, target_reward_q1),
                        (reward_q2, target_reward_q2),
                        (cost_q, target_cost_q),
                    ):
                        for parameter, target_parameter in zip(
                            source.parameters(), target.parameters()
                        ):
                            target_parameter.mul_(1.0 - args.tau).add_(
                                parameter, alpha=args.tau
                            )
                latest_diagnostics.update(
                    {
                        "reward_q_loss": float(reward_q_loss.item()),
                        "cost_q_loss": float(cost_q_loss.item()),
                        "actor_loss": float(actor_loss.item()),
                        "actor_reward_value": float(actor_reward_value.item()),
                        "actor_cost_value": float(actor_cost_value.item()),
                        "cost_q_out_of_range_fraction": float(
                            ((cost_q_value < 0.0) | (cost_q_value > 1.0))
                            .float()
                            .mean()
                            .item()
                        ),
                        "reward_q_gradient_norm": reward_q_gradient_norm,
                        "cost_q_gradient_norm": cost_q_gradient_norm,
                        "reward_actor_gradient_norm": reward_actor_gradient_norm,
                        "cost_actor_gradient_norm": cost_actor_gradient_norm,
                        "weighted_cost_actor_gradient_norm": (
                            lambda_value * cost_actor_gradient_norm
                        ),
                        "actor_gradient_norm": actor_gradient_norm,
                    }
                )

        while (
            global_step >= args.learning_starts
            and len(pending_dual_initial_observations) >= args.lambda_update_episodes
        ):
            initial_observations = np.stack(
                pending_dual_initial_observations[: args.lambda_update_episodes]
            )
            del pending_dual_initial_observations[: args.lambda_update_episodes]
            initial_tensor = torch.as_tensor(
                initial_observations, dtype=torch.float32, device=device
            )
            raw_failure_estimate, failure_estimate = (
                estimate_initial_failure_probability(actor, cost_q, initial_tensor)
            )
            lambda_value = update_lagrange_multiplier(
                lambda_value,
                failure_estimate,
                args.cost_limit,
                args.lambda_lr,
                args.lambda_max,
            )
            latest_diagnostics.update(
                {
                    "dual_raw_failure_estimate": raw_failure_estimate,
                    "dual_clipped_failure_estimate": failure_estimate,
                }
            )

        completed_steps = global_step + 1
        if completed_steps % args.log_every == 0 and latest_diagnostics:
            diagnostics_record = {
                "record_type": "diagnostics",
                "step": completed_steps,
                "lambda": lambda_value,
                **latest_diagnostics,
            }
            append_jsonl(metrics_path, diagnostics_record)
            log_mlflow_metrics(diagnostics_record, completed_steps, "train/diagnostics")
        if args.checkpoint_every > 0 and completed_steps % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                args,
                completed_steps,
                lambda_value,
                actor,
                target_actor,
                reward_q1,
                reward_q2,
                target_reward_q1,
                target_reward_q2,
                cost_q,
                target_cost_q,
                actor_optimizer,
                reward_q_optimizer,
                cost_q_optimizer,
                replay_buffer,
            )

    save_checkpoint(
        checkpoint_path,
        args,
        args.total_timesteps,
        lambda_value,
        actor,
        target_actor,
        reward_q1,
        reward_q2,
        target_reward_q1,
        target_reward_q2,
        cost_q,
        target_cost_q,
        actor_optimizer,
        reward_q_optimizer,
        cost_q_optimizer,
        replay_buffer,
    )
    envs.close()
    evaluation = evaluate(actor, args, device)
    paired_args = replace(
        args,
        evaluation_episodes=args.paired_evaluation_episodes,
        evaluation_seed=args.paired_evaluation_seed,
    )
    paired_evaluation = evaluate(actor, paired_args, device)
    if not math.isclose(
        float(evaluation["mean_episode_cost"]),
        1.0 - float(evaluation["goal_reaching_probability"]),
        abs_tol=1e-12,
    ):
        raise AssertionError("episode cost and goal-reaching rate disagree")
    evaluation.update(
        {
            "lambda": lambda_value,
            "cost_limit": args.cost_limit,
            "constraint_satisfied_empirically": bool(
                float(evaluation["mean_episode_cost"]) <= args.cost_limit
            ),
            "lambda_upper_bound_reached": bool(lambda_value >= args.lambda_max),
            "global_step": args.total_timesteps,
            "completed_training_episodes": completed_episodes,
            "training_behavior_successes": int(
                len(completed_behavior_costs) - sum(completed_behavior_costs)
            ),
            "training_behavior_failures": int(sum(completed_behavior_costs)),
            "elapsed_training_seconds": time.time() - started_at,
            "reward_scale": args.reward_scale,
            "checkpoint_path": str(checkpoint_path),
            "training_metrics_path": str(metrics_path),
            "environment": args.environment,
            "env_id": args.env_id,
            "horizon": args.horizon,
            "training_seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
            "config": serializable_config(args),
            "runtime": source_metadata(),
            "paired_evaluation": paired_evaluation,
            **latest_diagnostics,
        }
    )
    active_run = mlflow.active_run()
    evaluation["mlflow"] = {
        "tracking_uri": args.mlflow_tracking_uri,
        "experiment_name": args.mlflow_experiment_name,
        "run_name": args.mlflow_run_name,
        "run_id": active_run.info.run_id if active_run is not None else None,
    }
    result_path = (
        args.output_dir / f"td3_lagrangian_{args.environment}_seed{args.seed}.json"
    )
    result_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    log_mlflow_metrics(evaluation, args.total_timesteps, "evaluation")
    log_mlflow_metrics(paired_evaluation, args.total_timesteps, "evaluation/paired")
    if active_run is not None:
        mlflow.log_artifacts(str(args.output_dir), artifact_path="outputs")
    print(json.dumps(evaluation, indent=2), flush=True)


def main(args: Args) -> None:
    if args.mlflow_tracking_uri is None:
        run_training(args)
        return
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    run_name = args.mlflow_run_name or (
        f"td3-lagrangian__{args.environment}__seed-{args.seed}"
    )
    args.mlflow_run_name = run_name
    with mlflow.start_run(run_name=run_name):
        runtime = source_metadata()
        mlflow.set_tags(
            {
                "repro.run_status": "RUNNING",
                "algorithm": "td3-lagrangian",
                "environment": args.environment,
                "env_id": args.env_id,
                "training_seed": str(args.seed),
                **{f"runtime.{key}": str(value) for key, value in runtime.items()},
            }
        )
        parameters = serializable_config(args)
        parameters.pop("mlflow_tracking_uri", None)
        mlflow.log_params(parameters)
        try:
            run_training(args)
        except BaseException as error:
            mlflow.set_tags(
                {
                    "repro.run_status": "FAILED",
                    "repro.failure_type": type(error).__name__,
                }
            )
            raise
        mlflow.set_tag("repro.run_status", "COMPLETED")


if __name__ == "__main__":
    main(tyro.extras.overridable_config_cli(PRESETS))
