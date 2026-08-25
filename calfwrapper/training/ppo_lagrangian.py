"""PPO-Lagrangian trainer and evaluator.

The constrained objective uses a single undiscounted terminal cost:
``1`` iff the episode ends outside the prescribed goal set, and ``0``
otherwise.  Time-to-horizon is appended to every observation so that this
finite-horizon CMDP remains Markov.
"""

from __future__ import annotations

import hashlib
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
import torch.optim as optim
import tyro
from scipy.stats import beta
from torch.distributions.normal import Normal

import calfwrapper.environments  # noqa: F401
from calfwrapper.goal_reaching import goal_reaching_mask
from calfwrapper.paths import TRAINING_OUTPUT

CHECKPOINT_FORMAT = "calf-wrapper-cleanrl-ppo-lagrangian-v1"


@dataclass
class Args:
    environment: str = "pendulum"
    env_id: str = "Pendulum-v1"
    horizon: int = 200
    total_timesteps: int = 102_000
    num_envs: int = 1
    num_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    learning_rate: float = 1e-3
    anneal_lr: bool = False
    anneal_lr_start_fraction: float = 0.5
    reward_scale: float = 0.01
    gamma: float = 0.98
    gae_lambda: float = 0.95
    cost_gae_lambda: float = 0.95
    redistribute_terminal_cost: bool = False
    normalize_advantage_across_rollout: bool = False
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    reward_vf_coef: float = 0.5
    cost_vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    initial_action_std: float = 2.0
    cost_limit: float = 0.05
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    lambda_update_episodes: int = 20
    lambda_max: float = 10_000.0
    dual_start_after_first_success: bool = False
    max_action_std_after_first_success: float | None = None
    seed: int = 9
    device: str = "cpu"
    torch_deterministic: bool = True
    evaluation_episodes: int = 200
    evaluation_seed: int = 10_000
    paired_evaluation_episodes: int = 30
    paired_evaluation_seed: int = 42
    save_model_every_steps: int = 3_000
    evaluation_checkpoint: Path | None = None
    output_dir: Path = TRAINING_OUTPUT / "ppo-lagrangian"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "CALF-Wrapper/Lagrangian-Baselines"
    mlflow_run_name: str | None = None
    cartpole_terminate_on_out_of_bounds: bool = True
    cartpole_saturate_state_on_out_of_bounds: bool = False
    cartpole_position_termination_threshold: float = 5.0
    cartpole_velocity_termination_threshold: float = 8.0
    cartpole_angular_velocity_termination_threshold: float = 10.0
    cartpole_reward_position_clip: float | None = None


PRESETS = {
    "pendulum": (
        "PPO-Lagrangian on Pendulum-v1",
        Args(),
    ),
    "cartpole": (
        "PPO-Lagrangian on the 1000-step CartPole evaluation task",
        Args(
            environment="cartpole",
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            horizon=1000,
            total_timesteps=300_000,
            anneal_lr=True,
            cost_gae_lambda=1.0,
            redistribute_terminal_cost=False,
            normalize_advantage_across_rollout=True,
            initial_action_std=10.0,
            lambda_lr=0.05,
            seed=42,
            output_dir=TRAINING_OUTPUT / "ppo-lagrangian-cartpole",
        ),
    ),
    "cartpole-wide-600k": (
        "PPO-Lagrangian on wide-bound 1000-step CartPole through 600k steps",
        Args(
            environment="cartpole",
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            horizon=1000,
            total_timesteps=600_000,
            anneal_lr=True,
            cost_gae_lambda=1.0,
            redistribute_terminal_cost=False,
            normalize_advantage_across_rollout=True,
            initial_action_std=10.0,
            lambda_lr=0.05,
            seed=42,
            output_dir=TRAINING_OUTPUT / "ppo-lagrangian-cartpole-wide",
            cartpole_terminate_on_out_of_bounds=True,
            cartpole_position_termination_threshold=7.5,
            cartpole_velocity_termination_threshold=12.0,
            cartpole_angular_velocity_termination_threshold=15.0,
            cartpole_reward_position_clip=5.0,
        ),
    ),
    "cartpole-saturated-600k": (
        "PPO-Lagrangian on nonterminating saturated 1000-step CartPole through 600k steps",
        Args(
            environment="cartpole",
            env_id="CALFWrapper/CartPoleSwingUpLong-v0",
            horizon=1000,
            total_timesteps=600_000,
            anneal_lr=True,
            cost_gae_lambda=1.0,
            redistribute_terminal_cost=False,
            normalize_advantage_across_rollout=True,
            initial_action_std=10.0,
            lambda_lr=0.05,
            seed=42,
            output_dir=TRAINING_OUTPUT / "ppo-lagrangian-cartpole-saturated",
            cartpole_terminate_on_out_of_bounds=False,
            cartpole_saturate_state_on_out_of_bounds=True,
            cartpole_position_termination_threshold=7.5,
            cartpole_velocity_termination_threshold=12.0,
            cartpole_angular_velocity_termination_threshold=15.0,
            cartpole_reward_position_clip=5.0,
        ),
    ),
}


def serializable_config(args: Args) -> dict[str, object]:
    config = asdict(args)
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
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
    """Append elapsed-time fraction to preserve the finite-horizon Markov state."""

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.elapsed_steps = 0
        low = np.concatenate([np.asarray(env.observation_space.low, dtype=np.float32), [0.0]])
        high = np.concatenate([np.asarray(env.observation_space.high, dtype=np.float32), [1.0]])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        fraction = min(self.elapsed_steps / self.horizon, 1.0)
        return np.concatenate([np.asarray(observation, dtype=np.float32), [fraction]]).astype(
            np.float32
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.elapsed_steps = 0
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        return self._augment(observation), reward, terminated, truncated, info


def cartpole_env_kwargs(args: Args) -> dict[str, object]:
    if args.env_id != "CALFWrapper/CartPoleSwingUpLong-v0":
        return {}
    return {
        "terminate_on_out_of_bounds": args.cartpole_terminate_on_out_of_bounds,
        "saturate_state_on_out_of_bounds": (args.cartpole_saturate_state_on_out_of_bounds),
        "position_termination_threshold": args.cartpole_position_termination_threshold,
        "velocity_termination_threshold": args.cartpole_velocity_termination_threshold,
        "angular_velocity_termination_threshold": (
            args.cartpole_angular_velocity_termination_threshold
        ),
        "reward_position_clip": args.cartpole_reward_position_clip,
    }


def make_env(
    env_id: str,
    horizon: int,
    seed: int,
    index: int,
    *,
    env_kwargs: dict[str, object] | None = None,
):
    def thunk():
        env = gym.make(env_id, max_episode_steps=horizon, **(env_kwargs or {}))
        env = TimeAwareObservation(env, horizon)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + index)
        return env

    return thunk


def physical_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32)[..., :-1]


def reached_goal(env_id: str, observation: np.ndarray, info: dict | None = None) -> bool:
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
) -> np.ndarray:
    """Return one binary failure cost per transition."""

    episode_ends = np.logical_or(terminations, truncations)
    costs = np.zeros(len(episode_ends), dtype=np.float32)
    final_observations = infos.get("final_observation")
    final_infos = infos.get("final_info")
    for index, ended in enumerate(episode_ends):
        if not ended:
            continue
        observation = next_observations[index]
        if final_observations is not None and final_observations[index] is not None:
            observation = final_observations[index]
        info = {}
        if final_infos is not None and final_infos[index] is not None:
            info = final_infos[index]
        costs[index] = 0.0 if reached_goal(env_id, observation, info) else 1.0
    return costs


def cartpole_cost_potential(observations: np.ndarray) -> np.ndarray:
    """Return a bounded potential used only to redistribute terminal cost."""

    physical = physical_observation(np.asarray(observations, dtype=np.float32))
    x = physical[..., 0]
    x_dot = physical[..., 1]
    angle = np.arctan2(physical[..., 3], physical[..., 2])
    theta_dot = physical[..., 4]
    state_loss = 0.5 * angle**2 + 0.5 * x**2 + 0.05 * theta_dot**2 + 0.05 * x_dot**2
    return state_loss / (1.0 + state_loss)


def transition_costs(
    args: Args,
    next_observations: np.ndarray,
    episode_ends: np.ndarray,
    infos: dict,
    binary_terminal_costs: np.ndarray,
    current_potentials: np.ndarray,
    initial_potentials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redistribute binary episode cost without changing its episode sum."""

    if not args.redistribute_terminal_cost:
        return (
            binary_terminal_costs.copy(),
            np.zeros_like(current_potentials),
            np.zeros_like(initial_potentials),
        )
    if args.env_id != "CALFWrapper/CartPoleSwingUpLong-v0":
        raise ValueError("terminal-cost redistribution is implemented for CartPole")

    final_observations = infos.get("final_observation")
    successor_observations = np.asarray(next_observations, dtype=np.float32).copy()
    if final_observations is not None:
        for index, ended in enumerate(episode_ends):
            if ended and final_observations[index] is not None:
                successor_observations[index] = final_observations[index]
    successor_potentials = cartpole_cost_potential(successor_observations)
    reset_potentials = cartpole_cost_potential(next_observations)
    redistributed = successor_potentials - current_potentials
    for index, ended in enumerate(episode_ends):
        if ended:
            redistributed[index] = (
                binary_terminal_costs[index] - current_potentials[index] + initial_potentials[index]
            )
            successor_potentials[index] = reset_potentials[index]
            initial_potentials[index] = reset_potentials[index]
    return redistributed.astype(np.float32), successor_potentials, initial_potentials


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


def record_episode_cost_for_dual(
    episode_cost: float,
    dual_active: bool,
    pending_costs: list[float],
    start_after_first_success: bool,
) -> bool:
    """Start dual updates once the sampled terminal costs are nonconstant."""

    if not dual_active and start_after_first_success and episode_cost == 0.0:
        dual_active = True
        pending_costs.clear()
    if dual_active:
        pending_costs.append(episode_cost)
    return dual_active


def normalized_lagrangian_advantage(
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    combined = (reward_advantage - lambda_value * cost_advantage) / (1.0 + lambda_value)
    if len(combined) > 1:
        combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    return combined


def generalized_advantages(
    deltas: torch.Tensor,
    episode_ends: torch.Tensor,
    trace_coefficient: float,
) -> torch.Tensor:
    """Propagate one-step residuals without crossing episode boundaries."""

    advantages = torch.zeros_like(deltas)
    last_advantage = torch.zeros(deltas.shape[1], device=deltas.device)
    for step in reversed(range(len(deltas))):
        nonterminal = 1.0 - episode_ends[step]
        last_advantage = deltas[step] + trace_coefficient * nonterminal * last_advantage
        advantages[step] = last_advantage
    return advantages


def delayed_linear_learning_rate(
    initial_rate: float,
    update: int,
    total_updates: int,
    start_fraction: float,
) -> float:
    progress = (update - 1.0) / total_updates
    if progress <= start_fraction:
        return initial_rate
    fraction_remaining = (1.0 - progress) / (1.0 - start_fraction)
    return initial_rate * fraction_remaining


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv, initial_action_std: float = 1.0):
        super().__init__()
        if initial_action_std <= 0.0:
            raise ValueError("initial_action_std must be positive")
        observation_size = int(np.prod(envs.single_observation_space.shape))
        action_size = int(np.prod(envs.single_action_space.shape))
        self.reward_critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_size), std=0.01),
        )
        action_scale = torch.as_tensor(
            (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
            dtype=torch.float32,
        )
        initial_latent_std = torch.as_tensor(initial_action_std, dtype=torch.float32) / action_scale
        self.actor_logstd = nn.Parameter(initial_latent_std.log().reshape(1, -1))
        self.register_buffer("action_scale", action_scale)
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def get_values(self, observation: torch.Tensor):
        return (
            self.reward_critic(observation).squeeze(-1),
            self.cost_critic(observation).squeeze(-1),
        )

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        latent_action: torch.Tensor | None = None,
        *,
        estimate_entropy: bool = False,
    ):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        distribution = Normal(action_mean, action_logstd.exp())
        if latent_action is None:
            latent_action = distribution.rsample()
        squashed_action = torch.tanh(latent_action)
        action = self.action_bias + self.action_scale * squashed_action
        log_probability = distribution.log_prob(latent_action) - torch.log(
            self.action_scale * (1.0 - squashed_action.pow(2)) + 1e-6
        )
        log_probability = log_probability.sum(1)
        if estimate_entropy:
            entropy_latent = distribution.rsample()
            entropy_squashed = torch.tanh(entropy_latent)
            entropy_log_probability = distribution.log_prob(entropy_latent) - torch.log(
                self.action_scale * (1.0 - entropy_squashed.pow(2)) + 1e-6
            )
            entropy = -entropy_log_probability.sum(1)
        else:
            entropy = torch.zeros_like(log_probability)
        reward_value, cost_value = self.get_values(observation)
        return (
            action,
            latent_action,
            log_probability,
            entropy,
            reward_value,
            cost_value,
        )

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(self.actor_mean(observation))

    @torch.no_grad()
    def cap_action_std(self, maximum_action_std: float) -> None:
        if maximum_action_std <= 0.0:
            raise ValueError("maximum action standard deviation must be positive")
        maximum_latent_std = maximum_action_std / self.action_scale
        self.actor_logstd.copy_(
            torch.minimum(self.actor_logstd, maximum_latent_std.log().reshape(1, -1))
        )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def clopper_pearson_failure_upper(failures: int, trials: int, confidence: float = 0.95) -> float:
    """Return the one-sided exact upper bound on binomial failure probability."""

    if trials <= 0:
        return float("nan")
    if failures >= trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


def save_checkpoint(
    path: Path,
    args: Args,
    agent: Agent,
    optimizer: optim.Optimizer,
    lambda_value: float,
    global_step: int,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "args": serializable_config(args),
            "runtime": source_metadata(),
            "observation_shape": observation_shape,
            "action_shape": action_shape,
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lambda": lambda_value,
            "global_step": global_step,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_agent_checkpoint(
    path: Path,
    args: Args,
    agent: Agent,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported PPO-Lagrangian checkpoint: {path}")
    saved_args = payload.get("args", {})
    if saved_args.get("env_id") != args.env_id or saved_args.get("horizon") != args.horizon:
        raise ValueError("checkpoint environment or horizon is incompatible")
    if tuple(payload.get("observation_shape", ())) != tuple(observation_shape):
        raise ValueError("checkpoint observation shape is incompatible")
    if tuple(payload.get("action_shape", ())) != tuple(action_shape):
        raise ValueError("checkpoint action shape is incompatible")
    agent.load_state_dict(payload["agent"])
    return payload


@torch.no_grad()
def evaluate(
    agent: Agent,
    args: Args,
    device: torch.device,
    *,
    stochastic: bool,
) -> dict[str, object]:
    returns: list[float] = []
    costs: list[float] = []
    trials: list[dict[str, object]] = []
    action_low: np.ndarray | None = None
    action_high: np.ndarray | None = None
    was_training = agent.training
    agent.eval()
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    for episode in range(args.evaluation_episodes):
        trial_seed = args.evaluation_seed + episode
        env = TimeAwareObservation(
            gym.make(
                args.env_id,
                max_episode_steps=args.horizon,
                **cartpole_env_kwargs(args),
            ),
            args.horizon,
        )
        observation, _ = env.reset(seed=trial_seed)
        if action_low is None:
            action_low = env.action_space.low
            action_high = env.action_space.high
        episode_return = 0.0
        episode_length = 0
        terminated = truncated = False
        final_info: dict = {}
        first_goal_step: int | None = 0 if reached_goal(args.env_id, observation) else None
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(trial_seed)
            while not (terminated or truncated):
                tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                if stochastic:
                    action = agent.get_action_and_value(tensor.unsqueeze(0))[0][0]
                else:
                    action = agent.deterministic_action(tensor.unsqueeze(0))[0]
                action_array = np.clip(action.cpu().numpy(), action_low, action_high)
                observation, reward, terminated, truncated, final_info = env.step(action_array)
                episode_return += float(reward)
                episode_length += 1
                if first_goal_step is None and reached_goal(args.env_id, observation, final_info):
                    first_goal_step = episode_length
        returns.append(episode_return)
        episode_cost = 0.0 if reached_goal(args.env_id, observation, final_info) else 1.0
        costs.append(episode_cost)
        trials.append(
            {
                "trial": episode,
                "seed": trial_seed,
                "policy_noise_seed": trial_seed if stochastic else None,
                "episode_return": episode_return,
                "episode_cost": episode_cost,
                "goal_reached": not bool(episode_cost),
                "goal_visited": first_goal_step is not None,
                "first_goal_step": first_goal_step,
                "episode_length": episode_length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_observation": physical_observation(observation).tolist(),
            }
        )
        env.close()
    agent.train(was_training)
    returns_array = np.asarray(returns, dtype=np.float64)
    costs_array = np.asarray(costs, dtype=np.float64)
    successes = int(np.sum(1.0 - costs_array))
    failures = len(costs) - successes
    lower, upper = wilson_interval(successes, len(costs))
    failure_upper = clopper_pearson_failure_upper(failures, len(costs))
    return {
        "mean_reward": float(returns_array.mean()),
        "std_reward": float(returns_array.std()),
        "reward_ci95_half_width": float(1.96 * returns_array.std() / math.sqrt(len(returns_array))),
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
        "policy": "stochastic" if stochastic else "deterministic_mean",
        "trials": trials,
    }


def run_training(args: Args) -> None:
    supported_tasks = {
        ("Pendulum-v1", 200),
        ("CALFWrapper/CartPoleSwingUpLong-v0", 1000),
    }
    if (args.env_id, args.horizon) not in supported_tasks:
        raise ValueError(
            f"unsupported environment/horizon pair for PPO-Lagrangian: {args.env_id}/{args.horizon}"
        )
    if args.num_steps < args.horizon:
        raise ValueError("num_steps must cover at least one complete episode")
    if not 0.0 <= args.cost_limit <= 1.0:
        raise ValueError("cost_limit must lie in [0, 1]")
    if args.lambda_update_episodes <= 0:
        raise ValueError("lambda_update_episodes must be positive")
    if args.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    if args.paired_evaluation_episodes <= 0:
        raise ValueError("paired_evaluation_episodes must be positive")
    if args.save_model_every_steps <= 0:
        raise ValueError("save_model_every_steps must be positive")
    if args.env_id == "CALFWrapper/CartPoleSwingUpLong-v0":
        for name, value in (
            (
                "cartpole_position_termination_threshold",
                args.cartpole_position_termination_threshold,
            ),
            (
                "cartpole_velocity_termination_threshold",
                args.cartpole_velocity_termination_threshold,
            ),
            (
                "cartpole_angular_velocity_termination_threshold",
                args.cartpole_angular_velocity_termination_threshold,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            args.cartpole_reward_position_clip is not None
            and args.cartpole_reward_position_clip <= 0
        ):
            raise ValueError("cartpole_reward_position_clip must be positive")
    if args.num_envs <= 0 or args.num_minibatches <= 0 or args.update_epochs <= 0:
        raise ValueError("num_envs, num_minibatches, and update_epochs must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if args.reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    if args.initial_action_std <= 0.0:
        raise ValueError("initial_action_std must be positive")
    if (
        args.max_action_std_after_first_success is not None
        and args.max_action_std_after_first_success <= 0.0
    ):
        raise ValueError("max_action_std_after_first_success must be positive")
    if not 0.0 <= args.anneal_lr_start_fraction < 1.0:
        raise ValueError("anneal_lr_start_fraction must lie in [0, 1)")
    if not 0.0 <= args.lambda_init <= args.lambda_max:
        raise ValueError("lambda_init must lie in [0, lambda_max]")
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches:
        raise ValueError("num_steps * num_envs must divide evenly into minibatches")
    minibatch_size = batch_size // args.num_minibatches

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"ppo_lagrangian_{args.environment}_seed{args.seed}.jsonl"
    metrics_path.write_text("")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.horizon,
                args.seed,
                index,
                env_kwargs=cartpole_env_kwargs(args),
            )
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("PPO-Lagrangian currently supports continuous actions only")
    agent = Agent(envs, initial_action_std=args.initial_action_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observations = torch.zeros((args.num_steps, args.num_envs) + observation_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    log_probabilities = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    episode_ends = torch.zeros((args.num_steps, args.num_envs)).to(device)
    reward_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    update = 0
    lambda_value = float(args.lambda_init)
    completed_costs: list[float] = []
    pending_lambda_costs: list[float] = []
    dual_active = not args.dual_start_after_first_success
    dual_activation_step: int | None = 0 if dual_active else None
    action_std_cap_active = bool(
        dual_active and args.max_action_std_after_first_success is not None
    )
    next_observation, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(next_observation, dtype=torch.float32, device=device)
    current_cost_potentials = (
        cartpole_cost_potential(next_observation.cpu().numpy())
        if args.redistribute_terminal_cost
        else np.zeros(args.num_envs, dtype=np.float32)
    )
    initial_cost_potentials = current_cost_potentials.copy()
    redistributed_episode_costs = np.zeros(args.num_envs, dtype=np.float64)
    started_at = time.time()
    total_updates = math.ceil(args.total_timesteps / batch_size)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    next_checkpoint_step = args.save_model_every_steps

    while global_step < args.total_timesteps:
        update += 1
        if args.anneal_lr:
            current_learning_rate = delayed_linear_learning_rate(
                args.learning_rate,
                update,
                total_updates,
                args.anneal_lr_start_fraction,
            )
            optimizer.param_groups[0]["lr"] = current_learning_rate
        else:
            current_learning_rate = args.learning_rate
        saturation_values: list[float] = []
        for step in range(args.num_steps):
            if global_step >= args.total_timesteps:
                break
            global_step += args.num_envs
            observations[step] = next_observation
            with torch.no_grad():
                (
                    env_action,
                    latent_action,
                    log_probability,
                    _,
                    reward_value,
                    cost_value,
                ) = agent.get_action_and_value(next_observation)
            actions[step] = latent_action
            log_probabilities[step] = log_probability
            reward_values[step] = reward_value
            cost_values[step] = cost_value
            saturation_values.append(
                float((torch.tanh(latent_action).abs() > 0.99).float().mean().item())
            )
            if not bool(torch.isfinite(env_action).all()):
                raise FloatingPointError("non-finite PPO action")
            next_obs_np, reward_np, terminated, truncated, infos = envs.step(
                env_action.cpu().numpy()
            )
            end_np = np.logical_or(terminated, truncated)
            binary_cost_np = terminal_costs(args.env_id, next_obs_np, terminated, truncated, infos)
            cost_np, current_cost_potentials, initial_cost_potentials = transition_costs(
                args,
                next_obs_np,
                end_np,
                infos,
                binary_cost_np,
                current_cost_potentials,
                initial_cost_potentials,
            )
            redistributed_episode_costs += cost_np
            rewards[step] = args.reward_scale * torch.as_tensor(reward_np, device=device)
            costs[step] = torch.as_tensor(cost_np, device=device)
            episode_ends[step] = torch.as_tensor(end_np, device=device)
            for index, ended in enumerate(end_np):
                if ended:
                    value = float(binary_cost_np[index])
                    if args.redistribute_terminal_cost and not np.isclose(
                        redistributed_episode_costs[index], value, atol=1e-5
                    ):
                        raise AssertionError(
                            "redistributed transition costs do not sum to the binary episode cost"
                        )
                    redistributed_episode_costs[index] = 0.0
                    completed_costs.append(value)
                    was_dual_active = dual_active
                    dual_active = record_episode_cost_for_dual(
                        value,
                        dual_active,
                        pending_lambda_costs,
                        args.dual_start_after_first_success,
                    )
                    if dual_active and not was_dual_active:
                        dual_activation_step = global_step
            next_observation = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            if global_step >= next_checkpoint_step:
                periodic_checkpoint = checkpoint_dir / f"ppo_checkpoint_{global_step}_steps.pt"
                save_checkpoint(
                    periodic_checkpoint,
                    args,
                    agent,
                    optimizer,
                    lambda_value,
                    global_step,
                    observation_shape,
                    action_shape,
                )
                if mlflow.active_run() is not None:
                    mlflow.log_artifact(str(periodic_checkpoint), artifact_path="checkpoints")
                    mlflow.log_metric("checkpoint_step", global_step, step=global_step)
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += args.save_model_every_steps

        rollout_steps = min(
            args.num_steps,
            math.ceil(global_step / args.num_envs) - (update - 1) * args.num_steps,
        )
        if rollout_steps <= 0:
            break
        with torch.no_grad():
            next_reward_value, next_cost_value = agent.get_values(next_observation)
        reward_deltas = torch.zeros_like(rewards[:rollout_steps])
        cost_deltas = torch.zeros_like(costs[:rollout_steps])
        for step in reversed(range(rollout_steps)):
            nonterminal = 1.0 - episode_ends[step]
            if step == rollout_steps - 1:
                reward_value_next = next_reward_value
                cost_value_next = next_cost_value
            else:
                reward_value_next = reward_values[step + 1]
                cost_value_next = cost_values[step + 1]
            reward_deltas[step] = (
                rewards[step] + args.gamma * reward_value_next * nonterminal - reward_values[step]
            )
            cost_deltas[step] = costs[step] + cost_value_next * nonterminal - cost_values[step]
        reward_advantages = generalized_advantages(
            reward_deltas,
            episode_ends[:rollout_steps],
            args.gamma * args.gae_lambda,
        )
        cost_advantages = generalized_advantages(
            cost_deltas,
            episode_ends[:rollout_steps],
            args.cost_gae_lambda,
        )
        reward_returns = reward_advantages + reward_values[:rollout_steps]
        cost_returns = cost_advantages + cost_values[:rollout_steps]

        flat_observations = observations[:rollout_steps].reshape((-1,) + observation_shape)
        flat_actions = actions[:rollout_steps].reshape((-1,) + action_shape)
        flat_log_probabilities = log_probabilities[:rollout_steps].reshape(-1)
        flat_reward_advantages = reward_advantages.reshape(-1)
        flat_cost_advantages = cost_advantages.reshape(-1)
        flat_reward_returns = reward_returns.reshape(-1)
        flat_cost_returns = cost_returns.reshape(-1)
        flat_combined_advantages = (
            normalized_lagrangian_advantage(
                flat_reward_advantages,
                flat_cost_advantages,
                lambda_value,
            )
            if args.normalize_advantage_across_rollout
            else None
        )
        indices = np.arange(len(flat_observations))
        current_minibatch_size = min(minibatch_size, len(indices))

        clip_fractions: list[float] = []
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), current_minibatch_size):
                mb_indices = indices[start : start + current_minibatch_size]
                _, _, new_log_probability, entropy, new_reward_value, new_cost_value = (
                    agent.get_action_and_value(
                        flat_observations[mb_indices],
                        flat_actions[mb_indices],
                        estimate_entropy=args.ent_coef != 0.0,
                    )
                )
                log_ratio = new_log_probability - flat_log_probabilities[mb_indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )
                combined_advantage = (
                    flat_combined_advantages[mb_indices]
                    if flat_combined_advantages is not None
                    else normalized_lagrangian_advantage(
                        flat_reward_advantages[mb_indices],
                        flat_cost_advantages[mb_indices],
                        lambda_value,
                    )
                )
                policy_loss_unclipped = -combined_advantage * ratio
                policy_loss_clipped = -combined_advantage * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
                reward_value_loss = (
                    0.5 * (new_reward_value - flat_reward_returns[mb_indices]).pow(2).mean()
                )
                cost_value_loss = (
                    0.5 * (new_cost_value - flat_cost_returns[mb_indices]).pow(2).mean()
                )
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    - args.ent_coef * entropy_loss
                    + args.reward_vf_coef * reward_value_loss
                    + args.cost_vf_coef * cost_value_loss
                )
                optimizer.zero_grad()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite PPO loss")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("non-finite PPO gradient norm")
                optimizer.step()
                if action_std_cap_active:
                    agent.cap_action_std(args.max_action_std_after_first_success)
            if args.target_kl is not None and dual_active and approximate_kl > args.target_kl:
                break

        while len(pending_lambda_costs) >= args.lambda_update_episodes:
            batch_costs = pending_lambda_costs[: args.lambda_update_episodes]
            del pending_lambda_costs[: args.lambda_update_episodes]
            mean_cost = float(np.mean(batch_costs))
            lambda_value = update_lagrange_multiplier(
                lambda_value,
                mean_cost,
                args.cost_limit,
                args.lambda_lr,
                args.lambda_max,
            )
        if (
            dual_active
            and not action_std_cap_active
            and args.max_action_std_after_first_success is not None
        ):
            agent.cap_action_std(args.max_action_std_after_first_success)
            action_std_cap_active = True
        if update == 1 or update % 10 == 0 or update == total_updates:
            recent_cost = (
                float(np.mean(completed_costs[-100:])) if completed_costs else float("nan")
            )
            record = {
                "step": global_step,
                "lambda": lambda_value,
                "dual_active": dual_active,
                "dual_activation_step": dual_activation_step,
                "learning_rate": current_learning_rate,
                "recent_episode_cost": recent_cost,
                "clip_fraction": float(np.mean(clip_fractions)),
                "approximate_kl": float(approximate_kl.item()),
                "policy_loss": float(policy_loss.item()),
                "reward_value_loss": float(reward_value_loss.item()),
                "cost_value_loss": float(cost_value_loss.item()),
                "reward_advantage_std": float(flat_reward_advantages.std().item()),
                "cost_advantage_std": float(flat_cost_advantages.std().item()),
                "cost_advantage_abs_mean": float(flat_cost_advantages.abs().mean().item()),
                "gradient_norm": float(gradient_norm.item()),
                "completed_episodes": len(completed_costs),
                "action_saturation_fraction": float(np.mean(saturation_values)),
                "latent_action_std": float(agent.actor_logstd.exp().mean().item()),
                "approximate_action_std": float(
                    (agent.actor_logstd.exp() * agent.action_scale).mean().item()
                ),
                "sps": int(global_step / max(time.time() - started_at, 1e-9)),
            }
            append_jsonl(metrics_path, record)
            log_mlflow_metrics(record, global_step, "train")
            print(json.dumps(record), flush=True)

    envs.close()
    checkpoint_path = checkpoint_dir / f"ppo_checkpoint_{global_step}_steps.pt"
    save_checkpoint(
        checkpoint_path,
        args,
        agent,
        optimizer,
        lambda_value,
        global_step,
        observation_shape,
        action_shape,
    )
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_metric("checkpoint_step", global_step, step=global_step)
    evaluation = evaluate(agent, args, device, stochastic=True)
    deterministic_evaluation = evaluate(agent, args, device, stochastic=False)
    paired_args = replace(
        args,
        evaluation_episodes=args.paired_evaluation_episodes,
        evaluation_seed=args.paired_evaluation_seed,
    )
    paired_evaluation = evaluate(agent, paired_args, device, stochastic=False)
    if not math.isclose(
        float(evaluation["mean_episode_cost"]),
        1.0 - float(evaluation["goal_reaching_probability"]),
        abs_tol=1e-12,
    ):
        raise AssertionError("episode cost and goal-reaching probability disagree")
    evaluation["lambda"] = lambda_value
    evaluation["dual_active"] = dual_active
    evaluation["dual_activation_step"] = dual_activation_step
    evaluation["cost_limit"] = args.cost_limit
    evaluation["reward_scale"] = args.reward_scale
    evaluation["constraint_satisfied_empirically"] = bool(
        evaluation["mean_episode_cost"] <= args.cost_limit
    )
    evaluation["lambda_upper_bound_reached"] = bool(lambda_value >= args.lambda_max)
    evaluation["global_step"] = global_step
    evaluation["checkpoint_path"] = str(checkpoint_path)
    evaluation["training_metrics_path"] = str(metrics_path)
    evaluation["environment"] = args.environment
    evaluation["env_id"] = args.env_id
    evaluation["horizon"] = args.horizon
    evaluation["training_seed"] = args.seed
    evaluation["evaluation_seed"] = args.evaluation_seed
    evaluation["elapsed_training_seconds"] = time.time() - started_at
    evaluation["completed_training_episodes"] = len(completed_costs)
    evaluation["config"] = serializable_config(args)
    evaluation["runtime"] = source_metadata()
    active_run = mlflow.active_run()
    evaluation["mlflow"] = {
        "tracking_uri": args.mlflow_tracking_uri,
        "experiment_name": args.mlflow_experiment_name,
        "run_name": args.mlflow_run_name,
        "run_id": active_run.info.run_id if active_run is not None else None,
    }
    evaluation["deterministic_evaluation"] = deterministic_evaluation
    evaluation["paired_deterministic_evaluation"] = paired_evaluation
    result_path = args.output_dir / f"ppo_lagrangian_{args.environment}_seed{args.seed}.json"
    result_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    log_mlflow_metrics(evaluation, global_step, "evaluation/stochastic")
    log_mlflow_metrics(deterministic_evaluation, global_step, "evaluation/deterministic")
    log_mlflow_metrics(paired_evaluation, global_step, "evaluation/paired")
    if active_run is not None:
        mlflow.log_artifacts(str(args.output_dir), artifact_path="outputs")
    print(json.dumps(evaluation, indent=2), flush=True)


def run_evaluation_only(args: Args) -> None:
    """Evaluate a saved policy twice without changing the checkpoint."""

    if args.evaluation_checkpoint is None:
        raise ValueError("evaluation_checkpoint is required")
    checkpoint_path = args.evaluation_checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.horizon,
                args.seed,
                index=0,
                env_kwargs=cartpole_env_kwargs(args),
            )
        ]
    )
    try:
        agent = Agent(envs, initial_action_std=args.initial_action_std).to(device)
        payload = load_agent_checkpoint(
            checkpoint_path,
            args,
            agent,
            envs.single_observation_space.shape,
            envs.single_action_space.shape,
            device,
        )
    finally:
        envs.close()
    stochastic = evaluate(agent, args, device, stochastic=True)
    deterministic = evaluate(agent, args, device, stochastic=False)
    result = {
        "evaluation_only": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_lambda": payload.get("lambda"),
        "stochastic_evaluation": stochastic,
        "deterministic_evaluation": deterministic,
        "config": serializable_config(args),
        "runtime": source_metadata(),
    }
    result_path = args.output_dir / "ppo_lagrangian_checkpoint_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if mlflow.active_run() is not None:
        log_mlflow_metrics(stochastic, int(payload.get("global_step", 0)), "evaluation/stochastic")
        log_mlflow_metrics(
            deterministic, int(payload.get("global_step", 0)), "evaluation/deterministic"
        )
        mlflow.log_artifact(str(result_path), artifact_path="outputs")
    print(json.dumps(result, indent=2), flush=True)


def main(args: Args) -> None:
    operation = run_evaluation_only if args.evaluation_checkpoint else run_training
    if args.mlflow_tracking_uri is None:
        operation(args)
        return
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    run_name = args.mlflow_run_name or (f"ppo-lagrangian__{args.environment}__seed-{args.seed}")
    args.mlflow_run_name = run_name
    with mlflow.start_run(run_name=run_name):
        runtime = source_metadata()
        mlflow.set_tags(
            {
                "repro.run_status": "RUNNING",
                "algorithm": "ppo-lagrangian",
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
            operation(args)
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
