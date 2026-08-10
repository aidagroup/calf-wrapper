"""Standalone CleanRL-style PPO-Lagrangian trainer and evaluator.

The constrained objective uses a single undiscounted terminal cost:
``1`` iff the episode ends outside the prescribed goal set, and ``0``
otherwise.  Time-to-horizon is appended to every observation so that this
finite-horizon CMDP remains Markov.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.stats import beta
from torch.distributions.normal import Normal

import src  # noqa: F401  # Register the paper's custom environments.
from src import run_path
from src.goal_reaching import goal_reaching_mask

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
    reward_scale: float = 0.01
    gamma: float = 0.98
    gae_lambda: float = 0.95
    cost_gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    reward_vf_coef: float = 0.5
    cost_vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    cost_limit: float = 0.05
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    lambda_update_episodes: int = 20
    lambda_max: float = 10_000.0
    seed: int = 9
    device: str = "cpu"
    torch_deterministic: bool = True
    evaluation_episodes: int = 200
    evaluation_seed: int = 10_000
    paired_evaluation_episodes: int = 30
    paired_evaluation_seed: int = 42
    output_dir: Path = run_path / "artifacts" / "ppo_lagrangian"


PRESETS = {
    "pendulum": (
        "PPO-Lagrangian on Pendulum-v1",
        Args(),
    ),
    "cartpole": (
        "PPO-Lagrangian on the 1000-step CartPole evaluation task",
        Args(
            environment="cartpole",
            env_id="CartpoleSwingupEnvLong-v0",
            horizon=1000,
            total_timesteps=300_000,
            seed=42,
            output_dir=run_path / "artifacts" / "ppo_lagrangian_cartpole",
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


def make_env(env_id: str, horizon: int, seed: int, index: int):
    def thunk():
        env = gym.make(env_id, max_episode_steps=horizon)
        env = TimeAwareObservation(env, horizon)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + index)
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


def normalized_lagrangian_advantage(
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    combined = (reward_advantage - lambda_value * cost_advantage) / (1.0 + lambda_value)
    if len(combined) > 1:
        combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    return combined


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
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
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
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
        return self.action_bias + self.action_scale * torch.tanh(
            self.actor_mean(observation)
        )


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
    if (
        saved_args.get("env_id") != args.env_id
        or saved_args.get("horizon") != args.horizon
    ):
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
            gym.make(args.env_id, max_episode_steps=args.horizon), args.horizon
        )
        observation, _ = env.reset(seed=trial_seed)
        if action_low is None:
            action_low = env.action_space.low
            action_high = env.action_space.high
        episode_return = 0.0
        episode_length = 0
        terminated = truncated = False
        final_info: dict = {}
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(trial_seed)
            while not (terminated or truncated):
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                )
                if stochastic:
                    action = agent.get_action_and_value(tensor.unsqueeze(0))[0][0]
                else:
                    action = agent.deterministic_action(tensor.unsqueeze(0))[0]
                action_array = np.clip(action.cpu().numpy(), action_low, action_high)
                observation, reward, terminated, truncated, final_info = env.step(
                    action_array
                )
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
                "policy_noise_seed": trial_seed if stochastic else None,
                "episode_return": episode_return,
                "episode_cost": episode_cost,
                "goal_reached": not bool(episode_cost),
                "episode_length": episode_length,
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
        "policy": "stochastic" if stochastic else "deterministic_mean",
        "trials": trials,
    }


def main(args: Args) -> None:
    supported_tasks = {
        ("Pendulum-v1", 200),
        ("CartpoleSwingupEnvLong-v0", 1000),
    }
    if (args.env_id, args.horizon) not in supported_tasks:
        raise ValueError(
            "unsupported environment/horizon pair for PPO-Lagrangian: "
            f"{args.env_id}/{args.horizon}"
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
    if args.num_envs <= 0 or args.num_minibatches <= 0 or args.update_epochs <= 0:
        raise ValueError(
            "num_envs, num_minibatches, and update_epochs must be positive"
        )
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if args.reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
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
    metrics_path = (
        args.output_dir / f"ppo_lagrangian_{args.environment}_seed{args.seed}.jsonl"
    )
    metrics_path.write_text("")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.horizon, args.seed, index)
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("PPO-Lagrangian currently supports continuous actions only")
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observations = torch.zeros((args.num_steps, args.num_envs) + observation_shape).to(
        device
    )
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
    next_observation, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(
        next_observation, dtype=torch.float32, device=device
    )
    started_at = time.time()
    total_updates = math.ceil(args.total_timesteps / batch_size)

    while global_step < args.total_timesteps:
        update += 1
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
            cost_np = terminal_costs(
                args.env_id, next_obs_np, terminated, truncated, infos
            )
            rewards[step] = args.reward_scale * torch.as_tensor(
                reward_np, device=device
            )
            costs[step] = torch.as_tensor(cost_np, device=device)
            episode_ends[step] = torch.as_tensor(end_np, device=device)
            for index, ended in enumerate(end_np):
                if ended:
                    value = float(cost_np[index])
                    completed_costs.append(value)
                    pending_lambda_costs.append(value)
            next_observation = torch.as_tensor(
                next_obs_np, dtype=torch.float32, device=device
            )

        rollout_steps = min(
            args.num_steps,
            math.ceil(global_step / args.num_envs) - (update - 1) * args.num_steps,
        )
        if rollout_steps <= 0:
            break
        with torch.no_grad():
            next_reward_value, next_cost_value = agent.get_values(next_observation)
        reward_advantages = torch.zeros_like(rewards[:rollout_steps])
        cost_advantages = torch.zeros_like(costs[:rollout_steps])
        last_reward_gae = torch.zeros(args.num_envs, device=device)
        last_cost_gae = torch.zeros(args.num_envs, device=device)
        for step in reversed(range(rollout_steps)):
            nonterminal = 1.0 - episode_ends[step]
            if step == rollout_steps - 1:
                reward_value_next = next_reward_value
                cost_value_next = next_cost_value
            else:
                reward_value_next = reward_values[step + 1]
                cost_value_next = cost_values[step + 1]
            reward_delta = (
                rewards[step]
                + args.gamma * reward_value_next * nonterminal
                - reward_values[step]
            )
            cost_delta = costs[step] + cost_value_next * nonterminal - cost_values[step]
            last_reward_gae = (
                reward_delta
                + args.gamma * args.gae_lambda * nonterminal * last_reward_gae
            )
            last_cost_gae = (
                cost_delta + args.cost_gae_lambda * nonterminal * last_cost_gae
            )
            reward_advantages[step] = last_reward_gae
            cost_advantages[step] = last_cost_gae
        reward_returns = reward_advantages + reward_values[:rollout_steps]
        cost_returns = cost_advantages + cost_values[:rollout_steps]

        flat_observations = observations[:rollout_steps].reshape(
            (-1,) + observation_shape
        )
        flat_actions = actions[:rollout_steps].reshape((-1,) + action_shape)
        flat_log_probabilities = log_probabilities[:rollout_steps].reshape(-1)
        flat_reward_advantages = reward_advantages.reshape(-1)
        flat_cost_advantages = cost_advantages.reshape(-1)
        flat_reward_returns = reward_returns.reshape(-1)
        flat_cost_returns = cost_returns.reshape(-1)
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
                combined_advantage = normalized_lagrangian_advantage(
                    flat_reward_advantages[mb_indices],
                    flat_cost_advantages[mb_indices],
                    lambda_value,
                )
                policy_loss_unclipped = -combined_advantage * ratio
                policy_loss_clipped = -combined_advantage * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.max(
                    policy_loss_unclipped, policy_loss_clipped
                ).mean()
                reward_value_loss = (
                    0.5
                    * (new_reward_value - flat_reward_returns[mb_indices]).pow(2).mean()
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
                gradient_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("non-finite PPO gradient norm")
                optimizer.step()
            if args.target_kl is not None and approximate_kl > args.target_kl:
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
        if update == 1 or update % 10 == 0 or update == total_updates:
            recent_cost = (
                float(np.mean(completed_costs[-100:]))
                if completed_costs
                else float("nan")
            )
            record = {
                "step": global_step,
                "lambda": lambda_value,
                "recent_episode_cost": recent_cost,
                "clip_fraction": float(np.mean(clip_fractions)),
                "approximate_kl": float(approximate_kl.item()),
                "policy_loss": float(policy_loss.item()),
                "reward_value_loss": float(reward_value_loss.item()),
                "cost_value_loss": float(cost_value_loss.item()),
                "gradient_norm": float(gradient_norm.item()),
                "completed_episodes": len(completed_costs),
                "action_saturation_fraction": float(np.mean(saturation_values)),
                "sps": int(global_step / max(time.time() - started_at, 1e-9)),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)

    envs.close()
    checkpoint_path = (
        args.output_dir / f"ppo_lagrangian_{args.environment}_seed{args.seed}.pt"
    )
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
    evaluation["deterministic_evaluation"] = deterministic_evaluation
    evaluation["paired_deterministic_evaluation"] = paired_evaluation
    result_path = (
        args.output_dir / f"ppo_lagrangian_{args.environment}_seed{args.seed}.json"
    )
    result_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.extras.overridable_config_cli(PRESETS))
