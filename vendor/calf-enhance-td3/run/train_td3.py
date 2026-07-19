# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from src.utils.mlflow import mlflow_monitoring, MlflowConfig, log_json_artifact
from src.utils.robot_nav_logging import log_robot_nav_trajectory
from src import RUN_PATH
import stable_baselines3 as sb3
import mlflow
from collections import defaultdict, deque
from src.config import config
from src.envs.robot_dynamics import RobotDynamicsMetricsCollector
from src.envs.robot_navigation import RobotNavigationMetricsCollector
from src.envs.underwaterdrone import UnderwaterDroneMetricsCollector
from src.utils.metrics_controller import MetricsCollector

SOURCE_COMMIT = "afb5edc49427054c99d6fbfe87b603d126724eb8"


@dataclass
class Args:
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            experiment_name=os.path.basename(__file__)[: -len(".py")],
        )
    )
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    rolling_average_window: int = 20
    """the rolling average window for the metrics"""
    checkpoint_dir: Path = RUN_PATH / "artifacts" / "td3_checkpoints"
    """directory for periodic and final training checkpoints"""
    checkpoint_every: int = 300_000
    """checkpoint interval in completed environment steps"""

    def __post_init__(self):
        default_experiment_name = os.path.basename(__file__)[: -len(".py")]
        auto_experiment_name = default_experiment_name + "__" + self.env_id

        # Respect CLI override: only auto-generate if user didn't change it.
        if not self.mlflow.experiment_name or self.mlflow.experiment_name == default_experiment_name:
            self.mlflow.experiment_name = auto_experiment_name

        # Respect CLI override: only auto-generate if user didn't set it.
        if not self.mlflow.run_name:
            timestamp = int(time.time())
            if "__" + self.env_id in self.mlflow.experiment_name:
                self.mlflow.run_name = (
                    self.mlflow.experiment_name + "__" + str(self.seed) + "__" + str(timestamp)
                )
            else:
                self.mlflow.run_name = (
                    self.mlflow.experiment_name
                    + "__"
                    + self.env_id
                    + "__"
                    + str(self.seed)
                    + "__"
                    + str(timestamp)
                )


def create_metrics_collector(env_id: str, rolling_window_size: int = 20):
    if env_id.startswith("RobotDynamics"):
        return RobotDynamicsMetricsCollector(rolling_window_size)
    if env_id.startswith("RobotNavigation"):
        return RobotNavigationMetricsCollector(rolling_window_size)
    if env_id.startswith("UnderwaterDrone"):
        return UnderwaterDroneMetricsCollector(rolling_window_size)
    return MetricsCollector(rolling_window_size)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", seed=seed)
            env = gym.wrappers.RecordVideo(
                env,
                f"{RUN_PATH}/videos/{run_name}",
                episode_trigger=lambda e: e % 5 == 0,
            )
        else:
            env = gym.make(env_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def save_checkpoint(
    *,
    args: Args,
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
    """Atomically save the complete trainable TD3 state without sampling RNGs."""

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.checkpoint_dir / f"td3_checkpoint_{completed_steps}_steps.pt"
    )
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    payload = {
        "format": "calf-enhance-cleanrl-td3-v1",
        "source_commit": SOURCE_COMMIT,
        "completed_steps": completed_steps,
        "env_id": args.env_id,
        "seed": args.seed,
        "algorithm": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "policy_noise": args.policy_noise,
            "exploration_noise": args.exploration_noise,
            "learning_starts": args.learning_starts,
            "policy_frequency": args.policy_frequency,
            "noise_clip": args.noise_clip,
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
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": payload["format"],
                "source_commit": SOURCE_COMMIT,
                "completed_steps": completed_steps,
                "env_id": args.env_id,
                "seed": args.seed,
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    mlflow.log_artifact(str(metadata_path), artifact_path="checkpoints")
    print(f"checkpoint saved: {checkpoint_path}")
    return checkpoint_path


@mlflow_monitoring()
def main(args: Args):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    mlflow.set_tag("calf_enhance_source_commit", SOURCE_COMMIT)
    mlflow.set_tag("videos_path", f"{RUN_PATH}/videos/{args.mlflow.run_name}")

    device = torch.device(args.device)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                args.mlflow.run_name,
            )
            for i in range(args.num_envs)
        ]
    )
    metrics_collector = create_metrics_collector(
        args.env_id, args.rolling_average_window
    )
    # print(type(metrics_collector))

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rolling_window = defaultdict(lambda: deque(maxlen=args.rolling_average_window))

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    episode_trajectory = []
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            np.array(actions, dtype=float)
        )
        episode_trajectory.append(
            {
                "obs": obs.copy(),
                "actions": actions.copy(),
                "reward": np.array(rewards).copy(),
            }
        )
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metrics_collector.collect_metrics_from_final_episode_info(
                        info, global_step
                    )

                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )

                    metrics_collector.log_pending_metrics(synchronous=True)
                    log_json_artifact(
                        episode_trajectory,
                        f"trajectories",
                        json_name=f"{global_step:010d}.json",
                    )
                    if args.env_id.startswith("RobotNavigation"):
                        log_robot_nav_trajectory(
                            episode_trajectory,
                            global_step,
                            total_reward=float(info["episode"]["r"]),
                            goal_reached=bool(info.get("goal_reached", False)),
                        )
                    episode_trajectory = []
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                metrics_collector.append_metric(
                    "losses/qf1_values",
                    qf1_a_values.mean().item(),
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "losses/qf2_values",
                    qf2_a_values.mean().item(),
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "losses/qf1_loss",
                    qf1_loss.item(),
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "losses/qf2_loss",
                    qf2_loss.item(),
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "losses/qf_loss",
                    qf_loss.item() / 2.0,
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "losses/actor_loss",
                    actor_loss.item(),
                    step=global_step,
                )
                metrics_collector.append_metric(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    step=global_step,
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                metrics_collector.log_pending_metrics(synchronous=True)

        completed_steps = global_step + 1
        if completed_steps % args.checkpoint_every == 0:
            save_checkpoint(
                args=args,
                completed_steps=completed_steps,
                actor=actor,
                target_actor=target_actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                observation=obs,
                replay_buffer=rb,
            )

    if args.total_timesteps % args.checkpoint_every:
        save_checkpoint(
            args=args,
            completed_steps=args.total_timesteps,
            actor=actor,
            target_actor=target_actor,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            actor_optimizer=actor_optimizer,
            q_optimizer=q_optimizer,
            observation=obs,
            replay_buffer=rb,
        )

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.td3_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=(Actor, QNetwork),
    #         device=device,
    #         exploration_noise=args.exploration_noise,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         mlflow.log_metric("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(
    #             args,
    #             episodic_returns,
    #             repo_id,
    #             "TD3",
    #             f"runs/{run_name}",
    #             f"videos/{run_name}-eval",
    #         )

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
