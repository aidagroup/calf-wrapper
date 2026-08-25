"""Published training and evaluation configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfiguration:
    name: str
    module: str
    arguments: tuple[str, ...]
    total_timesteps: int
    smoke_arguments: tuple[str, ...]
    description: str


TRAIN_CONFIGURATIONS: dict[str, TrainConfiguration] = {
    "pendulum-ppo": TrainConfiguration(
        name="pendulum-ppo",
        module="calfwrapper.training.ppo",
        arguments=("pendulum", "--seed", "6"),
        total_timesteps=300_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--n-steps",
            "64",
            "--save-model-every-steps",
            "64",
        ),
        description="PPO base policy for Pendulum",
    ),
    "cartpole-ppo": TrainConfiguration(
        name="cartpole-ppo",
        module="calfwrapper.training.ppo",
        arguments=(
            "cartpole-long-saturated-600k-annealed",
            "--seed",
            "45",
        ),
        total_timesteps=600_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--n-steps",
            "64",
            "--save-model-every-steps",
            "64",
        ),
        description="PPO base policy for CartPole Swing-Up",
    ),
    "auv-td3": TrainConfiguration(
        name="auv-td3",
        module="calfwrapper.training.td3",
        arguments=("auv", "--seed", "0"),
        total_timesteps=3_000_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--learning-starts",
            "32",
            "--checkpoint-every",
            "64",
        ),
        description="TD3 base policy for Contaminated-Zone AUV Navigation",
    ),
    "robot-td3": TrainConfiguration(
        name="robot-td3",
        module="calfwrapper.training.td3",
        arguments=("robot", "--seed", "2"),
        total_timesteps=3_000_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--learning-starts",
            "32",
            "--checkpoint-every",
            "64",
        ),
        description="TD3 base policy for Treasure-Collecting Robot",
    ),
    "pendulum-ppo-lagrangian": TrainConfiguration(
        name="pendulum-ppo-lagrangian",
        module="calfwrapper.training.ppo_lagrangian",
        arguments=("pendulum", "--seed", "10"),
        total_timesteps=300_000,
        smoke_arguments=(
            "--total-timesteps",
            "200",
            "--num-steps",
            "200",
            "--num-minibatches",
            "4",
            "--update-epochs",
            "1",
            "--evaluation-episodes",
            "2",
            "--paired-evaluation-episodes",
            "2",
            "--save-model-every-steps",
            "200",
        ),
        description="PPO-Lagrangian baseline for Pendulum",
    ),
    "cartpole-ppo-lagrangian": TrainConfiguration(
        name="cartpole-ppo-lagrangian",
        module="calfwrapper.training.ppo_lagrangian",
        arguments=(
            "cartpole-saturated-600k",
            "--seed",
            "42",
        ),
        total_timesteps=600_000,
        smoke_arguments=(
            "--total-timesteps",
            "1000",
            "--num-steps",
            "1000",
            "--num-minibatches",
            "4",
            "--update-epochs",
            "1",
            "--evaluation-episodes",
            "2",
            "--paired-evaluation-episodes",
            "2",
            "--save-model-every-steps",
            "1000",
        ),
        description="PPO-Lagrangian baseline for CartPole Swing-Up",
    ),
    "auv-td3-lagrangian": TrainConfiguration(
        name="auv-td3-lagrangian",
        module="calfwrapper.training.td3_lagrangian",
        arguments=("underwater-drone", "--seed", "4"),
        total_timesteps=3_000_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--learning-starts",
            "32",
            "--buffer-size",
            "256",
            "--batch-size",
            "16",
            "--evaluation-episodes",
            "2",
            "--paired-evaluation-episodes",
            "2",
            "--checkpoint-every",
            "64",
            "--log-every",
            "32",
        ),
        description="TD3-Lagrangian baseline for AUV Navigation",
    ),
    "robot-td3-lagrangian": TrainConfiguration(
        name="robot-td3-lagrangian",
        module="calfwrapper.training.td3_lagrangian",
        arguments=("robot-navigation", "--seed", "1"),
        total_timesteps=3_000_000,
        smoke_arguments=(
            "--total-timesteps",
            "64",
            "--learning-starts",
            "32",
            "--buffer-size",
            "256",
            "--batch-size",
            "16",
            "--evaluation-episodes",
            "2",
            "--paired-evaluation-episodes",
            "2",
            "--checkpoint-every",
            "64",
            "--log-every",
            "32",
        ),
        description="TD3-Lagrangian baseline for Treasure-Collecting Robot",
    ),
}
