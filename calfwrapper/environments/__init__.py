"""Custom environments and article environment configurations."""

import gymnasium as gym

from calfwrapper.environments.auv import ContaminatedZoneAUVEnv
from calfwrapper.environments.robot import (
    TreasureCollectingRobotConfig,
    TreasureCollectingRobotEnv,
)

if "CALFWrapper/CartPoleSwingUp-v0" not in gym.registry:
    gym.register(
        id="CALFWrapper/CartPoleSwingUp-v0",
        entry_point="calfwrapper.environments.cartpole:CartPoleSwingUpEnv",
        max_episode_steps=200,
    )
if "CALFWrapper/CartPoleSwingUpLong-v0" not in gym.registry:
    gym.register(
        id="CALFWrapper/CartPoleSwingUpLong-v0",
        entry_point="calfwrapper.environments.cartpole:CartPoleSwingUpEnv",
        max_episode_steps=1000,
    )
if "CALFWrapper/ContaminatedZoneAUV-v0" not in gym.registry:
    gym.register(
        id="CALFWrapper/ContaminatedZoneAUV-v0",
        entry_point="calfwrapper.environments:ContaminatedZoneAUVEnv",
        max_episode_steps=1500,
    )
if "CALFWrapper/TreasureCollectingRobot-v0" not in gym.registry:
    gym.register(
        id="CALFWrapper/TreasureCollectingRobot-v0",
        entry_point="calfwrapper.environments:TreasureCollectingRobotEnv",
        kwargs={
            "config": TreasureCollectingRobotConfig(
                max_steps=1000,
                obstacle_count=1,
                target_count=1,
                moving_obstacle_count=1,
                collect_targets=True,
                target_radius=0.05,
                target_reward=50.0,
                target_step_penalty=0.001,
                moving_obstacle_radius=0.025,
                moving_obstacle_speed=0.12,
            )
        },
        max_episode_steps=1000,
    )

__all__ = [
    "TreasureCollectingRobotConfig",
    "TreasureCollectingRobotEnv",
    "ContaminatedZoneAUVEnv",
]
