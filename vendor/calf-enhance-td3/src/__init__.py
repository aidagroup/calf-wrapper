from pathlib import Path
import gymnasium as gym

from src.envs.robot_navigation import RobotNavigationConfig
from src.envs.robot_navigation_const_speed import RobotNavigationConstSpeedConfig

_DEFAULT_ROBOT_NAV_MAX_SPEED = RobotNavigationConfig().max_speed

REPO_PATH = Path(__file__).parent.parent
SRC_PATH = REPO_PATH / "src"
RUN_PATH = REPO_PATH / "run"

gym.register(
    id="UnderwaterDrone-v0",
    entry_point="src.envs:UnderwaterDroneEnv",
    max_episode_steps=1500,
)

gym.register(
    id="LidarNav-v0",
    entry_point="src.envs:LidarNavEnv",
)

gym.register(
    id="RobotNavigation-v0",
    entry_point="src.envs:RobotNavigationEnv",
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationEmpty-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={"config": RobotNavigationConfig(obstacle_count=0)},
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationSingle-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={"config": RobotNavigationConfig(obstacle_count=1)},
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationMoving-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={
        "config": RobotNavigationConfig(
            obstacle_count=2,
            moving_obstacle_count=2,
            moving_obstacle_radius=0.18,
            moving_obstacle_speed=_DEFAULT_ROBOT_NAV_MAX_SPEED,
        )
    },
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationCatch-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={
        "config": RobotNavigationConfig(
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

gym.register(
    id="RobotDynamics-v0",
    entry_point="src.envs:RobotDynamicsEnv",
    max_episode_steps=1000,
)

gym.register(
    id="RobotNavigationConstSpeed-v0",
    entry_point="src.envs:RobotNavigationConstSpeedEnv",
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationConstSpeedCatch-v0",
    entry_point="src.envs:RobotNavigationConstSpeedEnv",
    kwargs={
        "config": RobotNavigationConstSpeedConfig(
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
