import gymnasium as gym
from pathlib import Path

from src.envs.robot_navigation import RobotNavigationConfig
from src.envs.robot_navigation_const_speed import RobotNavigationConstSpeedConfig

repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
run_path = repo_root / "run"

gym.register(
    id="CartpoleSwingupEnv-v0",
    entry_point="src.envs.cartpole:CartPoleSwingupEnv",
    max_episode_steps=200,
)

gym.register(
    id="CartpoleSwingupEnvLong-v0",
    entry_point="src.envs.cartpole:CartPoleSwingupEnv",
    max_episode_steps=1000,
)

gym.register(
    id="UnderwaterDrone-v0",
    entry_point="src.envs:UnderwaterDroneEnv",
    max_episode_steps=1500,
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
