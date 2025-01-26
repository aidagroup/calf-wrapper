import gymnasium as gym
from pathlib import Path

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
