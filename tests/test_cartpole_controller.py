import numpy as np
import gymnasium as gym
import pytest
from stable_baselines3.common.env_util import make_vec_env
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy


@pytest.fixture
def policy():
    return CartpoleEnergyBasedStabilizingPolicy(
        env_id="CartpoleSwingupEnvLong-v0",
        pd_coefs=[70, 10.0, 20.0, 10.0],
        gain=0.5,
        gain_pos_vel=0.5,
        action_min=-10.0,
        action_max=10.0,
    )


def test_single_env_stabilization(policy):
    env = gym.make("CartpoleSwingupEnvLong-v0")
    observation, info = env.reset(seed=42)

    # Run for sufficient steps to allow stabilization
    for _ in range(999):
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

    # Check if cartpole is stabilized upright
    # observation = [x, x_dot, cos(theta), sin(theta), theta_dot]
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(observation, target, atol=0.001)

    env.close()


def test_vectorized_env_stabilization(policy):
    n_envs = 10
    env = make_vec_env(env_id="CartpoleSwingupEnvLong-v0", n_envs=n_envs, seed=42)
    obs = env.reset()

    # Run for sufficient steps to allow stabilization
    for _ in range(999):
        action = policy.get_action(obs)
        obs, rewards, dones, info = env.step(action)

    # Check if all cartpoles are stabilized upright
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    for single_obs in obs:
        np.testing.assert_allclose(single_obs, target, atol=0.001)

    env.close()
