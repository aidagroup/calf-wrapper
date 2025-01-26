import numpy as np
import gymnasium as gym
import pytest
from stable_baselines3.common.env_util import make_vec_env
from src.controllers.pendulum import EnergyBasedStabilizingPolicy


@pytest.fixture
def policy():
    return EnergyBasedStabilizingPolicy(
        gain=1.0,
        action_min=-2,
        action_max=2,
        switch_loc=np.cos(np.pi / 10),
        switch_vel_loc=0.2,
        pd_coeffs=[12, 4],
    )


def test_single_env_stabilization(policy):
    env = gym.make("Pendulum-v1")
    observation, info = env.reset(seed=42)

    # Run for sufficient steps to allow stabilization
    for _ in range(199):
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

    # Check if pendulum is stabilized upright
    # observation = [cos(theta), sin(theta), theta_dot]
    target = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(observation, target, atol=0.001)

    env.close()


def test_vectorized_env_stabilization(policy):
    n_envs = 10
    env = make_vec_env(env_id="Pendulum-v1", n_envs=n_envs, seed=42)
    obs = env.reset()

    # Run for sufficient steps to allow stabilization
    for _ in range(199):
        action = policy.get_action(obs)
        obs, rewards, dones, info = env.step(action)

    # Check if all pendulums are stabilized upright
    target = np.array([1.0, 0.0, 0.0])
    for single_obs in obs:
        np.testing.assert_allclose(single_obs, target, atol=0.001)

    env.close()
