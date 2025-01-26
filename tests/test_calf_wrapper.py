import numpy as np
import gymnasium as gym
import pytest
from stable_baselines3.common.env_util import make_vec_env
from src.calf_wrapper import CALFWrapper
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy
from stable_baselines3 import PPO


@pytest.fixture
def base_policy():
    model = PPO("MlpPolicy", "CartpoleSwingupEnvLong-v0", device="cpu", seed=42)
    return model


@pytest.fixture
def stabilizing_policy():
    return CartpoleEnergyBasedStabilizingPolicy(
        env_id="CartpoleSwingupEnvLong-v0",
        pd_coefs=[70, 10.0, 20.0, 10.0],
        gain=0.5,
        gain_pos_vel=0.5,
        action_min=-10.0,
        action_max=10.0,
    )


def test_single_env_relaxprob_comparison(base_policy, stabilizing_policy):
    # Create two environments with different relax probabilities
    env = gym.make("CartpoleSwingupEnvLong-v0")
    vec_env = make_vec_env("CartpoleSwingupEnvLong-v0", n_envs=1, seed=42)

    # Wrap environments with CALF
    wrapped_env = CALFWrapper(
        env,
        model=base_policy,
        stabilizing_policy=stabilizing_policy,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed=42,
    )
    obs = wrapped_env.reset(seed=42)
    vec_wrapped_env = CALFWrapper(
        vec_env,
        model=base_policy,
        stabilizing_policy=stabilizing_policy,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed=42,
    )
    vec_obs = vec_wrapped_env.reset()
    assert np.allclose(obs, vec_obs[0])
    # Run both environments for several steps
    decay_happended_n = 0
    base_action_applied_n = 0
    for _ in range(999):
        action = base_policy.predict(obs, deterministic=False)[
            0
        ]  # Same action for both

        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        vec_next_obs, vec_reward, vec_is_done, vec_info = vec_wrapped_env.step(
            action.reshape(1, -1)
        )

        assert np.allclose(next_obs, vec_next_obs[0])
        assert (
            info["calf.base_action_applied"] == vec_info[0]["calf.base_action_applied"]
        )
        assert info["calf.relaxprob"] == vec_info[0]["calf.relaxprob"]
        assert info["calf.decay_happened"] == vec_info[0]["calf.decay_happened"]
        if info["calf.decay_happened"]:
            decay_happended_n += 1
        if info["calf.base_action_applied"]:
            base_action_applied_n += 1
        obs = next_obs

    assert decay_happended_n > 0
    assert base_action_applied_n > 0


def test_calf_wrapper_vec_env(base_policy, stabilizing_policy):
    env = CALFWrapper(
        make_vec_env("CartpoleSwingupEnvLong-v0", n_envs=10, seed=42),
        model=base_policy,
        stabilizing_policy=stabilizing_policy,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed=42,
    )
    obs = env.reset()
    values, best_values = [], []
    for _ in range(999):
        action = base_policy.predict(obs, deterministic=False)[
            0
        ]  # Same action for both
        values.append(env.value(obs))
        alternative_action = stabilizing_policy.get_action(obs)
        next_obs, reward, is_done, infos = env.step(action)

        for i, info in enumerate(infos):
            if info["calf.base_action_applied"]:
                assert np.allclose(action[i, :], info["calf.action"])
            else:
                assert np.allclose(alternative_action[i, :], info["calf.action"])
        best_values.append(np.copy(env.best_value))
        obs = next_obs

    values = np.hstack(values)
    cur_best_value = np.copy(values[:, 0])
    for i in range(values.shape[1]):
        cur_best_value = np.where(
            values[:, i] - cur_best_value >= env.calf_change_rate,
            values[:, i],
            cur_best_value,
        )
        assert np.allclose(cur_best_value, best_values[i].reshape(-1))
