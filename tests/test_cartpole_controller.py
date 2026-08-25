import numpy as np
import gymnasium as gym
import pytest
from src.controllers.cartpole import CartpoleEnergyBasedStabilizingPolicy


@pytest.fixture
def policy():
    return CartpoleEnergyBasedStabilizingPolicy(
        env_id="CartpoleSwingupEnvLong-v0",
        pd_coefs=[77.76, 8.07, 20.72, 11.18],
        gain=2.5,
        gain_pos_vel=0.6,
        gain_pos=0.8,
        swing_position_reference_gain=4.10,
        switch_loc=0.82,
        blend_width=0.05,
        velocity_brake_threshold=5.0,
        velocity_brake_position_threshold=1.0,
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


@pytest.mark.parametrize("base_seed", [42, 20260801])
def test_initial_state_batch_stabilization_without_termination(policy, base_seed):
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    tolerance = np.array([0.3, 0.3, 0.05, 0.05, 0.05])

    for seed in range(base_seed, base_seed + 30):
        env = gym.make("CartpoleSwingupEnvLong-v0", terminate_on_out_of_bounds=False)
        observation, _ = env.reset(seed=seed)
        for step in range(1000):
            action = policy.get_action(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated or step == 999
        assert np.all(np.abs(observation - target) < tolerance)
        env.close()


def test_reward_position_clip_does_not_clip_cart_state():
    env = gym.make(
        "CartpoleSwingupEnvLong-v0",
        terminate_on_out_of_bounds=False,
        reward_position_clip=5.0,
    )
    env.reset(seed=42)
    env.unwrapped.state = (100.0, 0.0, 0.0, 0.0)

    observation, reward, terminated, truncated, _ = env.step(np.array([0.0]))

    assert observation[0] == pytest.approx(100.0)
    assert reward == pytest.approx(-12.5)
    assert not terminated
    assert not truncated
    env.close()


def test_nonterminating_state_saturation_clips_and_brakes_at_bounds():
    env = gym.make(
        "CartpoleSwingupEnvLong-v0",
        terminate_on_out_of_bounds=False,
        saturate_state_on_out_of_bounds=True,
        position_termination_threshold=7.5,
        velocity_termination_threshold=12.0,
        angular_velocity_termination_threshold=15.0,
    )
    env.reset(seed=42)
    env.unwrapped.state = (8.0, 20.0, 0.0, 20.0)

    observation, _, terminated, truncated, _ = env.step(np.array([10.0]))

    assert observation[0] == pytest.approx(7.5)
    assert observation[1] == pytest.approx(0.0)
    assert abs(observation[4]) <= 15.0
    assert not terminated
    assert not truncated
    env.close()


@pytest.mark.parametrize(
    ("state", "terminated"),
    [
        ((7.49, 0.0, 0.0, 0.0), False),
        ((7.51, 0.0, 0.0, 0.0), True),
        ((0.0, 12.01, 0.0, 0.0), True),
        ((0.0, 0.0, 0.0, 15.01), True),
    ],
)
def test_configurable_cartpole_termination_thresholds(state, terminated):
    env = gym.make(
        "CartpoleSwingupEnvLong-v0",
        position_termination_threshold=7.5,
        velocity_termination_threshold=12.0,
        angular_velocity_termination_threshold=15.0,
    )
    env.reset(seed=42)
    env.unwrapped.state = state

    _, _, actual_terminated, _, _ = env.step(np.array([0.0]))

    assert actual_terminated is terminated
    env.close()


@pytest.mark.parametrize(
    ("state", "expected_action"),
    [
        ((1.01, 5.01), -10.0),
        ((-1.01, -5.01), 10.0),
        ((0.99, 5.01), None),
    ],
)
def test_high_cart_velocity_has_braking_priority(policy, state, expected_action):
    position, velocity = state
    observation = np.array([position, velocity, 0.0, 1.0, 0.0])

    action = policy.get_action(observation)

    if expected_action is None:
        assert action.item() != pytest.approx(-10.0)
    else:
        assert action.item() == pytest.approx(expected_action)
