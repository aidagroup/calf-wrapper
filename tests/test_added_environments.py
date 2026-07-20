import gymnasium as gym
import numpy as np
import pytest

from src.controllers.robot_navigation import RobotNavigationConstSpeedGoalController
from src.controllers.underwaterdrone import UnderwaterDroneNominalController


@pytest.mark.parametrize(
    ("env_id", "controller"),
    [
        ("UnderwaterDrone-v0", UnderwaterDroneNominalController()),
        (
            "RobotNavigationConstSpeedCatch-v0",
            RobotNavigationConstSpeedGoalController(),
        ),
    ],
)
def test_added_environment_accepts_fallback_action(env_id, controller):
    env = gym.make(env_id)
    observation, _ = env.reset(seed=42)
    action = controller.get_action(observation)

    assert action.shape == env.action_space.shape
    assert action.dtype == np.float32
    assert env.action_space.contains(action)
    assert len(env.step(action)) == 5
    env.close()


def test_added_environment_reset_is_seeded():
    for env_id in ("UnderwaterDrone-v0", "RobotNavigationConstSpeedCatch-v0"):
        env = gym.make(env_id)
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        env.close()


def test_underwater_intrusion_penalty_changes_only_reward():
    common = {"init_x": 0.0, "init_y": 2.0}
    original = gym.make("UnderwaterDrone-v0", high_cost_penalty=5.0, **common)
    stress = gym.make("UnderwaterDrone-v0", high_cost_penalty=50.0, **common)
    original_observation, _ = original.reset(seed=17)
    stress_observation, _ = stress.reset(seed=17)
    np.testing.assert_array_equal(original_observation, stress_observation)

    action = np.zeros(2, dtype=np.float32)
    original_next, original_reward, *original_done = original.step(action)
    stress_next, stress_reward, *stress_done = stress.step(action)

    np.testing.assert_array_equal(original_next, stress_next)
    assert original_done == stress_done
    assert stress_reward == pytest.approx(original_reward - 45.0)
    original.close()
    stress.close()


def test_underwater_intrusion_penalty_must_be_non_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        gym.make("UnderwaterDrone-v0", high_cost_penalty=-1.0)
