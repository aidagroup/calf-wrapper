import gymnasium as gym
import numpy as np
import pytest

from calfwrapper.fallback.auv import AUVFallbackPolicy
from calfwrapper.fallback.robot import RobotFallbackPolicy


@pytest.mark.parametrize(
    ("env_id", "controller"),
    [
        ("CALFWrapper/ContaminatedZoneAUV-v0", AUVFallbackPolicy()),
        (
            "CALFWrapper/TreasureCollectingRobot-v0",
            RobotFallbackPolicy(),
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
    for env_id in ("CALFWrapper/ContaminatedZoneAUV-v0", "CALFWrapper/TreasureCollectingRobot-v0"):
        env = gym.make(env_id)
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        env.close()
