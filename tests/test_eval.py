import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from run.eval import goal_reaching_mask, run_episode


def test_run_episode_executes_requested_number_of_steps():
    env = make_vec_env("Pendulum-v1", n_envs=2, seed=42)
    data = run_episode(lambda obs: np.zeros((len(obs), 1)), env, n_steps=7)
    assert len(data) == 7
    assert data[-1]["step"] == 6


def test_goal_reaching_mask_is_per_trial():
    observations = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        goal_reaching_mask("Pendulum-v1", observations), [True, False]
    )
