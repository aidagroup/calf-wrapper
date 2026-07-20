import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env

from run.eval import goal_reaching_mask, run_episode, trial_successes
from src.models.cleanrl_td3 import CleanRLActor, CleanRLQNetwork, CleanRLTD3


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


def test_goal_reaching_mask_supports_added_environments():
    drone_observations = np.array(
        [[0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(
        goal_reaching_mask("UnderwaterDrone-v0", drone_observations), [True, False]
    )

    robot_observations = np.array(
        [
            [0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        goal_reaching_mask("RobotNavigationConstSpeedCatch-v0", robot_observations),
        [True, False],
    )


def test_trial_successes_preserves_terminal_success_after_vector_env_reset():
    data = [
        {
            "obs": np.array(
                [
                    [0.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                ]
            ),
            "info": [{"goal_reached": True}, {"goal_reached": False}],
        }
    ]
    np.testing.assert_array_equal(
        trial_successes("RobotNavigationConstSpeedCatch-v0", data), [True, False]
    )


def test_trial_successes_uses_terminal_observation_instead_of_reset_state():
    reset_observation = np.array([[-1.0, 0.0, 0.0]])
    terminal_observation = np.array([1.0, 0.0, 0.0])
    data = [
        {
            "obs": reset_observation.copy(),
            "next_obs": reset_observation.copy(),
            "active": np.array([True]),
            "is_done": np.array([True]),
            "info": [{"terminal_observation": terminal_observation}],
        }
    ]

    np.testing.assert_array_equal(trial_successes("Pendulum-v1", data), [True])


def test_cleanrl_td3_checkpoint_is_loadable_for_evaluation(tmp_path):
    actor = CleanRLActor(observation_dim=3, action_dim=1)
    actor.action_scale.fill_(2.0)
    qf1 = CleanRLQNetwork(observation_dim=3, action_dim=1)
    qf2 = CleanRLQNetwork(observation_dim=3, action_dim=1)
    checkpoint = tmp_path / "td3.pt"
    torch.save(
        {
            "format": "calf-enhance-cleanrl-td3-v1",
            "completed_steps": 100,
            "env_id": "test-env",
            "seed": 7,
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
        },
        checkpoint,
    )

    model = CleanRLTD3.load(checkpoint)
    observations = np.zeros((2, 3), dtype=np.float32)
    actions = model.predict(observations)[0]
    critic_values = model.critic(
        torch.as_tensor(observations), torch.as_tensor(actions)
    )

    assert actions.shape == (2, 1)
    assert len(critic_values) == 2
    assert critic_values[0].shape == (2, 1)
    assert model.metadata["completed_steps"] == 100
