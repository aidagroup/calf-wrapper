from pathlib import Path

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from src.advantage_intervention import (
    AdvantageInterventionWrapper,
    GoalCostCritic,
)
from src.controllers.controller import Controller
from src.intervention_training import GoalCostDataset, fit_goal_cost_critic


class ZeroController(Controller):
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        if observation.ndim == 1:
            return np.zeros(1, dtype=np.float32)
        return np.zeros((len(observation), 1), dtype=np.float32)


class ActionRiskCritic:
    horizon = 10

    def goal_cost(self, observations, actions, remaining_steps):
        del observations, remaining_steps
        return np.maximum(np.asarray(actions)[:, 0], 0.0)


def test_advantage_intervention_compares_base_to_fallback_action():
    env = AdvantageInterventionWrapper(
        make_vec_env("Pendulum-v1", n_envs=2, seed=7),
        goal_cost_critic=ActionRiskCritic(),
        fallback_policy=ZeroController(),
        threshold=0.0,
    )
    env.reset()
    _, _, _, infos = env.step(np.array([[1.0], [-1.0]], dtype=np.float32))

    assert not infos[0]["intervention.base_action_applied"]
    assert infos[1]["intervention.base_action_applied"]
    np.testing.assert_allclose(infos[0]["intervention.action"], [0.0])
    np.testing.assert_allclose(infos[1]["intervention.action"], [-1.0])
    env.close()


def test_goal_cost_critic_checkpoint_round_trip(tmp_path: Path):
    critic = GoalCostCritic(observation_dim=3, action_dim=1, horizon=200)
    observations = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    actions = np.array([[0.0], [1.0]], dtype=np.float32)
    remaining = np.array([200.0, 50.0], dtype=np.float32)
    critic.set_normalization(observations, actions, remaining)
    expected = critic.goal_cost(observations, actions, remaining)
    checkpoint = tmp_path / "critic.pt"
    critic.save(checkpoint, metadata={"environment": "Pendulum-v1"})

    restored, metadata = GoalCostCritic.load(checkpoint)

    np.testing.assert_allclose(
        restored.goal_cost(observations, actions, remaining), expected
    )
    assert metadata["environment"] == "Pendulum-v1"


def test_goal_cost_critic_training_preserves_anchor_pairs_in_split():
    observations = np.array(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.tile(np.array([[1.0], [0.0]], dtype=np.float32), (4, 1))
    dataset = GoalCostDataset(
        observations=observations,
        actions=actions,
        remaining_steps=np.full(8, 10.0, dtype=np.float32),
        goal_costs=np.tile(np.array([0.8, 0.2], dtype=np.float32), 4),
        failures=np.tile(np.array([1.0, 0.0], dtype=np.float32), 4),
        anchor_ids=np.repeat(np.arange(4), 2),
        action_sources=np.tile(np.array(["base", "fallback"]), 4),
    )

    critic, metrics = fit_goal_cost_critic(
        dataset,
        horizon=10,
        hidden_sizes=(16,),
        epochs=20,
        batch_size=8,
        seed=3,
    )

    assert metrics["n_training_anchors"] + metrics["n_validation_anchors"] == 4
    predictions = critic.goal_cost(
        dataset.observations, dataset.actions, dataset.remaining_steps
    )
    assert predictions.shape == (8,)
    assert np.all((0.0 <= predictions) & (predictions <= 1.0))
    assert metrics["paired_outcomes"]["base_cost_worse_rate"] == 1.0
