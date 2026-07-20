from __future__ import annotations

import numpy as np
import pytest
import torch

import src  # noqa: F401
from run.train_sooper import controller_for
from src.models.cleanrl_td3 import CleanRLActor, CleanRLTwinCritic
from src.sooper import (
    PriorValueEnsemble,
    ProbabilisticEnsemble,
    SOOPERSafetyFilter,
    TD3Planner,
    cost_definition,
)


@pytest.mark.parametrize(
    ("env_id", "observation", "expected"),
    [
        ("Pendulum-v1", np.array([1.0, 0.0, 7.0]), 1.0),
        ("CartpoleSwingupEnvLong-v0", np.array([2.5, 0.0, 1.0, 0.0, 0.0]), 1.0),
        ("UnderwaterDrone-v0", np.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 1.0),
        (
            "RobotNavigationConstSpeedCatch-v0",
            np.array([0.1, 0.1, 1.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.05]),
            1.0,
        ),
    ],
)
def test_cost_definitions_detect_constraint_violation(env_id, observation, expected):
    assert cost_definition(env_id).observation_cost(observation)[0] == expected


class ZeroPrior:
    def get_action(self, observation):
        observation = np.asarray(observation)
        return (
            np.zeros((1,), dtype=np.float32)
            if observation.ndim == 1
            else np.zeros((len(observation), 1), dtype=np.float32)
        )


def constant_model(observation_dim=3, action_dim=1, cost=0.25):
    torch.manual_seed(7)
    model = ProbabilisticEnsemble(
        observation_dim, action_dim, ensemble_size=2, hidden=8, device="cpu"
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.target_mean[-1] = cost
    model.is_fitted = True
    model.eval()
    return model


def test_online_cost_tracking_and_prior_switch():
    observation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    proposed = np.array([0.75], dtype=np.float32)
    model = constant_model(cost=0.25)
    filter_ = SOOPERSafetyFilter(
        model,
        ZeroPrior(),
        cost_definition("Pendulum-v1"),
        budget=0.9,
        gamma=0.9,
        pessimism_beta=0.0,
        prior_horizon=2,
    )
    accepted = filter_.decide(observation, proposed)
    assert not accepted.intervention
    np.testing.assert_allclose(accepted.action, proposed)
    filter_.observe_cost(0.6)
    rejected = filter_.decide(observation, proposed)
    assert rejected.intervention
    np.testing.assert_allclose(rejected.action, [0.0])
    assert filter_.accumulated_cost == pytest.approx(0.6)


def test_world_model_fit_is_seed_reproducible():
    rng = np.random.default_rng(4)
    observations = rng.normal(size=(32, 3)).astype(np.float32)
    actions = rng.normal(size=(32, 1)).astype(np.float32)
    next_observations = observations + 0.1 * actions
    rewards = observations[:, 0]
    costs = (observations[:, 1] > 0).astype(np.float32)
    states = []
    for _ in range(2):
        torch.manual_seed(19)
        model = ProbabilisticEnsemble(3, 1, ensemble_size=2, hidden=8)
        model.fit(
            observations,
            actions,
            next_observations,
            rewards,
            costs,
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            seed=23,
        )
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def test_learned_prior_value_heads_drive_constant_time_filter():
    torch.manual_seed(13)
    values = PriorValueEnsemble(3, 1, ensemble_size=2, hidden=8, device="cpu")
    for parameter in values.parameters():
        parameter.data.zero_()
    values.is_fitted = True
    model = constant_model(cost=0.0)
    filter_ = SOOPERSafetyFilter(
        model,
        ZeroPrior(),
        cost_definition("Pendulum-v1"),
        prior_value_model=values,
        budget=1.0,
        pessimism_beta=0.0,
        prior_horizon=100,
    )
    decision = filter_.decide(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    # softplus(0) * cost_scale is above the budget; direct Q evaluation should
    # therefore intervene independently of the unused rollout horizon.
    assert decision.intervention


def test_cleanrl_td3_initialization_is_exact():
    torch.manual_seed(29)
    actor = CleanRLActor(3, 1)
    critic = CleanRLTwinCritic(3, 1)
    actor.action_scale.fill_(2.0)
    planner = TD3Planner.create(
        3,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        "cpu",
    )
    planner.initialize_from_cleanrl(actor, critic)
    observations = torch.randn(7, 3)
    actions = torch.randn(7, 1).clamp(-2, 2)
    torch.testing.assert_close(planner.actor(observations), actor(observations))
    expected_q1, expected_q2 = critic(observations, actions)
    actual_q1, actual_q2 = planner.critic(observations, actions)
    torch.testing.assert_close(actual_q1, expected_q1)
    torch.testing.assert_close(actual_q2, expected_q2)


@pytest.mark.parametrize(
    ("environment", "env_id"),
    [
        ("pendulum", "Pendulum-v1"),
        ("cartpole", "CartpoleSwingupEnvLong-v0"),
        ("underwater-drone", "UnderwaterDrone-v0"),
        ("robot-navigation", "RobotNavigationConstSpeedCatch-v0"),
    ],
)
def test_sooper_filter_integrates_with_each_environment(environment, env_id):
    import gymnasium as gym

    env = gym.make(env_id)
    observation, _ = env.reset(seed=31)
    prior = controller_for(environment)
    action = np.asarray(prior.get_action(observation), dtype=np.float32).reshape(-1)
    model = constant_model(len(observation), len(action), cost=0.0)
    filter_ = SOOPERSafetyFilter(
        model,
        prior,
        cost_definition(env_id),
        budget=1e6,
        pessimism_beta=0.0,
        prior_horizon=1,
        observation_low=env.observation_space.low,
        observation_high=env.observation_space.high,
    )
    decision = filter_.decide(observation, action)
    next_observation, _, _, _, info = env.step(decision.action)
    cost = cost_definition(env_id).transition_cost(info, next_observation)
    assert cost in (0.0, 1.0)
    env.close()
