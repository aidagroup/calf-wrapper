from __future__ import annotations

import numpy as np
import pytest
import torch

import src  # noqa: F401
from run.train_sooper import controller_for
from scripts.run_sooper_matrix import command, shuffled_shard
from scripts.prepare_sooper_screening import shortlist
from scripts.launch_sooper_workers import parse_assignment
from scripts.select_sooper_scenario import aggregate, identify
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


def test_matrix_shards_are_deterministic_disjoint_and_portable(tmp_path):
    tasks = [
        {
            "environment": "pendulum",
            "algorithm": "ppo",
            "model_path": f"checkpoints/model-{index}.zip",
            "seed": index,
            "tracking_uri": "file:///tmp/mlruns",
            "experiment_name": "test",
        }
        for index in range(12)
    ]
    shards = [shuffled_shard(tasks, index, 3, 91) for index in range(3)]
    assert sum(map(len, shards)) == len(tasks)
    assert {task["seed"] for shard in shards for task in shard} == set(range(12))
    assert shards == [shuffled_shard(tasks, index, 3, 91) for index in range(3)]
    launch = command(tasks[0], tmp_path / "out", "cuda:0", tmp_path)
    assert str((tmp_path / tasks[0]["model_path"]).resolve()) in launch


def test_screening_shortlist_uses_only_complete_feasible_development_groups():
    def row(mode, reward, goal):
        return {
            "mode": mode,
            "mean_reward": str(reward),
            "goal_reaching_rate": str(goal),
            "task_id": f"task-{mode}",
            "mlflow_run_id": f"run-{mode}",
        }

    complete = {
        "base": row("base", -100.0, 50.0),
        "conservative": row("conservative", -20.0, 100.0),
        "moderate": row("moderate", -10.0, 90.0),
        "high": row("high", -50.0, 100.0),
        "almost_open": row("almost_open", -80.0, 60.0),
    }
    incomplete = dict(complete)
    incomplete.pop("high")
    protocol = {
        "primary_environment": "UnderwaterDrone-v0",
        "checkpoint_shortlist": {
            "required_checkpoint_methods": list(complete),
            "base_goal_reaching_rate_max": 0.8,
            "calf_goal_reaching_rate_min": 0.95,
            "retain": 1,
        },
    }
    selected = shortlist(
        {
            ("cleanrl_td3", "UnderwaterDrone-v0", 0, 30): complete,
            ("cleanrl_td3", "UnderwaterDrone-v0", 1, 60): incomplete,
        },
        protocol,
    )
    assert len(selected) == 1
    assert selected[0]["selected_calf_mode"] == "conservative"
    assert selected[0]["calf_reward_gain"] == pytest.approx(80.0)


def test_worker_assignment_preserves_cuda_device_colon():
    assert parse_assignment("2=cuda:1") == (2, "cuda:1")


def test_sooper_screening_aggregation_uses_canonical_checkpoint_identity():
    import pandas as pd

    assert identify(
        "/mnt/raid0/calf-eval-wrapper/run/artifacts/td3_UnderwaterDrone-v0_2/"
        "checkpoints/td3_checkpoint_120000_steps.pt"
    ) == (2, 120000)
    trials = pd.DataFrame(
        [
            {
                "checkpoint_training_seed": 2,
                "checkpoint_step": 120000,
                "online_iterations": 5,
                "sooper_training_seed": seed,
                "episode_return": -100.0 + seed,
                "goal_reached": True,
                "constraint_satisfied": True,
                "intervention_fraction": 0.25,
                "cost_budget": 7.0,
                "real_interactions": 7500,
                "offline_prior_interactions": 7500,
                "wall_clock_seconds": 60.0,
            }
            for seed in (1, 2, 3)
        ]
    )
    summary = aggregate(trials).iloc[0]
    assert summary.n_trials == 3
    assert summary.mean_reward == pytest.approx(-98.0)
    assert summary.cost_budget == pytest.approx(7.0)
    assert summary.gpu_hours_per_training_seed == pytest.approx(1 / 60)


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
