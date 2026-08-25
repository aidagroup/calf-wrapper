from __future__ import annotations

import copy
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

import src  # noqa: F401
from calfwrapper.training import ppo_lagrangian as ppo_lag
from calfwrapper.training import td3_lagrangian as td3_lag


@pytest.mark.parametrize(
    "wrapper_class",
    [ppo_lag.TimeAwareObservation, td3_lag.TimeAwareObservation],
)
def test_time_aware_observation_reaches_one_at_horizon(wrapper_class):
    env = wrapper_class(gym.make("Pendulum-v1", max_episode_steps=2), horizon=2)
    observation, _ = env.reset(seed=3)
    assert observation[-1] == pytest.approx(0.0)
    observation, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert not truncated
    assert observation[-1] == pytest.approx(0.5)
    observation, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert truncated
    assert observation[-1] == pytest.approx(1.0)
    env.close()


@pytest.mark.parametrize("module", [ppo_lag, td3_lag])
def test_terminal_cost_uses_final_observation_before_autoreset(module):
    autoreset_observation = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    successful_final_observation = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    infos = {
        "final_observation": successful_final_observation,
        "final_info": np.array([{}], dtype=object),
    }
    result = module.terminal_costs(
        "Pendulum-v1",
        autoreset_observation,
        np.array([False]),
        np.array([True]),
        infos,
    )
    costs = result[0] if isinstance(result, tuple) else result
    assert costs.tolist() == [0.0]
    if isinstance(result, tuple):
        np.testing.assert_allclose(result[1], successful_final_observation)


@pytest.mark.parametrize("module", [ppo_lag, td3_lag])
def test_failed_truncation_has_unit_cost(module):
    failed = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    infos = {
        "final_observation": failed,
        "final_info": np.array([{}], dtype=object),
    }
    result = module.terminal_costs(
        "Pendulum-v1",
        failed,
        np.array([False]),
        np.array([True]),
        infos,
    )
    costs = result[0] if isinstance(result, tuple) else result
    assert costs.tolist() == [1.0]


@pytest.mark.parametrize("module", [ppo_lag, td3_lag])
def test_lagrange_multiplier_moves_in_constraint_direction(module):
    increased = module.update_lagrange_multiplier(0.2, 1.0, 0.05, 0.1, 10.0)
    decreased = module.update_lagrange_multiplier(0.2, 0.0, 0.05, 0.1, 10.0)
    projected = module.update_lagrange_multiplier(0.0, 0.0, 0.05, 0.1, 10.0)
    assert increased > 0.2
    assert decreased < 0.2
    assert projected == 0.0


def test_ppo_sparse_cost_warm_start_activates_on_first_success():
    pending: list[float] = []
    active = False
    for _ in range(5):
        active = ppo_lag.record_episode_cost_for_dual(1.0, active, pending, True)
    assert not active
    assert pending == []
    active = ppo_lag.record_episode_cost_for_dual(0.0, active, pending, True)
    assert active
    assert pending == [0.0]
    active = ppo_lag.record_episode_cost_for_dual(1.0, active, pending, True)
    assert pending == [0.0, 1.0]


@pytest.mark.parametrize("module", [ppo_lag, td3_lag])
def test_exact_failure_upper_bound_is_stricter_than_point_estimate(module):
    upper = module.clopper_pearson_failure_upper(0, 30)
    assert upper > 0.05
    assert module.clopper_pearson_failure_upper(30, 30) == 1.0


def test_td3_targets_do_not_bootstrap_through_episode_end():
    reward = torch.tensor([2.0, 2.0])
    cost = torch.tensor([1.0, 0.0])
    done = torch.tensor([1.0, 0.0])
    next_value = torch.tensor([9.0, 9.0])
    reward_target = td3_lag.reward_bellman_target(
        reward, done, next_value, gamma=0.9, reward_scale=0.5
    )
    cost_target = td3_lag.cost_bellman_target(cost, done, next_value)
    assert reward_target[0].item() == pytest.approx(1.0)
    assert reward_target[1].item() == pytest.approx(9.1)
    assert cost_target[0].item() == pytest.approx(1.0)
    assert cost_target[1].item() == pytest.approx(9.0)


def test_ppo_squashed_policy_is_bounded_and_log_probability_reproducible():
    envs = gym.vector.SyncVectorEnv([ppo_lag.make_env("Pendulum-v1", horizon=200, seed=4, index=0)])
    agent = ppo_lag.Agent(envs)
    observation, _ = envs.reset(seed=4)
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    action, latent, log_probability, _, _, _ = agent.get_action_and_value(observation_tensor)
    repeated_action, _, repeated_log_probability, _, _, _ = agent.get_action_and_value(
        observation_tensor, latent
    )
    assert torch.all(action <= agent.action_bias + agent.action_scale)
    assert torch.all(action >= agent.action_bias - agent.action_scale)
    assert torch.isfinite(log_probability).all()
    torch.testing.assert_close(action, repeated_action)
    torch.testing.assert_close(log_probability, repeated_log_probability)
    envs.close()


def test_ppo_initial_action_std_is_expressed_in_environment_units():
    envs = gym.vector.SyncVectorEnv(
        [ppo_lag.make_env("CartpoleSwingupEnvLong-v0", horizon=1000, seed=44, index=0)]
    )
    agent = ppo_lag.Agent(envs, initial_action_std=1.0)
    torch.testing.assert_close(
        agent.actor_logstd.exp() * agent.action_scale,
        torch.ones_like(agent.action_scale).reshape(1, -1),
    )
    envs.close()


def test_ppo_action_std_cap_is_expressed_in_environment_units():
    envs = gym.vector.SyncVectorEnv(
        [ppo_lag.make_env("CartpoleSwingupEnvLong-v0", horizon=1000, seed=44, index=0)]
    )
    agent = ppo_lag.Agent(envs, initial_action_std=10.0)
    agent.cap_action_std(2.0)
    torch.testing.assert_close(
        agent.actor_logstd.exp() * agent.action_scale,
        2.0 * torch.ones_like(agent.action_scale).reshape(1, -1),
    )
    agent.cap_action_std(5.0)
    torch.testing.assert_close(
        agent.actor_logstd.exp() * agent.action_scale,
        2.0 * torch.ones_like(agent.action_scale).reshape(1, -1),
    )
    envs.close()


def test_actor_state_round_trip_preserves_deterministic_actions():
    envs = gym.vector.SyncVectorEnv([td3_lag.make_env("UnderwaterDrone-v0", horizon=1500, seed=2)])
    observation_size = int(np.prod(envs.single_observation_space.shape))
    actor = td3_lag.Actor(observation_size, envs.single_action_space)
    restored = td3_lag.Actor(observation_size, envs.single_action_space)
    restored.load_state_dict(copy.deepcopy(actor.state_dict()))
    observation, _ = envs.reset(seed=2)
    tensor = torch.as_tensor(observation, dtype=torch.float32)
    torch.testing.assert_close(actor(tensor), restored(tensor))
    envs.close()


def test_auv_constructor_seed_controls_later_unseeded_resets():
    first = td3_lag.make_env("UnderwaterDrone-v0", horizon=1500, seed=17)()
    second = td3_lag.make_env("UnderwaterDrone-v0", horizon=1500, seed=17)()
    first.reset(seed=17)
    second.reset(seed=17)
    first_later, _ = first.reset()
    second_later, _ = second.reset()
    np.testing.assert_allclose(first_later, second_later)
    first.close()
    second.close()


def test_ppo_stochastic_evaluation_preserves_model_mode_parameters_and_rng():
    envs = gym.vector.SyncVectorEnv([ppo_lag.make_env("Pendulum-v1", horizon=200, seed=4, index=0)])
    agent = ppo_lag.Agent(envs)
    agent.eval()
    before_parameters = copy.deepcopy(agent.state_dict())
    before_rng = torch.get_rng_state().clone()
    args = ppo_lag.Args(evaluation_episodes=1, evaluation_seed=42)
    result = ppo_lag.evaluate(agent, args, torch.device("cpu"), stochastic=True)
    assert not agent.training
    torch.testing.assert_close(torch.get_rng_state(), before_rng)
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, before_parameters[name])
    assert result["evaluation_episodes"] == 1
    assert not result["constraint_assessment_eligible"]
    assert len(result["trials"]) == 1
    envs.close()


def test_ppo_checkpoint_disk_round_trip_and_compatibility_rejection(tmp_path):
    envs = gym.vector.SyncVectorEnv([ppo_lag.make_env("Pendulum-v1", horizon=200, seed=5, index=0)])
    source = ppo_lag.Agent(envs)
    restored = ppo_lag.Agent(envs)
    optimizer = torch.optim.Adam(source.parameters())
    path = tmp_path / "ppo.pt"
    args = ppo_lag.Args()
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    ppo_lag.save_checkpoint(
        path,
        args,
        source,
        optimizer,
        0.4,
        123,
        observation_shape,
        action_shape,
    )
    payload = ppo_lag.load_agent_checkpoint(
        path,
        args,
        restored,
        observation_shape,
        action_shape,
        torch.device("cpu"),
    )
    assert payload["global_step"] == 123
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="environment or horizon"):
        ppo_lag.load_agent_checkpoint(
            path,
            ppo_lag.Args(horizon=201),
            restored,
            observation_shape,
            action_shape,
            torch.device("cpu"),
        )
    envs.close()


def test_td3_checkpoint_disk_round_trip(tmp_path):
    envs = gym.vector.SyncVectorEnv([td3_lag.make_env("UnderwaterDrone-v0", horizon=1500, seed=6)])
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observation_size = int(np.prod(observation_shape))
    action_size = int(np.prod(action_shape))
    actor = td3_lag.Actor(observation_size, envs.single_action_space)
    target_actor = copy.deepcopy(actor)
    reward_q1 = td3_lag.QNetwork(observation_size, action_size)
    reward_q2 = td3_lag.QNetwork(observation_size, action_size)
    target_reward_q1 = copy.deepcopy(reward_q1)
    target_reward_q2 = copy.deepcopy(reward_q2)
    cost_q = td3_lag.CostQNetwork(observation_size, action_size)
    target_cost_q = copy.deepcopy(cost_q)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    reward_q_optimizer = torch.optim.Adam(
        list(reward_q1.parameters()) + list(reward_q2.parameters())
    )
    cost_q_optimizer = torch.optim.Adam(cost_q.parameters())
    replay = td3_lag.ReplayBuffer(4, observation_shape, action_shape, torch.device("cpu"))
    path = tmp_path / "td3.pt"
    args = td3_lag.Args()
    td3_lag.save_checkpoint(
        path,
        args,
        321,
        0.7,
        actor,
        target_actor,
        reward_q1,
        reward_q2,
        target_reward_q1,
        target_reward_q2,
        cost_q,
        target_cost_q,
        actor_optimizer,
        reward_q_optimizer,
        cost_q_optimizer,
        replay,
    )
    restored = td3_lag.Actor(observation_size, envs.single_action_space)
    payload = td3_lag.load_actor_checkpoint(
        path,
        args,
        restored,
        observation_shape,
        action_shape,
        torch.device("cpu"),
    )
    assert payload["global_step"] == 321
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    envs.close()


def test_ppo_primal_advantage_penalizes_the_higher_cost_action():
    reward_advantage = torch.tensor([0.0, 0.0])
    cost_advantage = torch.tensor([0.0, 1.0])
    combined = ppo_lag.normalized_lagrangian_advantage(
        reward_advantage, cost_advantage, lambda_value=1.0
    )
    assert combined[0] > combined[1]


def test_undiscounted_cost_trace_propagates_terminal_failure_over_long_horizon():
    deltas = torch.zeros((1000, 1))
    deltas[-1] = 1.0
    episode_ends = torch.zeros_like(deltas)
    episode_ends[-1] = 1.0
    monte_carlo = ppo_lag.generalized_advantages(deltas, episode_ends, 1.0)
    truncated_trace = ppo_lag.generalized_advantages(deltas, episode_ends, 0.95)
    assert monte_carlo[0].item() == pytest.approx(1.0)
    assert truncated_trace[-101].item() == pytest.approx(0.95**100, rel=2e-6)
    assert truncated_trace[0].item() < 1e-20


def test_cartpole_saturated_preset_uses_nonterminating_saturated_environment():
    args = ppo_lag.PRESETS["cartpole-saturated-600k"][1]
    kwargs = ppo_lag.cartpole_env_kwargs(args)
    assert kwargs["terminate_on_out_of_bounds"] is False
    assert kwargs["saturate_state_on_out_of_bounds"] is True
    assert kwargs["position_termination_threshold"] == pytest.approx(7.5)
    assert kwargs["velocity_termination_threshold"] == pytest.approx(12.0)
    assert kwargs["angular_velocity_termination_threshold"] == pytest.approx(15.0)
    assert kwargs["reward_position_clip"] == pytest.approx(5.0)


@pytest.mark.parametrize("terminal_cost", [0.0, 1.0])
def test_cartpole_cost_redistribution_preserves_binary_episode_sum(terminal_cost):
    args = dataclasses.replace(ppo_lag.PRESETS["cartpole"][1], redistribute_terminal_cost=True)
    initial_observation = np.array([[0.4, -0.2, -1.0, 0.0, 0.3, 0.0]])
    current = ppo_lag.cartpole_cost_potential(initial_observation)
    initial = current.copy()
    total = 0.0
    for step in range(10):
        angle = np.pi * (1.0 - (step + 1) / 10.0)
        successor = np.array([[0.4 - 0.04 * step, 0.0, np.cos(angle), np.sin(angle), 0.0, 0.1]])
        ended = np.array([step == 9])
        infos = {"final_observation": np.array([successor[0]], dtype=object)}
        costs, current, initial = ppo_lag.transition_costs(
            args,
            successor,
            ended,
            infos,
            np.array([terminal_cost if ended[0] else 0.0], dtype=np.float32),
            current,
            initial,
        )
        total += float(costs[0])
    assert total == pytest.approx(terminal_cost, abs=1e-6)


def test_combined_advantage_normalization_is_rollout_global():
    rewards = torch.tensor([3.0, 1.0, -2.0, 0.5])
    costs = torch.tensor([0.0, 1.0, 0.2, 0.8])
    combined = ppo_lag.normalized_lagrangian_advantage(rewards, costs, 0.7)
    first_partition = combined[torch.tensor([0, 2])]
    second_partition = combined[torch.tensor([1, 3])]
    reconstructed = torch.empty_like(combined)
    reconstructed[torch.tensor([0, 2])] = first_partition
    reconstructed[torch.tensor([1, 3])] = second_partition
    torch.testing.assert_close(reconstructed, combined)


def test_cartpole_learning_rate_anneals_only_after_discovery_phase():
    schedule = ppo_lag.delayed_linear_learning_rate
    assert schedule(1e-3, 1, 100, 0.5) == pytest.approx(1e-3)
    assert schedule(1e-3, 51, 100, 0.5) == pytest.approx(1e-3)
    assert schedule(1e-3, 100, 100, 0.5) == pytest.approx(2e-5)


def test_feasible_and_violating_batches_drive_projected_dual_response():
    value = 0.2
    for _ in range(100):
        value = td3_lag.update_lagrange_multiplier(value, 0.0, 0.05, 0.1, 1.0)
    assert value == 0.0
    for _ in range(10):
        value = td3_lag.update_lagrange_multiplier(value, 1.0, 0.05, 0.1, 1.0)
    assert value > 0.0


def test_td3_initial_state_dual_estimate_uses_current_actor_and_cost_critic():
    class ScalarActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action = torch.nn.Parameter(torch.tensor(0.8))

        def forward(self, observations):
            return self.action.expand(len(observations), 1)

    class ActionCost(torch.nn.Module):
        def forward(self, observations, actions):
            return actions

    actor = ScalarActor()
    cost_q = ActionCost()
    initial_observations = torch.zeros((4, 1))
    raw, clipped = td3_lag.estimate_initial_failure_probability(actor, cost_q, initial_observations)
    assert raw == pytest.approx(0.8)
    assert clipped == pytest.approx(0.8)
    with torch.no_grad():
        actor.action.fill_(0.2)
    _, changed = td3_lag.estimate_initial_failure_probability(actor, cost_q, initial_observations)
    assert changed == pytest.approx(0.2)


def test_td3_cost_critic_is_a_bounded_failure_probability():
    critic = td3_lag.CostQNetwork(observation_size=3, action_size=2)
    observations = torch.randn(128, 3) * 100.0
    actions = torch.randn(128, 2) * 100.0
    probabilities = critic(observations, actions)
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_td3_cost_probability_loss_accepts_soft_targets_and_rejects_invalid_ones():
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    targets = torch.tensor([0.0, 0.4, 1.0])
    loss = td3_lag.cost_probability_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(FloatingPointError, match="probability interval"):
        td3_lag.cost_probability_loss(logits.detach(), torch.tensor([0.0, 1.1, 1.0]))


def test_td3_bounded_target_network_keeps_cost_bellman_target_in_range():
    critic = td3_lag.CostQNetwork(observation_size=3, action_size=2)
    next_probabilities = critic(torch.randn(16, 3), torch.randn(16, 2)).squeeze(-1)
    costs = torch.zeros(16)
    dones = torch.zeros(16)
    costs[0] = 1.0
    dones[0] = 1.0
    targets = td3_lag.cost_bellman_target(costs, dones, next_probabilities)
    td3_lag.require_probability("test targets", targets)


def test_td3_primal_step_reduces_cost_when_reward_is_flat():
    action = torch.nn.Parameter(torch.tensor(0.8))
    optimizer = torch.optim.SGD([action], lr=0.1)
    reward_value = torch.zeros(())
    cost_value = action.square()
    loss = td3_lag.actor_lagrangian_loss(reward_value, cost_value, lambda_value=1.0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert action.item() < 0.8
