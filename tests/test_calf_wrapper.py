from typing import Any

import numpy as np

from calfwrapper.wrapper import CALFWrapper


class FakeEnvironment:
    num_envs = 2

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        return np.array([[1.0], [2.0]])

    def step(self, actions: np.ndarray) -> tuple[Any, ...]:
        self.actions.append(np.copy(actions))
        states = np.array([[2.0], [3.0]])
        return states, np.zeros(2), np.zeros(2, dtype=bool), [{}, {}]

    def close(self) -> None:
        pass


class FakeFallbackPolicy:
    def get_action(self, states: np.ndarray) -> np.ndarray:
        return np.full_like(states, -1.0)


class FakeBasePolicy:
    pass


def make_wrapper(
    monkeypatch,
    *,
    values: np.ndarray,
    nu: float = 1.0,
    p_relax: float = 0.0,
    lambda_: float = 0.0,
    fallback_only=lambda states: np.zeros(len(states), dtype=bool),
    critic_upper_bound: float = np.inf,
    critic_transform=None,
) -> CALFWrapper:
    monkeypatch.setattr(
        "calfwrapper.wrapper.critic_values",
        lambda base_policy, states: np.copy(values),
    )
    return CALFWrapper(
        FakeEnvironment(),
        FakeBasePolicy(),
        FakeFallbackPolicy(),
        nu=nu,
        p_relax=p_relax,
        lambda_=lambda_,
        seed=42,
        critic_upper_bound=critic_upper_bound,
        fallback_only=fallback_only,
        critic_transform=critic_transform,
    )


def test_fallback_only_states_disable_both_base_policy_conditions(monkeypatch) -> None:
    environment = make_wrapper(
        monkeypatch,
        values=np.array([1.0, 2.0]),
        p_relax=1.0,
        lambda_=1.0,
        fallback_only=lambda states: np.array([True, False]),
    )
    environment.reset()
    _, _, _, information = environment.step(np.array([[10.0], [20.0]]))

    np.testing.assert_allclose(environment.environment.actions[-1], [[-1.0], [20.0]])
    assert information[0]["calfwrapper.fallback_only"]
    assert not information[0]["calfwrapper.base_policy_selected"]
    assert information[1]["calfwrapper.random_acceptance_condition"]


def test_critic_condition_updates_the_reference_value(monkeypatch) -> None:
    values = np.array([1.0, 2.0])
    environment = make_wrapper(monkeypatch, values=values, nu=0.5)
    environment.reset()
    values[:] = [1.5, 2.4]
    _, _, _, information = environment.step(np.array([[10.0], [20.0]]))

    np.testing.assert_allclose(environment.reference_value[:, 0], [1.5, 2.0])
    np.testing.assert_allclose(environment.environment.actions[-1], [[10.0], [-1.0]])
    assert information[0]["calfwrapper.critic_condition"]
    assert not information[1]["calfwrapper.critic_condition"]


def test_random_acceptance_probability_decays_after_each_step(monkeypatch) -> None:
    environment = make_wrapper(
        monkeypatch,
        values=np.array([1.0, 2.0]),
        p_relax=1.0,
        lambda_=0.5,
    )
    environment.reset()
    environment.step(np.array([[10.0], [20.0]]))

    assert environment.p_relax == 0.5


def test_critic_transform_is_applied_before_upper_clipping(monkeypatch) -> None:
    environment = make_wrapper(
        monkeypatch,
        values=np.array([5.0, -2.0]),
        critic_upper_bound=3.0,
        critic_transform=lambda values: values + 1.0,
    )

    np.testing.assert_allclose(
        environment.critic_value(np.zeros((2, 1)))[:, 0],
        [3.0, -1.0],
    )
