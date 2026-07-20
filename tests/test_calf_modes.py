import pytest

from src.calf_modes import (
    normalized_acceptance_budget,
    resolve_calf_mode,
    solve_relaxprob_factor,
)


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (
            1500,
            {
                "conservative": 0.9933336262,
                "moderate": 0.9982285316,
                "high": 0.9994921192,
                "almost_open": 0.9999309697,
            },
        ),
        (
            1000,
            {
                "conservative": 0.9900004319,
                "moderate": 0.9973425025,
                "high": 0.9992379849,
                "almost_open": 0.9998964211,
            },
        ),
    ],
)
def test_modes_match_auv_budget_equation(horizon, expected):
    for mode, expected_factor in expected.items():
        resolved = resolve_calf_mode(mode, horizon)
        assert resolved.relaxprob_init == 1.0
        assert resolved.relaxprob_factor == pytest.approx(expected_factor, abs=1e-9)
        assert resolved.acceptance_budget_lower_bound == pytest.approx(
            resolved.target_acceptance_budget, abs=1e-10
        )


def test_solver_handles_open_gate_and_rejects_unreachable_budget():
    assert solve_relaxprob_factor(1.0, 200) == 1.0
    assert normalized_acceptance_budget(1.0, 1.0, 200) == 1.0
    with pytest.raises(ValueError, match="unreachable"):
        solve_relaxprob_factor(0.001, 200)
