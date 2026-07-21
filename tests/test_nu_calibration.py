import numpy as np
import pytest

from src.nu_calibration import calibrate_nu


def test_nu_is_maximum_of_goal_variation_and_scaled_fallback_range():
    result = calibrate_nu(
        current_values=np.array([0.0, 2.0, 5.0]),
        next_values=np.array([2.0, 5.0, 5.5]),
        current_goal_mask=np.array([False, False, True]),
        next_goal_mask=np.array([False, True, True]),
        horizon=10,
        n=2,
    )

    assert result.goal_local_variation == 3.0
    assert result.fallback_value_range == 5.5
    assert result.range_increment == pytest.approx(0.275)
    assert result.nu == 3.0
    assert result.goal_neighborhood_transitions == 1


def test_nu_uses_range_term_when_it_is_larger():
    result = calibrate_nu(
        current_values=np.array([-10.0, 0.0]),
        next_values=np.array([0.0, 0.1]),
        current_goal_mask=np.array([False, False]),
        next_goal_mask=np.array([False, True]),
        horizon=2,
        n=1,
    )

    assert result.goal_local_variation == pytest.approx(0.1)
    assert result.range_increment == pytest.approx(5.05)
    assert result.nu == pytest.approx(5.05)


def test_nu_requires_goal_entering_fallback_transition():
    with pytest.raises(ValueError, match="no transition entering the goal"):
        calibrate_nu(
            current_values=np.array([0.0]),
            next_values=np.array([1.0]),
            current_goal_mask=np.array([False]),
            next_goal_mask=np.array([False]),
            horizon=10,
            n=2,
        )
