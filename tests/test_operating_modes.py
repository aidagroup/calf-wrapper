import pytest

from calfwrapper.operating_modes import (
    OPERATING_MODES,
    fixed_acceptance_budget,
    operating_mode_parameters,
)

EXPECTED_ACCEPTANCE_BUDGETS = {
    "conservative": 0.00,
    "guarded": 0.10,
    "moderate": 0.35,
    "balanced": 0.50,
    "high": 0.70,
    "almost_open": 0.95,
}


@pytest.mark.parametrize("horizon", [200, 1000, 1500])
@pytest.mark.parametrize("operating_mode", OPERATING_MODES)
def test_article_operating_modes_have_the_published_acceptance_budget(
    operating_mode: str,
    horizon: int,
) -> None:
    parameters = operating_mode_parameters(operating_mode, horizon)
    actual = sum(parameters.p_relax * parameters.lambda_**time for time in range(horizon)) / horizon

    assert actual == pytest.approx(
        EXPECTED_ACCEPTANCE_BUDGETS[operating_mode],
        abs=1e-12,
    )


@pytest.mark.parametrize("horizon", [200, 1000, 1500])
def test_fixed_acceptance_budget_preserves_the_requested_budget(horizon: int) -> None:
    parameters = fixed_acceptance_budget(0.5, 0.7, horizon)
    actual = sum(parameters.p_relax * parameters.lambda_**time for time in range(horizon)) / horizon

    assert actual == pytest.approx(0.5, abs=1e-12)
