import pandas as pd

from scripts.analyze_nu_sensitivity import candidate_scores, select_candidate


def synthetic_results():
    rows = []
    for environment, scale in (("a", 1.0), ("b", 100.0)):
        for seed in (1, 2):
            for rule, n, reward, goal in (
                ("goal_guarded_max", 2.0, 1.0, 100.0),
                ("trajectory_scale", 2.0, 2.0, 100.0),
                ("trajectory_scale", 10.0, 3.0, 80.0),
            ):
                rows.append(
                    {
                        "environment": environment,
                        "training_seed": seed,
                        "checkpoint_step": 10,
                        "task_id": f"{environment}-{seed}-{rule}-{n}",
                        "nu_calibration_rule": rule,
                        "nu_calibration_n": n,
                        "mean_reward": reward * scale,
                        "goal_reaching_rate": goal,
                    }
                )
    return pd.DataFrame(rows)


def test_candidate_scores_balance_environments_and_selection_enforces_goal_rate():
    scores = candidate_scores(synthetic_results(), {"a": 100.0, "b": 100.0})
    selected, basis = select_candidate(
        scores,
        minimum_goal_rate=None,
        goal_noninferiority_margin=2.0,
        practical_tie=0.01,
    )

    assert selected["nu_calibration_rule"] == "trajectory_scale"
    assert selected["nu_calibration_n"] == 2.0
    assert "versus fallback" in basis
