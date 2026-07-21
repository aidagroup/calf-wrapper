from pathlib import Path

from scripts.prepare_nu_tasks import infinity_ablation_tasks, sensitivity_tasks


def calibration_row(n=2.0, nu=0.25):
    return {
        "environment": "Pendulum-v1",
        "preset": "pendulum",
        "algorithm": "ppo",
        "training_seed": 3,
        "checkpoint_step": 9000,
        "checkpoint_path": str(Path("checkpoint.zip")),
        "n": n,
        "nu": nu,
        "rule_variant": "goal_guarded_max",
    }


def test_sensitivity_tasks_keep_calibrated_threshold_and_n():
    tasks = sensitivity_tasks(
        [calibration_row()],
        matrix_id="sensitivity",
        evaluation_seed=17,
        trials=30,
    )

    assert len(tasks) == 1
    assert tasks[0].calf_mode == "conservative"
    assert tasks[0].calf_change_rate == 0.25
    assert tasks[0].nu_calibration_n == 2.0
    assert tasks[0].nu_calibration_rule == "goal_guarded_max"
    assert tasks[0].evaluation_seed == 17


def test_infinity_ablation_has_paired_finite_and_disabled_gate_tasks():
    tasks = infinity_ablation_tasks(
        [calibration_row()],
        matrix_id="ablation",
        preset="pendulum",
        training_seed=3,
        checkpoint_step=9000,
        selected_n=2.0,
        selected_rule="goal_guarded_max",
        evaluation_seed=17,
        trials=30,
    )

    assert len(tasks) == 8
    wrapper_tasks = [task for task in tasks if task.eval_mode == "calf_wrapper"]
    assert {task.calf_mode for task in wrapper_tasks} == {
        "conservative",
        "guarded",
        "moderate",
    }
    assert sum(task.calf_change_rate == 0.25 for task in wrapper_tasks) == 3
    assert sum(task.calf_change_rate == float("inf") for task in wrapper_tasks) == 3
