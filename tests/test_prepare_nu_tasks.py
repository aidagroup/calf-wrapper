from pathlib import Path

from scripts.prepare_nu_tasks import (
    infinity_ablation_tasks,
    sensitivity_tasks,
    threshold_sweep_tasks,
)


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


def test_threshold_sweep_includes_nominal_and_infinity(tmp_path):
    checkpoint_dir = tmp_path / "ppo_Pendulum-v1_3" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in (3000, 6000, 9000):
        (checkpoint_dir / f"ppo_checkpoint_{step}_steps.zip").write_bytes(b"model")
    protocol = {
        "threshold_sweep": {"multipliers": [0.5, 1, "infinity"]},
        "evaluation": {"evaluation_seed": 17, "trials": 30},
        "environments": {
            "pendulum": {
                "env_id": "Pendulum-v1",
                "algorithm": "ppo",
                "checkpoint_directory_glob": "ppo_Pendulum-v1_*",
                "checkpoint_filename_glob": "ppo_checkpoint_*_steps.zip",
                "training_horizon": 9000,
                "nominal_nu": 0.25,
                "training_seed": 3,
                "checkpoint_stages": {"early": 3000, "mid": 6000, "late": 9000},
            }
        },
    }

    tasks = threshold_sweep_tasks(
        protocol, artifacts_root=tmp_path, matrix_id="threshold-sweep"
    )

    assert len(tasks) == 9
    assert {task.calf_mode for task in tasks} == {"conservative"}
    assert {task.calf_change_rate for task in tasks} == {0.125, 0.25, float("inf")}
    assert {task.nu_multiplier for task in tasks} == {0.5, 1.0, float("inf")}
    assert {task.checkpoint_stage for task in tasks} == {"early", "mid", "late"}
