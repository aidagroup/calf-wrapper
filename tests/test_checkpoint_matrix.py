import json
from pathlib import Path

from scripts.run_checkpoint_matrix import (
    evaluation_command,
    prepare_tasks,
    read_tasks,
    write_tasks,
)


def protocol():
    return json.loads(Path("experiments/checkpoint-sweep-v1.json").read_text())


def add_checkpoints(root, directory, extension, steps):
    checkpoint_dir = root / directory / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    prefix = "ppo_checkpoint" if extension == "zip" else "td3_checkpoint"
    for step in steps:
        (checkpoint_dir / f"{prefix}_{step}_steps.{extension}").write_bytes(b"model")


def test_prepare_tasks_keeps_all_eligible_checkpoints_and_one_fallback(tmp_path):
    add_checkpoints(tmp_path, "ppo_Pendulum-v1_1", "zip", [3000, 102000, 105000])
    add_checkpoints(tmp_path, "ppo_Pendulum-v1_2", "zip", [3000, 102000])
    tasks = prepare_tasks(
        protocol(),
        tmp_path,
        "matrix",
        ["pendulum"],
        ["fallback", "base", "moderate"],
    )

    assert sum(task.mode == "fallback" for task in tasks) == 1
    learned = [task for task in tasks if task.mode != "fallback"]
    assert len(learned) == 8
    assert {task.checkpoint_step for task in learned} == {3000, 102000}
    assert all(task.n_envs == 30 for task in tasks)


def test_task_csv_roundtrip_and_named_mode_command(tmp_path):
    add_checkpoints(tmp_path, "td3_UnderwaterDrone-v0_2", "pt", [30000, 3000000])
    tasks = prepare_tasks(
        protocol(), tmp_path, "matrix", ["underwater-drone"], ["moderate"]
    )
    task_file = tmp_path / "tasks.csv"
    write_tasks(tasks, task_file)
    loaded = read_tasks(task_file)
    assert loaded == tasks

    command = evaluation_command(
        loaded[0],
        tracking_uri="http://tracking",
        experiment_prefix="sweep",
        device="cuda:1",
        result_path=tmp_path / "result.json",
    )
    assert command[command.index("--calf.mode") + 1] == "moderate"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert "--no-save-episode-data" in command
