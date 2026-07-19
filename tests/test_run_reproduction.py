from pathlib import Path

from scripts.run_reproduction import commands_for_environment


def test_training_device_is_explicit_in_training_command():
    commands = commands_for_environment(
        env_name="pendulum",
        tracking_uri="file:///tmp/mlruns",
        artifact_root=Path("artifacts/test"),
        experiment_prefix="test",
        training_device="cuda:0",
        smoke=True,
        skip_training=False,
    )

    training_command = commands[0]
    device_index = training_command.index("--device")
    assert training_command[device_index + 1] == "cuda:0"
