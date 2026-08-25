from pathlib import Path

from calfwrapper.cli import training_command


def test_td3_training_command_uses_the_public_trainer(tmp_path: Path) -> None:
    command = training_command("robot-td3", smoke=False, output=tmp_path)

    assert command[1:3] == ["-m", "calfwrapper.training.td3"]
    assert command[3] == "robot"
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--checkpoint-dir") + 1] == str(
        tmp_path / "robot-td3" / "checkpoints"
    )
