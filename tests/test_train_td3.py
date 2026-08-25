from pathlib import Path

import gymnasium as gym

from calfwrapper.cli import training_command
from calfwrapper.training import td3  # noqa: F401


def test_td3_training_command_uses_the_public_trainer(tmp_path: Path) -> None:
    command = training_command("robot-td3", smoke=False, output=tmp_path)

    assert command[1:3] == ["-m", "calfwrapper.training.td3"]
    assert command[3] == "robot"
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--checkpoint-dir") + 1] == str(
        tmp_path / "robot-td3" / "checkpoints"
    )


def test_td3_trainer_registers_article_environments() -> None:
    assert gym.spec("CALFWrapper/ContaminatedZoneAUV-v0") is not None
    assert gym.spec("CALFWrapper/TreasureCollectingRobot-v0") is not None
