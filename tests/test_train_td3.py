from argparse import Namespace

from calfwrapper.training.td3 import ENHANCE_LOCK_SHA256, enhance_command


def test_td3_command_uses_the_pinned_training_runtime() -> None:
    command = enhance_command(
        Namespace(
            environment="robot-navigation",
            seed=1,
            device="cuda:1",
            total_timesteps=50_000,
            learning_starts=25_000,
            tracking_uri="file:///tmp/mlruns",
            experiment_name="reproduction",
            run_name="robot-seed-1",
            checkpoint_dir="artifacts/checkpoints",
            checkpoint_every=10_000,
        )
    )

    assert ENHANCE_LOCK_SHA256 == (
        "26812bc65b4f091bf16da07e10b7d67c9ae21ccc9d4432704795da6850055f40"
    )
    assert "vendor/calf-enhance-td3" in command[command.index("--project") + 1]
    assert command[command.index("--env-id") + 1] == ("RobotNavigationConstSpeedCatch-v0")
    assert command[command.index("--buffer-size") + 1] == "1000000"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--checkpoint-every") + 1] == "10000"
