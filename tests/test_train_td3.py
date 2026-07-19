from argparse import Namespace

from run.train_td3 import ENHANCE_COMMIT, ENHANCE_LOCK_SHA256, enhance_command
from scripts.run_td3_matrix import parse_seeds, training_command


def test_td3_command_uses_pinned_enhance_runtime_and_hyperparameters():
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

    assert ENHANCE_COMMIT == "afb5edc49427054c99d6fbfe87b603d126724eb8"
    assert "vendor/calf-enhance-td3" in command[command.index("--project") + 1]
    assert ENHANCE_LOCK_SHA256 == (
        "26812bc65b4f091bf16da07e10b7d67c9ae21ccc9d4432704795da6850055f40"
    )
    assert command[command.index("--env-id") + 1] == (
        "RobotNavigationConstSpeedCatch-v0"
    )
    assert command[command.index("--buffer-size") + 1] == "1000000"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--checkpoint-every") + 1] == "10000"


def test_td3_matrix_defaults_and_explicit_cuda_device():
    assert parse_seeds(None, "underwater-drone") == list(range(10))
    assert parse_seeds(None, "robot-navigation") == list(range(1, 11))
    command = training_command(
        environment="underwater-drone",
        seed=3,
        gpu=1,
        tracking_uri="file:///tmp/mlruns",
        experiment_prefix="test",
        smoke=True,
    )
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--total-timesteps") + 1] == "1000"
    assert command[command.index("--learning-starts") + 1] == "25000"
