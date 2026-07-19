from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from run.train_td3 import ENHANCE_COMMIT, ENHANCE_LOCK_SHA256, enhance_command
from scripts.run_td3_matrix import (
    ensure_experiments,
    parse_seeds,
    start_tmux_sessions,
    training_command,
)


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
    assert command[command.index("--checkpoint-every") + 1] == "30000"


def test_td3_matrix_starts_one_concurrent_tmux_session_per_seed(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["tmux", "has-session"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.run_td3_matrix.subprocess.run", fake_run)
    jobs = [
        ("td3-drone-s0-g0", "python train.py --seed 0", Path("run/logs/s0.log")),
        ("td3-drone-s1-g0", "python train.py --seed 1", Path("run/logs/s1.log")),
        ("td3-drone-s2-g0", "python train.py --seed 2", Path("run/logs/s2.log")),
    ]

    start_tmux_sessions(jobs, Path("/repo"))

    launches = [call for call, _ in calls if call[:2] == ["tmux", "new-session"]]
    assert len(launches) == 3
    assert [launch[launch.index("-s") + 1] for launch in launches] == [
        "td3-drone-s0-g0",
        "td3-drone-s1-g0",
        "td3-drone-s2-g0",
    ]
    assert all(" && " not in launch[-1] for launch in launches)


def test_td3_matrix_precreates_experiments_before_concurrent_workers():
    class FakeClient:
        def __init__(self):
            self.created = []

        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name):
            self.created.append(name)
            return str(len(self.created))

    client = FakeClient()
    experiment_ids = ensure_experiments(
        "http://tracking",
        "nightly",
        ["underwater-drone", "robot-navigation"],
        client=client,
    )

    assert client.created == [
        "nightly/underwater-drone",
        "nightly/robot-navigation",
    ]
    assert experiment_ids == {
        "nightly/underwater-drone": "1",
        "nightly/robot-navigation": "2",
    }
