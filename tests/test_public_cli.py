from __future__ import annotations

import csv
import json
from pathlib import Path

from calfwrapper.cli import write_summary, write_trials
from calfwrapper.config import TRAIN_CONFIGURATIONS
from calfwrapper.environments import ENVIRONMENTS
from calfwrapper.evaluation import Trial
from calfwrapper.paths import ROOT
from calfwrapper.verify import verify_trials

TRIAL_FIELDS = (
    "trial",
    "episode_return",
    "goal_reached",
    "episode_length",
    "base_policy_actions",
    "fallback_policy_actions",
    "critic_evaluations",
    "policy_sequence_sha256",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_all_published_train_and_evaluation_configurations_are_registered() -> None:
    assert set(ENVIRONMENTS) == {"pendulum", "cartpole", "auv", "robot"}
    assert len(TRAIN_CONFIGURATIONS) == 8


def test_training_commands_use_the_checkpoint_training_seeds() -> None:
    expected = {
        "pendulum-ppo": ("pendulum", "base"),
        "cartpole-ppo": ("cartpole", "base"),
        "auv-td3": ("auv", "base"),
        "robot-td3": ("robot", "base"),
        "pendulum-ppo-lagrangian": ("pendulum", "lagrangian"),
        "cartpole-ppo-lagrangian": ("cartpole", "lagrangian"),
        "auv-td3-lagrangian": ("auv", "lagrangian"),
        "robot-td3-lagrangian": ("robot", "lagrangian"),
    }
    checkpoints = json.loads((ROOT / "artifacts" / "manifest.json").read_text())["checkpoints"]
    for name, (environment, kind) in expected.items():
        arguments = TRAIN_CONFIGURATIONS[name].arguments
        configured_seed = int(arguments[arguments.index("--seed") + 1])
        checkpoint_seeds = {
            checkpoint["training_seed"]
            for checkpoint in checkpoints
            if checkpoint["environment"] == environment and checkpoint["kind"] == kind
        }
        assert checkpoint_seeds == {configured_seed}


def test_write_trials_uses_article_terms(tmp_path: Path) -> None:
    destination = tmp_path / "trials.csv"
    write_trials(
        [Trial(1.0, True, 2, 1, 1, 2, "abc")],
        destination,
    )
    with destination.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["episode_return"] == "1.0"


def test_write_summary_uses_article_terms(tmp_path: Path) -> None:
    trials = tmp_path / "trials.csv"
    summary = tmp_path / "summary.json"
    _write_csv(
        trials,
        TRIAL_FIELDS,
        [
            {
                "trial": "0",
                "episode_return": "1.5",
                "goal_reached": "True",
                "episode_length": "2",
                "base_policy_actions": "1",
                "fallback_policy_actions": "1",
                "critic_evaluations": "2",
                "policy_sequence_sha256": "abc",
            }
        ],
    )
    write_summary(trials, summary)
    assert json.loads(summary.read_text()) == {
        "trials": 1,
        "mean_episode_return": 1.5,
        "goal_reaching_rate": 100.0,
    }


def test_verify_trials_requires_exact_published_values(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    reference = tmp_path / "reference.csv"
    trial = {
        "trial": "0",
        "episode_return": "1.25",
        "goal_reached": "True",
        "episode_length": "2",
        "base_policy_actions": "1",
        "fallback_policy_actions": "1",
        "critic_evaluations": "2",
        "policy_sequence_sha256": "abc",
    }
    _write_csv(generated / "pendulum-high-guarded.csv", TRIAL_FIELDS, [trial])
    reference_fields = (
        "environment",
        "checkpoint_stage",
        "mode",
        "seed",
        "return",
        "ggr_success",
        "episode_length",
        "base_policy_calls",
        "fallback_calls",
        "critic_calls",
        "selection_sha256",
    )
    reference_row = {
        "environment": "Pendulum-v1",
        "checkpoint_stage": "High",
        "mode": "guarded",
        "seed": "20260801",
        "return": "1.25",
        "ggr_success": "1",
        "episode_length": "2",
        "base_policy_calls": "1",
        "fallback_calls": "1",
        "critic_calls": "2",
        "selection_sha256": "abc",
    }
    _write_csv(reference, reference_fields, [reference_row])

    report = verify_trials(generated, reference, {"pendulum"})
    assert report["status"] == "passed"

    trial["episode_return"] = "1.2500001"
    _write_csv(generated / "pendulum-high-guarded.csv", TRIAL_FIELDS, [trial])
    report = verify_trials(generated, reference, {"pendulum"})
    assert report["status"] == "failed"
    assert report["mismatch_count"] == 1
