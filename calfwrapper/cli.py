"""Command-line entry point for training and evaluating CALF-Wrapper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from calfwrapper._protocol import evaluation_batches
from calfwrapper.baselines import evaluate_lagrangian
from calfwrapper.config import TRAIN_CONFIGURATIONS
from calfwrapper.environments import ENVIRONMENTS
from calfwrapper.evaluation import Policy, Trial, evaluate
from calfwrapper.figures import main_results
from calfwrapper.operating_modes import OPERATING_MODES
from calfwrapper.paths import OUTPUTS, REFERENCE_TRIALS, ROOT
from calfwrapper.results import summarize
from calfwrapper.results import write_summary as write_results
from calfwrapper.statistics import paired_return_tests, write_tests
from calfwrapper.studies import critic_noise, critic_threshold, relaxation_probability
from calfwrapper.tables import all_tables
from calfwrapper.verify import combine_reports, verify_trials, write_report


def run(command: list[str], *, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=ROOT, check=True)


def training_command(name: str, smoke: bool, output: Path) -> list[str]:
    output = output.resolve()
    configuration = TRAIN_CONFIGURATIONS[name]
    command = [sys.executable, "-m", configuration.module, *configuration.arguments]
    if configuration.module == "calfwrapper.training.ppo":
        command.extend(("--local-artifacts-path", str(output)))
    elif configuration.module == "calfwrapper.training.td3":
        command.extend(
            (
                "--tracking-uri",
                f"file://{output / 'mlruns'}",
                "--experiment-name",
                "calfwrapper/train",
                "--run-name",
                name,
                "--checkpoint-dir",
                str(output / name / "checkpoints"),
            )
        )
    else:
        command.extend(("--output-dir", str(output / name)))
    if smoke:
        command.extend(configuration.smoke_arguments)
    return command


def command_train(args: argparse.Namespace) -> None:
    names = tuple(TRAIN_CONFIGURATIONS) if args.name == "all" else (args.name,)
    for name in names:
        run(training_command(name, args.smoke, args.output), dry_run=args.dry_run)


def write_trials(trials: list[Trial], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trial",
                "episode_return",
                "goal_reached",
                "episode_length",
                "base_policy_actions",
                "fallback_policy_actions",
                "critic_evaluations",
                "policy_sequence_sha256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for number, trial in enumerate(trials):
            writer.writerow(trial.as_row(number))


def write_summary(trials_path: Path, destination: Path) -> None:
    with trials_path.open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    returns = [float(row["episode_return"]) for row in trials]
    goal_reaching_rate = 100 * statistics.fmean(
        row["goal_reached"].lower() == "true" for row in trials
    )
    summary = {
        "trials": len(trials),
        "mean_episode_return": statistics.fmean(returns),
        "goal_reaching_rate": goal_reaching_rate,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")


def run_evaluation_task(
    environment_name: str,
    stage: str,
    checkpoint_name: str,
    mode: Policy,
    output: Path,
    device: str,
) -> None:
    task_id = f"{environment_name}-{stage.lower()}-{mode}"
    print(f"Evaluating {environment_name}: {stage} / {mode}", flush=True)
    trials = [
        trial
        for size, seed in evaluation_batches(environment_name)
        for trial in evaluate(
            environment_name,
            checkpoint_name,
            mode,
            size,
            seed,
            device,
        )
    ]
    trials_path = output / "trials" / f"{task_id}.csv"
    write_trials(trials, trials_path)
    write_summary(trials_path, output / "summaries" / f"{task_id}.json")


def run_lagrangian_task(
    environment_name: str,
    stage: str,
    output: Path,
    device: str,
) -> None:
    task_id = f"{environment_name}-{stage.lower()}-lagrangian"
    print(f"Evaluating {environment_name}: {stage} / lagrangian", flush=True)
    trials = evaluate_lagrangian(environment_name, stage, device)
    trials_path = output / "trials" / f"{task_id}.csv"
    write_trials(trials, trials_path)
    write_summary(trials_path, output / "summaries" / f"{task_id}.json")


def command_eval(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    environments = tuple(ENVIRONMENTS) if args.name == "main" else (args.name,)
    verify_artifacts(set(environments))
    for environment_name in environments:
        environment = ENVIRONMENTS[environment_name]
        run_evaluation_task(
            environment_name,
            "all",
            environment.checkpoints[-1][1],
            "fallback",
            args.output,
            args.device,
        )
        for stage, checkpoint_name in environment.checkpoints:
            for mode in ("base", *OPERATING_MODES):
                run_evaluation_task(
                    environment_name,
                    stage,
                    checkpoint_name,
                    mode,
                    args.output,
                    args.device,
                )
            run_lagrangian_task(
                environment_name,
                stage,
                args.output,
                args.device,
            )
    report = combine_reports(
        verify_trials(
            args.output / "trials",
            REFERENCE_TRIALS / "main.csv",
            set(environments),
        ),
        verify_trials(
            args.output / "trials",
            REFERENCE_TRIALS / "lagrangian.csv",
            set(environments),
        ),
    )
    write_report(report, args.output / "verification.json")
    print(
        f"{report['status']}: {report['trials']} trials across "
        f"{report['tasks']} tasks; {report['mismatch_count']} mismatches"
    )
    if report["status"] != "passed":
        raise SystemExit(1)

    results = summarize(args.output / "trials", environments)
    write_results(results, args.output / "tables" / "table-6-main-results.csv")
    if set(environments) == set(ENVIRONMENTS):
        tests = paired_return_tests(args.output / "trials", environments)
        write_tests(tests, args.output / "tables" / "tables-7-8-10-11-tests.csv")
        all_tables(results, tests, args.output / "tables")
        main_results(results, args.output / "figures" / "figure-4-main-results.pdf")
        relaxation_probability(args.output, args.device)
        critic_threshold(args.output, args.device)
        critic_noise(args.output, args.device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(environments: set[str]) -> None:
    manifest_path = ROOT / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for item in manifest["checkpoints"]:
        if item["environment"] not in environments:
            continue
        path = ROOT / item["path"]
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"]:
            failures.append(item["id"])
    if failures:
        raise SystemExit(f"checkpoint verification failed: {', '.join(failures)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calfwrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a published configuration")
    train.add_argument("name", choices=("all", *TRAIN_CONFIGURATIONS))
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--output", type=Path, default=OUTPUTS / "training")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("eval", help="Run the main evaluation")
    evaluate.add_argument("name", choices=("main", *ENVIRONMENTS))
    evaluate.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device; the published exact-reproduction run used CUDA",
    )
    evaluate.add_argument("--output", type=Path, default=OUTPUTS / "evaluation")
    evaluate.set_defaults(handler=command_eval)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
