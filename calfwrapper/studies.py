"""Sensitivity and ablation studies reported in the article."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from calfwrapper._protocol import evaluation_batches
from calfwrapper.environments import ENVIRONMENTS
from calfwrapper.evaluation import Policy, Trial, evaluate
from calfwrapper.operating_modes import (
    OperatingModeParameters,
    fixed_acceptance_budget,
)
from calfwrapper.paths import REFERENCE_STUDIES

MODE_LABELS = {"balanced": "Balanced", "high": "High"}
ENVIRONMENT_LABELS = {
    "pendulum": "Pendulum",
    "cartpole": "CartPole",
    "auv": "AUV",
    "robot": "Robot",
}
ENVIRONMENT_COLORS = {
    "pendulum": "#0072B2",
    "cartpole": "#E69F00",
    "auv": "#009E73",
    "robot": "#CC79A7",
}


def _trials(
    environment: str,
    stage: str,
    mode: Policy,
    device: str,
    *,
    parameters: OperatingModeParameters,
    nu: float | None = None,
) -> list[Trial]:
    suffix = ".zip" if ENVIRONMENTS[environment].algorithm == "ppo" else ".pt"
    return [
        trial
        for size, seed in evaluation_batches(environment)
        for trial in evaluate(
            environment,
            f"{stage.lower()}{suffix}",
            mode,
            size,
            seed,
            device,
            parameters=parameters,
            nu=nu,
        )
    ]


def _summary(trials: list[Trial]) -> dict[str, float]:
    returns = [trial.episode_return for trial in trials]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return {
        "mean_episode_return": mean,
        "return_ci95_half_width": 1.96 * math.sqrt(variance) / math.sqrt(len(returns)),
        "goal_reaching_rate": 100 * sum(trial.goal_reached for trial in trials) / len(trials),
        "fallback_action_percentage": 100
        * sum(trial.fallback_policy_actions / trial.episode_length for trial in trials)
        / len(trials),
    }


def _write(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify(rows: list[dict[str, object]], reference_name: str) -> None:
    with (REFERENCE_STUDIES / reference_name).open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(rows) != len(expected):
        raise RuntimeError(
            f"{reference_name}: generated {len(rows)} rows; expected {len(expected)}"
        )
    for row_number, (actual, reference) in enumerate(zip(rows, expected, strict=True), start=2):
        for field, expected_value in reference.items():
            actual_value = str(actual[field])
            try:
                equal = math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            except ValueError:
                equal = actual_value == expected_value
            if not equal:
                raise RuntimeError(
                    f"{reference_name}:{row_number}: {field} is {actual_value}; "
                    f"expected {expected_value}"
                )
    print(f"Verified {reference_name}: {len(rows)} rows match the published results")


def relaxation_probability(output: Path, device: str) -> list[dict[str, object]]:
    """Vary p_relax at fixed acceptance budgets for the Mid checkpoints."""

    settings = (
        ("balanced", 0.50, (0.55, 0.60, 0.70, 0.85, 1.00)),
        ("high", 0.70, (0.75, 0.80, 0.85, 0.90, 1.00)),
    )
    rows: list[dict[str, object]] = []
    for environment, config in ENVIRONMENTS.items():
        for mode, budget, probabilities in settings:
            for p_relax in probabilities:
                parameters = fixed_acceptance_budget(budget, p_relax, config.horizon)
                trials = _trials(
                    environment,
                    "Mid",
                    mode,
                    device,
                    parameters=parameters,
                )
                rows.append(
                    {
                        "environment": environment,
                        "mode": mode,
                        "acceptance_budget": budget,
                        "p_relax": p_relax,
                        "lambda": parameters.lambda_,
                        "trials": len(trials),
                        **_summary(trials),
                    }
                )
    _write(rows, output / "tables" / "figure-2-relaxation-probability.csv")
    _verify(rows, "relaxation_probability.csv")
    _plot_relaxation_probability(rows, output / "figures" / "figure-2-relaxation-probability.pdf")
    return rows


def _plot_relaxation_probability(
    rows: list[dict[str, object]],
    destination: Path,
) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(3.75, 1.68), sharey=True)
    for axis, mode in zip(axes, ("balanced", "high"), strict=True):
        for environment in ENVIRONMENTS:
            selected = sorted(
                [row for row in rows if row["environment"] == environment and row["mode"] == mode],
                key=lambda row: cast(float, row["p_relax"]),
            )
            axis.plot(
                [cast(float, row["p_relax"]) for row in selected],
                [cast(float, row["goal_reaching_rate"]) for row in selected],
                marker="o",
                markersize=3,
                linewidth=1.1,
                color=ENVIRONMENT_COLORS[environment],
                label=ENVIRONMENT_LABELS[environment],
            )
        budget = 0.50 if mode == "balanced" else 0.70
        axis.set_title(
            f"{MODE_LABELS[mode]}\n" + rf"(Target $\mathrm{{NAB}}_T={budget:.2f}$)",
            pad=1,
        )
        axis.set_ylim(-3, 103)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.grid(True, color="#dddddd", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel(r"$p_{\mathrm{relax}}$")
        axis.tick_params(axis="x", rotation=25, pad=1)
    axes[0].set_ylabel("Goal-reaching\nrate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.30, wspace=0.18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    plt.close(figure)


class _CriticNoise:
    def __init__(
        self,
        environment: str,
        first_seed: int,
        size: int,
        scale: float,
    ) -> None:
        gym_id = ENVIRONMENTS[environment].gym_id
        self.random = [
            np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"calf-critic-noise-v1|{gym_id}|{first_seed + trial}".encode()
                    ).digest()[:8],
                    "big",
                )
            )
            for trial in range(size)
        ]
        self.scale = scale

    def __call__(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values).reshape(-1)
        noise = np.asarray([random.standard_normal() for random in self.random])
        return (flat + self.scale * noise).reshape(-1, 1)


def critic_noise(output: Path, device: str) -> list[dict[str, object]]:
    """Add controlled noise to the critic at the High-return checkpoints."""

    parameters = OperatingModeParameters(p_relax=0.0, lambda_=0.0)
    rows: list[dict[str, object]] = []
    for environment, config in ENVIRONMENTS.items():
        for sigma in (0.0, 2.0, 20.0, 200.0):
            trials = [
                trial
                for size, seed in evaluation_batches(environment)
                for trial in evaluate(
                    environment,
                    "high.zip" if config.algorithm == "ppo" else "high.pt",
                    "conservative",
                    size,
                    seed,
                    device,
                    parameters=parameters,
                    critic_transform=_CriticNoise(
                        environment,
                        seed,
                        size,
                        sigma * config.nu,
                    ),
                )
            ]
            rows.append(
                {
                    "environment": environment,
                    "sigma": sigma,
                    "noise_scale": sigma * config.nu,
                    "trials": len(trials),
                    **_summary(trials),
                }
            )
    _write(rows, output / "tables" / "figure-5-critic-noise.csv")
    _verify(rows, "critic_noise.csv")
    _plot_critic_noise(rows, output / "figures" / "figure-5-critic-noise.pdf")
    return rows


def _plot_critic_noise(rows: list[dict[str, object]], destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 5.5,
            "axes.titlesize": 6.0,
            "axes.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(3.5, 2.0), sharex="col")
    for column, environment in enumerate(ENVIRONMENTS):
        selected = [row for row in rows if row["environment"] == environment]
        sigma = [cast(float, row["sigma"]) for row in selected]
        x = list(range(len(sigma)))
        means = [cast(float, row["mean_episode_return"]) for row in selected]
        intervals = [cast(float, row["return_ci95_half_width"]) for row in selected]
        axes[0, column].fill_between(
            x,
            [mean - interval for mean, interval in zip(means, intervals, strict=True)],
            [mean + interval for mean, interval in zip(means, intervals, strict=True)],
            color="#007D3C",
            alpha=0.16,
            linewidth=0,
        )
        axes[0, column].plot(
            x,
            means,
            color="#007D3C",
            marker="o",
            markersize=2.3,
            linewidth=1.0,
        )
        axes[1, column].plot(
            x,
            [cast(float, row["goal_reaching_rate"]) for row in selected],
            color="#009E73",
            marker="o",
            markersize=2.3,
            linewidth=1.0,
            label="Goal-reaching rate",
        )
        axes[1, column].plot(
            x,
            [cast(float, row["fallback_action_percentage"]) for row in selected],
            color="#D55E00",
            marker="s",
            markersize=2.1,
            linewidth=0.95,
            linestyle="--",
            label="Fallback-policy actions",
        )
        axes[0, column].set_title(ENVIRONMENT_LABELS[environment], pad=2)
        axes[1, column].set_xticks(x, [f"{value:g}" for value in sigma])
        axes[1, column].set_ylim(65, 102)
        axes[1, column].set_yticks((70, 80, 90, 100))
        axes[1, column].set_xlabel(r"$\sigma$", labelpad=1)
        for axis in axes[:, column]:
            axis.grid(True, color="#dddddd", linewidth=0.45)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=5.0)
    axes[0, 0].set_ylabel("Mean return")
    axes[1, 0].set_ylabel("Rate (%)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6)
    figure.subplots_adjust(left=0.17, right=0.985, top=0.91, bottom=0.22, hspace=0.18, wspace=0.68)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    plt.close(figure)


def critic_threshold(output: Path, device: str) -> list[dict[str, object]]:
    """Vary the critic threshold at the High-return checkpoints."""

    multipliers = tuple(2.0**power for power in range(-4, 11)) + (math.inf,)
    parameters = OperatingModeParameters(p_relax=0.0, lambda_=0.0)
    rows: list[dict[str, object]] = []
    for environment, config in ENVIRONMENTS.items():
        for multiplier in multipliers:
            seed = 20260801 if environment == "cartpole" else 42
            trials = evaluate(
                environment,
                "high.zip" if config.algorithm == "ppo" else "high.pt",
                "conservative",
                100,
                seed,
                device,
                parameters=parameters,
                nu=multiplier * config.nu,
            )
            rows.append(
                {
                    "environment": environment,
                    "eta": multiplier,
                    "nu": multiplier * config.nu,
                    "trials": len(trials),
                    **_summary(trials),
                }
            )
    _write(rows, output / "tables" / "figure-3-critic-threshold.csv")
    _verify(rows, "critic_threshold.csv")
    _plot_critic_threshold(rows, output / "figures" / "figure-3-critic-threshold.pdf")
    return rows


def _plot_critic_threshold(rows: list[dict[str, object]], destination: Path) -> None:
    names = {
        "pendulum": "Pendulum",
        "cartpole": "CartPole Swing-Up",
        "auv": "Contaminated-Zone AUV",
        "robot": "Treasure-Collecting Robot",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(3.5, 2.0), sharex=True)
    for axis, environment in zip(axes.flat, ENVIRONMENTS, strict=True):
        selected = [row for row in rows if row["environment"] == environment]
        x = [
            12 if math.isinf(cast(float, row["eta"])) else math.log2(cast(float, row["eta"]))
            for row in selected
        ]
        means = [cast(float, row["mean_episode_return"]) for row in selected]
        intervals = [cast(float, row["return_ci95_half_width"]) for row in selected]
        axis.fill_between(
            x,
            [mean - interval for mean, interval in zip(means, intervals, strict=True)],
            [mean + interval for mean, interval in zip(means, intervals, strict=True)],
            color="#0072B2",
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(x, means, color="#0072B2", marker="o", markersize=3, linewidth=1.25)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        axis.axvline(11, color="#BBBBBB", linewidth=0.6)
        axis.grid(True, color="#dddddd", linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.set_title(names[environment])
    for axis in axes[-1]:
        axis.set_xlabel(r"Scaling coefficient $\eta$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Episode return")
    for axis in axes.flat:
        axis.set_xticks((-4, 0, 4, 8, 12), (r"$1/16$", "$1$", "$16$", "$256$", r"$\infty$"))
    for axis in axes.flat:
        axis.tick_params(axis="both", labelsize=6.5)
        axis.tick_params(axis="x", labelrotation=25)
    figure.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.21, hspace=0.54, wspace=0.42)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    plt.close(figure)
