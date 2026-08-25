"""Figures generated from the reproduced evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calfwrapper.results import Result

ENVIRONMENTS = (
    ("pendulum", "Pendulum-v1"),
    ("cartpole", "CartPole Swing-Up"),
    ("auv", "Contaminated-Zone\nAUV"),
    ("robot", "Treasure-Collecting\nRobot"),
)
STAGES = ("Low", "Mid", "High")
METHODS = (
    ("fallback", "Fallback"),
    ("conservative", "Conservative"),
    ("guarded", "Guarded"),
    ("moderate", "Moderate"),
    ("balanced", "Balanced"),
    ("high", "High"),
    ("almost_open", "Almost Open"),
    ("base", "Base policy"),
    ("lagrangian", "Lagrangian baseline"),
)
COLORS = {
    "fallback": "#111111",
    "conservative": "#352A87",
    "guarded": "#0072B2",
    "moderate": "#009E73",
    "balanced": "#E69F00",
    "high": "#D55E00",
    "almost_open": "#CC79A7",
    "base": "#666666",
    "lagrangian": "#8C564B",
}


def _scaled_axis(axis, environment: str, lower: float, upper: float) -> None:
    if environment not in {"auv", "robot"}:
        ticks = [tick for tick in axis.get_yticks() if lower <= tick <= upper]
        if environment in {"pendulum", "cartpole"}:
            ticks.append(0.0)
        axis.set_yticks(sorted(set(ticks)))
        return

    boundary = -675.74 if environment == "auv" else -38.541
    split = 0.65

    def forward(value):
        value = np.asarray(value)
        return np.where(
            value <= boundary,
            split * (value - lower) / (boundary - lower),
            split + (1.0 - split) * (value - boundary) / (upper - boundary),
        )

    def inverse(value):
        value = np.asarray(value)
        return np.where(
            value <= split,
            lower + value * (boundary - lower) / split,
            boundary + (value - split) * (upper - boundary) / (1.0 - split),
        )

    axis.set_yscale("function", functions=(forward, inverse))
    axis.set_ylim(lower, upper)
    if environment == "auv":
        axis.set_yticks((-6000, -4000, -2000, boundary, -550, -450))
    else:
        axis.set_yticks((-500, -300, -150, boundary, 0, 25, 50))
    style = {
        "color": "black",
        "clip_on": False,
        "linewidth": 0.7,
        "transform": axis.transAxes,
    }
    axis.plot((-0.025, 0.025), (split - 0.018, split + 0.006), **style)
    axis.plot((-0.025, 0.025), (split + 0.012, split + 0.036), **style)


def main_results(rows: list[Result], destination: Path) -> None:
    """Render the episode returns and goal-reaching rates shown in Figure 4."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.2, 8.4))
    outer = figure.add_gridspec(2, 1, hspace=0.18)
    grids = (
        outer[0].subgridspec(2, 3, hspace=0.10, wspace=0.28),
        outer[1].subgridspec(2, 3, hspace=0.10, wspace=0.28),
    )

    for environment_index, (environment, environment_label) in enumerate(ENVIRONMENTS):
        environment_rows = [row for row in rows if row["environment"] == environment]
        data_lower = min(
            float(row["mean_episode_return"]) - float(row["return_ci95_half_width"])
            for row in environment_rows
        )
        data_upper = max(
            float(row["mean_episode_return"]) + float(row["return_ci95_half_width"])
            for row in environment_rows
        )
        span = data_upper - data_lower
        lower = data_lower - 0.34 * span
        upper = data_upper + 0.08 * span
        if environment in {"pendulum", "cartpole"}:
            upper = max(upper, 0.0)
        elif environment == "auv":
            upper = -450.0
        else:
            upper = 50.0

        for stage_index, stage in enumerate(STAGES):
            axis = figure.add_subplot(
                grids[environment_index // 2][environment_index % 2, stage_index]
            )
            stage_rows: dict[str, Result] = {
                row["method"]: row for row in environment_rows if row["checkpoint"] == stage
            }
            for method_index, (method, _) in enumerate(METHODS):
                row = stage_rows[method]
                mean = float(row["mean_episode_return"])
                interval = float(row["return_ci95_half_width"])
                axis.bar(
                    method_index,
                    2.0 * interval,
                    bottom=mean - interval,
                    width=0.62,
                    color=COLORS[method],
                    alpha=0.72,
                    linewidth=0,
                )
                axis.plot(
                    method_index,
                    mean,
                    marker="o",
                    color="black",
                    markersize=2.8,
                    linestyle="none",
                )
                axis.text(
                    method_index,
                    0.035,
                    f"{float(row['goal_reaching_rate']):.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    transform=axis.get_xaxis_transform(),
                )

            axis.set_xlim(-0.6, len(METHODS) - 0.4)
            axis.set_ylim(lower, upper)
            _scaled_axis(axis, environment, lower, upper)
            axis.set_xticks(range(len(METHODS)))
            labels = [
                (
                    "PPO-Lagrangian"
                    if method == "lagrangian" and environment in {"pendulum", "cartpole"}
                    else "TD3-Lagrangian"
                    if method == "lagrangian"
                    else label
                )
                for method, label in METHODS
            ]
            axis.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=6.5)
            if environment_index in {0, 2}:
                axis.tick_params(axis="x", labelbottom=False)
            axis.tick_params(axis="x", pad=2)
            axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(
                (len(METHODS) - 1) / 2,
                0.15,
                "Goal-reaching rate (%)",
                ha="center",
                va="center",
                fontsize=6.5,
                transform=axis.get_xaxis_transform(),
            )
            if environment_index == 0:
                axis.set_title(f"{stage}-return checkpoint", pad=3)
            if stage_index == 0:
                axis.set_ylabel("Episode return")
                axis.text(
                    -0.58 if environment_index >= 2 else -0.42,
                    0.5,
                    environment_label,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    transform=axis.transAxes,
                )
            else:
                axis.tick_params(axis="y", labelleft=False)

    figure.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    plt.close(figure)
