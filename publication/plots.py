import matplotlib.pyplot as plt
import scienceplots
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import time

os.environ["TZ"] = "Europe/Moscow"
time.tzset()

plt.style.use("science")
plt.rcParams["font.size"] = 28
plt.rcParams["grid.linewidth"] = 2.5  # Width of lines in legend

data_path = Path(__file__).parent.parent / "reference-results"

with open(data_path / "pendulum.json", "r") as f:
    pendulum_data = json.load(f)

with open(data_path / "cartpole.json", "r") as f:
    cartpole_data = json.load(f)

Path("images").mkdir(exist_ok=True)


def plot(data, ylim, yticks, ylabel, delta_goal_reaching_rate, output_name, scale=1.0, creation_date=None):
    early = data["stages"]["early"]
    mid = data["stages"]["mid"]
    late = data["stages"]["late"]

    mapping_stage = {
        early: "Early training\nphase of base policy",
        mid: "Mid-stage training\nphase of base policy",
        late: "Late training\nphase of base policy",
    }

    mapping_colors = {
        "conservative": "green",
        "balanced": "orange",
        "brave": "red",
        "base": "blue",
        "fallback": "gray",
        "residual": "purple",
    }

    # Scale the figure size and font sizes
    fig_width, fig_height = 15 * scale, 7 * scale
    font_size = 28 * scale
    marker_size = 5 * scale  # Base marker size scaled with the figure

    plt.rcParams["font.size"] = font_size

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(fig_width, fig_height), sharey=True
    )
    axes = {early: ax1, mid: ax2, late: ax3}
    fallback_cartpole = data["fallback"]

    for step_id, step in [(early, "early"), (mid, "mid"), (late, "late")]:
        ax = axes[step_id]

        width = 0.5
        ax.plot(
            [0],
            [data[step]["conservative"]["mean_reward"]],
            "ko",
            markersize=marker_size,
            label="Mean",
        )
        ax.plot(
            [1], [data[step]["balanced"]["mean_reward"]], "ko", markersize=marker_size
        )
        ax.plot([2], [data[step]["brave"]["mean_reward"]], "ko", markersize=marker_size)
        ax.plot([3], [data[step]["base"]["mean_reward"]], "ko", markersize=marker_size)
        ax.plot([4], [data[step]["residual"]["mean_reward"]], "ko", markersize=marker_size)
        ax.plot([5], [fallback_cartpole["mean_reward"]], "ko", markersize=marker_size)

        ax.bar(
            [0],
            2 * data[step]["conservative"]["std_reward"],
            bottom=data[step]["conservative"]["mean_reward"]
            - data[step]["conservative"]["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["conservative"],
            label="±std",
        )
        ax.bar(
            [1],
            2 * data[step]["balanced"]["std_reward"],
            bottom=data[step]["balanced"]["mean_reward"]
            - data[step]["balanced"]["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["balanced"],
        )
        ax.bar(
            [2],
            2 * data[step]["brave"]["std_reward"],
            bottom=data[step]["brave"]["mean_reward"]
            - data[step]["brave"]["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["brave"],
        )
        ax.bar(
            [3],
            2 * data[step]["base"]["std_reward"],
            bottom=data[step]["base"]["mean_reward"] - data[step]["base"]["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["base"],
        )
        ax.bar(
            [4],
            2 * data[step]["residual"]["std_reward"],
            bottom=data[step]["residual"]["mean_reward"] - data[step]["residual"]["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["residual"],
        )
        ax.bar(
            [5],
            2 * fallback_cartpole["std_reward"],
            bottom=fallback_cartpole["mean_reward"] - fallback_cartpole["std_reward"],
            width=width,
            alpha=0.4,
            color=mapping_colors["fallback"],
        )
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(
            [
                "Conservative",
                "Balanced",
                "Brave",
                "Base policy",
                "Residual RL policy",
                "Fallback policy",
            ],
            rotation=90,
        )
        ax.set_title(mapping_stage[step_id], position=(0.5, 1.2))
        ax.grid(True, alpha=0.3)

        goal_reach = {
            "conservative": int(data[step]["conservative"]["goal_reaching_rate"]),
            "balanced": int(data[step]["balanced"]["goal_reaching_rate"]),
            "brave": int(data[step]["brave"]["goal_reaching_rate"]),
            "base": int(data[step]["base"]["goal_reaching_rate"]),
            "residual": int(data[step]["residual"]["goal_reaching_rate"]),
            "fallback": 100,
        }
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)

        ax.text(
            2.5,
            ax.get_ylim()[0] + delta_goal_reaching_rate,
            "Goal Reaching Rate (\\%)",
            ha="center",
            va="top",
            fontsize=23 * scale,
        )

        for idx, mode in enumerate(
            ["conservative", "balanced", "brave", "base", "residual", "fallback"]
        ):
            ax.text(
                idx,
                ax.get_ylim()[0],
                goal_reach[mode],
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=25 * scale,
            )

        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2)
        if step_id == early:
            ax.set_ylabel(f"Accumulated\nReward", y=0.65)
    plt.tight_layout()
    plt.suptitle(f"\\texttt{{{ylabel}}}", y=1.02, fontsize=font_size * 1.1)
    plt.savefig(f"images/{output_name}", dpi=300 * scale, metadata={"CreationDate": creation_date})


plot(
    cartpole_data,
    ylim=(-4000, 300),
    yticks=[-2400, -1700, -1000, -300],
    delta_goal_reaching_rate=1300,
    ylabel="CartPoleSwingUpEnv",
    output_name="cartpole.pdf",
    scale=0.6,  # Default scale
    creation_date=datetime(2025, 8, 5, 15, 33, 55),
)

plot(
    pendulum_data,
    ylim=(-1350, 50),
    yticks=[-800, -600, -400, -200],
    delta_goal_reaching_rate=400,
    ylabel="Pendulum-v1",
    output_name="pendulum.pdf",
    scale=0.6,  # Default scale
    creation_date=datetime(2025, 8, 5, 15, 33, 56),
)
