"""Aggregate frozen SOOPER held-out results and render paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {"base": "Bare TD3", "fallback": "Fallback", "calf": "CALF-Wrapper", "sooper": "SOOPER"}
COLORS = {"base": "#777777", "fallback": "#2c7bb6", "calf": "#009e73", "sooper": "#d55e00"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load_held_out(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = []
    raw_sooper = []
    for method in ("base", "fallback", "calf"):
        frame = pd.read_csv(root / method / "held_out_trials.csv")
        frame["method"] = method
        paired.append(frame)
    for path in sorted(root.glob("sooper-seed-*/held_out_trials.csv")):
        frame = pd.read_csv(path)
        frame["method"] = "sooper"
        frame["sooper_training_seed"] = int(path.parent.name.rsplit("-", 1)[-1])
        raw_sooper.append(frame)
    if not raw_sooper:
        raise SystemExit("No held-out SOOPER trials found")
    raw = pd.concat(raw_sooper, ignore_index=True)
    averaged = raw.groupby("evaluation_seed", as_index=False).agg(
        episode_return=("episode_return", "mean"),
        discounted_cost=("discounted_cost", "mean"),
        constraint_satisfied=("constraint_satisfied", "mean"),
        goal_reached=("goal_reached", "mean"),
        intervention_fraction=("intervention_fraction", "mean"),
        episode_length=("episode_length", "mean"),
    )
    averaged["method"] = "sooper"
    paired.append(averaged)
    return pd.concat(paired, ignore_index=True), raw


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method"):
        reward = group.episode_return.to_numpy(float)
        rows.append({
            "method": method,
            "n_paired_held_out_seeds": len(group),
            "mean_reward": reward.mean(),
            "std_reward": reward.std(ddof=1),
            "reward_ci95_half_width": 1.984 * reward.std(ddof=1) / np.sqrt(len(reward)),
            "goal_reaching_rate": group.goal_reached.mean(),
            "goal_ci95_half_width": 1.984 * group.goal_reached.std(ddof=1) / np.sqrt(len(group)),
            "constraint_satisfaction_rate": group.constraint_satisfied.mean(),
            "constraint_ci95_half_width": 1.984 * group.constraint_satisfied.std(ddof=1) / np.sqrt(len(group)),
            "intervention_fraction": group.intervention_fraction.mean(),
            "intervention_ci95_half_width": 1.984 * group.intervention_fraction.std(ddof=1) / np.sqrt(len(group)),
        })
    return pd.DataFrame(rows).set_index("method").loc[["base", "sooper", "calf", "fallback"]].reset_index()


def learning_curve(protocol: dict, screening_root: Path) -> pd.DataFrame:
    rows = []
    for run in protocol["sooper"]["runs"]:
        path = screening_root / run["run_dir"] / "raw" / "evaluation_trials.csv"
        frame = pd.read_csv(path)
        frame["sooper_training_seed"] = run["sooper_training_seed"]
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["online_interactions"] = np.maximum(data.iteration + 1, 0) * protocol["horizon"]
    grouped = data.groupby("online_interactions", as_index=False).agg(
        mean_reward=("episode_return", "mean"),
        reward_std=("episode_return", "std"),
        goal_reaching_rate=("goal_reached", "mean"),
        constraint_satisfaction_rate=("constraint_satisfied", "mean"),
        intervention_fraction=("intervention_fraction", "mean"),
        n_trials=("episode_return", "size"),
    )
    grouped["reward_ci95_half_width"] = 1.96 * grouped.reward_std / np.sqrt(grouped.n_trials)
    return grouped


def plot_learning(curve: pd.DataFrame, output: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    x = curve.online_interactions
    axes[0].plot(x, curve.mean_reward, marker="o", color=COLORS["sooper"])
    axes[0].fill_between(x, curve.mean_reward - curve.reward_ci95_half_width, curve.mean_reward + curve.reward_ci95_half_width, color=COLORS["sooper"], alpha=.18)
    axes[0].set_ylabel("Episode return")
    axes[1].plot(x, curve.goal_reaching_rate, marker="o", label="Goal reaching", color="#2c7bb6")
    axes[1].plot(x, curve.constraint_satisfaction_rate, marker="s", label="Constraint satisfaction", color="#009e73")
    axes[1].plot(x, 1 - curve.intervention_fraction, marker="^", label="Non-intervention", color="#d55e00")
    axes[1].set_ylim(-.05, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Additional online interactions")
        axis.grid(alpha=.2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, pareto: Path, reliability: Path) -> None:
    style()
    fig, axis = plt.subplots(figsize=(3.55, 2.65), constrained_layout=True)
    for row in summary.itertuples():
        axis.errorbar(row.goal_reaching_rate, row.mean_reward, xerr=row.goal_ci95_half_width, yerr=row.reward_ci95_half_width, marker="o", capsize=2, color=COLORS[row.method], label=LABELS[row.method])
    axis.set_xlabel("Goal-reaching rate")
    axis.set_ylabel("Episode return")
    axis.set_xlim(-.05, 1.05)
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    fig.savefig(pareto, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    metrics = [("goal_reaching_rate", "Goal-reaching rate"), ("constraint_satisfaction_rate", "Constraint satisfaction"), ("intervention_fraction", "Intervention fraction")]
    for axis, (column, title) in zip(axes, metrics):
        axis.bar([LABELS[x] for x in summary.method], summary[column], color=[COLORS[x] for x in summary.method])
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(reliability, bbox_inches="tight")
    plt.close(fig)


def compute_table(protocol: dict, screening_root: Path) -> pd.DataFrame:
    wall = []
    for run in protocol["sooper"]["runs"]:
        summary = json.loads((screening_root / run["run_dir"] / "summary.json").read_text())
        wall.append(summary["wall_clock_seconds"])
    return pd.DataFrame([
        {"method": "base", "offline_interactions": 0, "online_interactions": 0, "additional_trainable_components": 0, "mean_adaptation_wall_clock_seconds": 0.0, "mean_gpu_hours": 0.0},
        {"method": "fallback", "offline_interactions": 0, "online_interactions": 0, "additional_trainable_components": 0, "mean_adaptation_wall_clock_seconds": 0.0, "mean_gpu_hours": 0.0},
        {"method": "calf", "offline_interactions": 0, "online_interactions": 0, "additional_trainable_components": 0, "mean_adaptation_wall_clock_seconds": 0.0, "mean_gpu_hours": 0.0},
        {"method": "sooper", "offline_interactions": protocol["sooper"]["offline_interactions_per_training_seed"], "online_interactions": protocol["sooper"]["real_interactions_per_training_seed"], "additional_trainable_components": 3, "mean_adaptation_wall_clock_seconds": np.mean(wall), "mean_gpu_hours": np.mean(wall) / 3600.0},
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    held_out, raw_sooper = load_held_out(args.held_out_root)
    summary = summarize(held_out)
    curve = learning_curve(protocol, args.screening_root)
    compute = compute_table(protocol, args.screening_root)
    held_out.to_csv(args.output_dir / "held_out_paired_trials.csv", index=False)
    raw_sooper.to_csv(args.output_dir / "held_out_sooper_all_training_seeds.csv", index=False)
    summary.to_csv(args.output_dir / "held_out_method_summary.csv", index=False)
    curve.to_csv(args.output_dir / "sooper_learning_curve.csv", index=False)
    compute.to_csv(args.output_dir / "comparison_compute.csv", index=False)
    plot_learning(curve, args.output_dir / "sooper_learning_curves.pdf")
    plot_comparison(summary, args.output_dir / "sooper_reward_reliability_pareto.pdf", args.output_dir / "sooper_reliability_comparison.pdf")
    manifest = {"format": "calf-wrapper-sooper-comparison-analysis-v1", "protocol_sha256": sha256(args.protocol), "paired_seeds_per_method": 100, "sooper_training_seeds": len(protocol["sooper"]["runs"]), "outputs": sorted(path.name for path in args.output_dir.iterdir())}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
