"""Aggregate frozen SOOPER held-out results and render paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LABELS = {"base": "Bare TD3", "fallback": "Fallback", "calf": "CALF-Wrapper", "sooper": "SOOPER"}
COLORS = {"base": "#777777", "fallback": "#2c7bb6", "calf": "#009e73", "sooper": "#d55e00"}
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_probability(value: float) -> str:
    if value >= 1e-3:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


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


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [("calf", "sooper"), ("sooper", "base"), ("calf", "base")]
    rows = []
    for metric in ("episode_return", "goal_reached", "constraint_satisfied"):
        pivot = frame.pivot(index="evaluation_seed", columns="method", values=metric)
        for left, right in pairs:
            difference = (pivot[left] - pivot[right]).to_numpy(float)
            p_value = 1.0 if not np.any(difference) else float(wilcoxon(difference, alternative="greater").pvalue)
            rows.append({
                "metric": metric,
                "alternative": f"{left}>{right}",
                "n_paired_seeds": len(difference),
                "mean_paired_difference": difference.mean(),
                "std_paired_difference": difference.std(ddof=1),
                "difference_ci95_half_width": 1.984 * difference.std(ddof=1) / np.sqrt(len(difference)),
                "wilcoxon_one_sided_p": p_value,
            })
    result = pd.DataFrame(rows)
    order = np.argsort(result.wilcoxon_one_sided_p.to_numpy())
    adjusted = np.empty(len(result))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(result) - rank) * result.wilcoxon_one_sided_p.iloc[index])
        adjusted[index] = min(running, 1.0)
    result["holm_adjusted_p"] = adjusted
    return result


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
    fig.savefig(output, bbox_inches="tight", metadata=PDF_METADATA)
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
    fig.savefig(pareto, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    metrics = [("goal_reaching_rate", "Goal-reaching rate"), ("constraint_satisfaction_rate", "Constraint satisfaction"), ("intervention_fraction", "Intervention fraction")]
    for axis, (column, title) in zip(axes, metrics):
        axis.bar([LABELS[x] for x in summary.method], summary[column], color=[COLORS[x] for x in summary.method])
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(reliability, bbox_inches="tight", metadata=PDF_METADATA)
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


def preserve_source_tables(protocol_path: Path, held_out_root: Path, output: Path) -> None:
    """Copy complete selection tables and summarize verified held-out sources."""
    selection_dir = protocol_path.parent
    for name in (
        "sooper_screening_trials.csv",
        "sooper_screening_summary.csv",
        "sooper_scenario_candidates.csv",
        "frozen_held_out_protocol.json",
    ):
        shutil.copyfile(selection_dir / name, output / name)
    shutil.copyfile(
        held_out_root / "command_manifest.json",
        output / "held_out_command_manifest.json",
    )
    summaries = []
    for path in sorted(held_out_root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        summaries.append(
            {
                "source": str(path.relative_to(held_out_root)),
                "source_size_bytes": path.stat().st_size,
                "source_sha256": sha256(path),
                "mlflow_run_id": payload["mlflow_run_id"],
                "method": payload.get("method", "sooper"),
                "training_seed": payload.get("training_seed"),
                "metrics": payload["metrics"],
            }
        )
    (output / "held_out_verified_run_summaries.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


def write_table(summary: pd.DataFrame, compute: pd.DataFrame, output: Path) -> None:
    joined = summary.merge(compute, on="method", validate="one_to_one")
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Frozen held-out comparison on 100 paired underwater-drone initial states. Reward reports mean $\pm$ standard deviation; brackets give 95\% confidence-interval half-widths for reward and rates. Interaction and compute columns count adaptation beyond the common pretrained TD3 checkpoint.}",
        r"\label{tab:sooper_heldout}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Method & Episode return & Goal rate & Constraint rate & Intervention & Offline steps & Online steps & Adaptation time (min) \\",
        r"\hline",
    ]
    for row in joined.itertuples(index=False):
        lines.append(
            f"{LABELS[row.method]} & {row.mean_reward:.1f} $\\pm$ {row.std_reward:.1f} "
            f"[$\\pm${row.reward_ci95_half_width:.1f}] & {row.goal_reaching_rate:.3f} "
            f"[$\\pm${row.goal_ci95_half_width:.3f}] & {row.constraint_satisfaction_rate:.3f} "
            f"[$\\pm${row.constraint_ci95_half_width:.3f}] & {row.intervention_fraction:.3f} "
            f"[$\\pm${row.intervention_ci95_half_width:.3f}] & "
            f"{int(row.offline_interactions)} & {int(row.online_interactions)} & "
            f"{row.mean_adaptation_wall_clock_seconds / 60.0:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    output.write_text("\n".join(lines) + "\n")


def write_macros(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    compute: pd.DataFrame,
    protocol: dict,
    screening: pd.DataFrame,
    output: Path,
) -> None:
    """Write machine-generated values used by the manuscript prose."""
    by_method = summary.set_index("method")
    by_comparison = tests.set_index(["metric", "alternative"])
    sooper_compute = compute.set_index("method").loc["sooper"]
    names = {"base": "Base", "sooper": "Sooper", "calf": "Calf", "fallback": "Fallback"}
    lines = [r"% Generated by scripts/analyze_sooper_comparison.py; do not edit by hand."]
    for method, macro_name in names.items():
        row = by_method.loc[method]
        lines.extend([
            rf"\newcommand{{\Sooper{macro_name}RewardMean}}{{{row.mean_reward:.1f}}}",
            rf"\newcommand{{\Sooper{macro_name}RewardStd}}{{{row.std_reward:.1f}}}",
            rf"\newcommand{{\Sooper{macro_name}RewardCI}}{{{row.reward_ci95_half_width:.1f}}}",
            rf"\newcommand{{\Sooper{macro_name}GoalRate}}{{{row.goal_reaching_rate:.3f}}}",
            rf"\newcommand{{\Sooper{macro_name}ConstraintRate}}{{{row.constraint_satisfaction_rate:.3f}}}",
            rf"\newcommand{{\Sooper{macro_name}Intervention}}{{{row.intervention_fraction:.3f}}}",
            rf"\newcommand{{\Sooper{macro_name}EmpiricalNab}}{{{1.0 - row.intervention_fraction:.3f}}}",
        ])
    for metric, metric_name, digits in (
        ("episode_return", "Reward", 1),
        ("goal_reached", "Goal", 3),
        ("constraint_satisfied", "Constraint", 3),
    ):
        for alternative, comparison_name in (
            (("calf", "sooper"), "CalfVsSooper"),
            (("sooper", "base"), "SooperVsBase"),
            (("calf", "base"), "CalfVsBase"),
        ):
            key = f"{alternative[0]}>{alternative[1]}"
            row = by_comparison.loc[(metric, key)]
            lines.extend([
                rf"\newcommand{{\Sooper{metric_name}{comparison_name}Difference}}{{{row.mean_paired_difference:.{digits}f}}}",
                rf"\newcommand{{\Sooper{metric_name}{comparison_name}CI}}{{{row.difference_ci95_half_width:.{digits}f}}}",
                rf"\newcommand{{\Sooper{metric_name}{comparison_name}PValue}}{{{latex_probability(row.holm_adjusted_p)}}}",
            ])
    lines.extend([
        rf"\newcommand{{\SooperScreeningSettings}}{{{len(screening)}}}",
        rf"\newcommand{{\SooperScreeningMaxGoal}}{{{screening.goal_reaching_rate.max():.3f}}}",
        rf"\newcommand{{\SooperDevCheckpointStep}}{{{protocol['checkpoint_step']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\SooperDevBaseReward}}{{{protocol['backbone']['development_reward']:.1f}}}",
        rf"\newcommand{{\SooperDevBaseGoal}}{{{protocol['backbone']['development_goal_reaching_rate']:.3f}}}",
        rf"\newcommand{{\SooperDevSooperReward}}{{{protocol['sooper']['development_reward']:.1f}}}",
        rf"\newcommand{{\SooperDevSooperGoal}}{{{protocol['sooper']['development_goal_reaching_rate']:.3f}}}",
        rf"\newcommand{{\SooperDevSooperConstraint}}{{{protocol['sooper']['development_constraint_satisfaction_rate']:.3f}}}",
        rf"\newcommand{{\SooperDevCalfReward}}{{{protocol['calf']['development_reward']:.1f}}}",
        rf"\newcommand{{\SooperDevCalfGoal}}{{{protocol['calf']['development_goal_reaching_rate']:.3f}}}",
        rf"\newcommand{{\SooperOfflineInteractions}}{{{int(sooper_compute.offline_interactions)}}}",
        rf"\newcommand{{\SooperOnlineInteractions}}{{{int(sooper_compute.online_interactions)}}}",
        rf"\newcommand{{\SooperAdaptationMinutes}}{{{sooper_compute.mean_adaptation_wall_clock_seconds / 60.0:.1f}}}",
        rf"\newcommand{{\SooperTotalInteractions}}{{{int((sooper_compute.offline_interactions + sooper_compute.online_interactions) * len(protocol['sooper']['runs'])):,}}}".replace(",", "{,}"),
        rf"\newcommand{{\SooperCostBudget}}{{{protocol['cost_budget']:.2f}}}",
        rf"\newcommand{{\SooperHeldOutSeeds}}{{{int(summary.n_paired_held_out_seeds.min())}}}",
    ])
    output.write_text("\n".join(lines) + "\n")


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
    tests = paired_tests(held_out)
    curve = learning_curve(protocol, args.screening_root)
    compute = compute_table(protocol, args.screening_root)
    preserve_source_tables(args.protocol, args.held_out_root, args.output_dir)
    held_out.to_csv(args.output_dir / "held_out_paired_trials.csv", index=False)
    raw_sooper.to_csv(args.output_dir / "held_out_sooper_all_training_seeds.csv", index=False)
    summary.to_csv(args.output_dir / "held_out_method_summary.csv", index=False)
    tests.to_csv(args.output_dir / "held_out_paired_tests.csv", index=False)
    curve.to_csv(args.output_dir / "sooper_learning_curve.csv", index=False)
    compute.to_csv(args.output_dir / "comparison_compute.csv", index=False)
    write_table(summary, compute, args.output_dir / "sooper_held_out_table.tex")
    screening = pd.read_csv(args.protocol.parent / "sooper_screening_summary.csv")
    write_macros(summary, tests, compute, protocol, screening, args.output_dir / "sooper_numbers.tex")
    plot_learning(curve, args.output_dir / "sooper_learning_curves.pdf")
    plot_comparison(summary, args.output_dir / "sooper_reward_reliability_pareto.pdf", args.output_dir / "sooper_reliability_comparison.pdf")
    output_records = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "format": "calf-wrapper-sooper-comparison-analysis-v1",
        "protocol_sha256": sha256(args.protocol),
        "paired_seeds_per_method": 100,
        "sooper_training_seeds": len(protocol["sooper"]["runs"]),
        "outputs": output_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
