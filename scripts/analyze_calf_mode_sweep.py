"""Aggregate the completed NAB sweep and render manuscript-ready figures."""

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


MODE_ORDER = ["conservative", "moderate", "high", "almost_open"]
MODE_LABEL = {
    "conservative": "Conservative",
    "moderate": "Moderate",
    "high": "High",
    "almost_open": "Almost Open",
}
ENV_LABEL = {
    "Pendulum-v1": "Pendulum",
    "CartpoleSwingupEnvLong-v0": "CartPole",
    "UnderwaterDrone-v0": "Underwater drone",
    "RobotNavigationConstSpeedCatch-v0": "Robot navigation",
}
LEGACY_KEYS = {"conservative": "Legacy Conservative", "balanced": "Legacy Balanced", "brave": "Legacy Brave"}
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(source: Path, expected_rows: int) -> pd.DataFrame:
    data = pd.read_csv(source)
    if len(data) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} completed tasks, found {len(data)}")
    key = ["environment", "algorithm", "training_seed", "checkpoint_step", "mode"]
    checkpoint_rows = data[data["training_seed"].notna()].copy()
    if checkpoint_rows.duplicated(key).any():
        raise SystemExit("Duplicate canonical checkpoint-mode rows detected")
    counts = checkpoint_rows.groupby(key[:-1])["mode"].nunique()
    if not (counts == 5).all():
        raise SystemExit("At least one checkpoint lacks base or a NAB mode")
    base = checkpoint_rows[checkpoint_rows["mode"] == "base"][
        key[:-1] + ["mean_reward", "goal_reaching_rate"]
    ].rename(columns={"mean_reward": "base_reward", "goal_reaching_rate": "base_goal_reaching_rate"})
    checkpoint_rows = checkpoint_rows.merge(base, on=key[:-1], validate="many_to_one")
    checkpoint_rows["reward_gain_over_base"] = checkpoint_rows["mean_reward"] - checkpoint_rows["base_reward"]
    checkpoint_rows["goal_reaching_rate"] = checkpoint_rows["goal_reaching_rate"] / 100.0
    checkpoint_rows["base_goal_reaching_rate"] = checkpoint_rows["base_goal_reaching_rate"] / 100.0
    checkpoint_rows["model_path"] = checkpoint_rows.apply(
        lambda row: (
            f"run/artifacts/{'td3' if row.algorithm == 'cleanrl_td3' else 'ppo'}_"
            f"{row.environment}_{int(row.training_seed)}/checkpoints/"
            f"{'td3' if row.algorithm == 'cleanrl_td3' else 'ppo'}_checkpoint_"
            f"{int(row.checkpoint_step)}_steps.{ 'pt' if row.algorithm == 'cleanrl_td3' else 'zip'}"
        ),
        axis=1,
    )
    return checkpoint_rows


def ci95(values: pd.Series) -> float:
    values = values.dropna().to_numpy(dtype=float)
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def one_sided_wilcoxon(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if not np.any(array):
        return 1.0
    return float(wilcoxon(array, alternative="greater").pvalue)


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(float))
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values.iloc[index]))
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=values.index)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[data["mode"].isin(MODE_ORDER)].copy()
    rows = []
    for (environment, mode), group in selected.groupby(["environment", "mode"], sort=False):
        rows.append(
            {
                "environment": environment,
                "mode": mode,
                "checkpoints": len(group),
                "horizon": int(group["horizon"].iloc[0]),
                "target_nab": group["calf_target_acceptance_budget"].mean(),
                "p0": group["calf_relaxprob_init"].mean(),
                "lambda": group["calf_relaxprob_factor"].mean(),
                "empirical_nab": group["base_action_fraction"].mean(),
                "fallback_fraction": group["fallback_action_fraction"].mean(),
                "reward_gain_mean": group["reward_gain_over_base"].mean(),
                "reward_gain_std": group["reward_gain_over_base"].std(ddof=1),
                "reward_gain_ci95": ci95(group["reward_gain_over_base"]),
                "goal_reaching_rate": group["goal_reaching_rate"].mean(),
                "goal_reaching_ci95": ci95(group["goal_reaching_rate"]),
                "goal_rate_gain": (group["goal_reaching_rate"] - group["base_goal_reaching_rate"]).mean(),
                "reward_win_fraction": (group["reward_gain_over_base"] > 0).mean(),
                "goal_noninferiority_fraction": (group["goal_reaching_rate"] >= group["base_goal_reaching_rate"]).mean(),
                "reward_gain_wilcoxon_p": one_sided_wilcoxon(group["reward_gain_over_base"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["reward_gain_wilcoxon_holm_p"] = holm_adjust(summary["reward_gain_wilcoxon_p"])
    return summary.sort_values(
        ["environment", "mode"], key=lambda col: col.map({mode: i for i, mode in enumerate(MODE_ORDER)}).fillna(col)
    )


def matched_legacy(data: pd.DataFrame, pendulum: Path, cartpole: Path) -> pd.DataFrame:
    specs = [
        ("Pendulum-v1", 9, json.loads(pendulum.read_text())),
        ("CartpoleSwingupEnvLong-v0", 42, json.loads(cartpole.read_text())),
    ]
    rows = []
    for environment, training_seed, legacy in specs:
        for stage, checkpoint_step in legacy["stages"].items():
            for mode, label in LEGACY_KEYS.items():
                metric = legacy[stage][mode]
                rows.append(
                    {
                        "environment": environment,
                        "training_seed": training_seed,
                        "checkpoint_step": checkpoint_step,
                        "stage": stage,
                        "family": "legacy",
                        "mode": mode,
                        "label": label,
                        "mean_reward": metric["mean_reward"],
                        "std_reward": metric["std_reward"],
                        "goal_reaching_rate": metric["goal_reaching_rate"] / 100.0,
                    }
                )
            matched = data[
                (data.environment == environment)
                & (data.training_seed == training_seed)
                & (data.checkpoint_step == checkpoint_step)
                & data["mode"].isin(MODE_ORDER)
            ]
            if len(matched) != len(MODE_ORDER):
                raise SystemExit(f"Missing matched NAB modes for {environment}, {stage}")
            for _, metric in matched.iterrows():
                rows.append(
                    {
                        "environment": environment,
                        "training_seed": training_seed,
                        "checkpoint_step": checkpoint_step,
                        "stage": stage,
                        "family": "horizon-normalized",
                        "mode": metric["mode"],
                        "label": MODE_LABEL[metric["mode"]],
                        "mean_reward": metric["mean_reward"],
                        "std_reward": metric["std_reward"],
                        "goal_reaching_rate": metric["goal_reaching_rate"],
                    }
                )
    return pd.DataFrame(rows)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), constrained_layout=True)
    colors = {mode: color for mode, color in zip(MODE_ORDER, ["#2c7bb6", "#00a6ca", "#fdae61", "#d7191c"])}
    for axis, environment in zip(axes.flat, ENV_LABEL):
        group = summary[summary.environment == environment].set_index("mode").loc[MODE_ORDER]
        twin = axis.twinx()
        for mode, row in group.iterrows():
            axis.errorbar(row.target_nab, row.reward_gain_mean, yerr=row.reward_gain_ci95, marker="o", color=colors[mode], capsize=2)
        twin.plot(group.target_nab, group.goal_reaching_rate, color="#333333", marker="s", linestyle="--", linewidth=1, label="Goal-reaching rate")
        axis.axhline(0, color="0.7", linewidth=0.7)
        axis.set_title(ENV_LABEL[environment])
        axis.set_xlabel("Target normalized acceptance budget")
        axis.set_ylabel("Reward gain over backbone")
        twin.set_ylabel("Goal-reaching rate")
        twin.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker="o", color=colors[m], label=MODE_LABEL[m]) for m in MODE_ORDER]
    handles.append(plt.Line2D([0], [0], marker="s", linestyle="--", color="#333333", label="Goal-reaching rate"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(output, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.2), constrained_layout=True)
    for axis, environment in zip(axes, ENV_LABEL):
        group = summary[summary.environment == environment].set_index("mode").loc[MODE_ORDER]
        axis.plot(group.goal_reaching_rate, group.reward_gain_mean, color="0.6", linewidth=0.8)
        for mode, row in group.iterrows():
            axis.scatter(row.goal_reaching_rate, row.reward_gain_mean, s=24)
            near_right = row.goal_reaching_rate > 0.9
            axis.annotate(
                MODE_LABEL[mode].replace("Almost Open", "Open"),
                (row.goal_reaching_rate, row.reward_gain_mean),
                xytext=(-3 if near_right else 3, -7 if near_right else 3),
                textcoords="offset points",
                fontsize=6,
                ha="right" if near_right else "left",
            )
        axis.axhline(0, color="0.75", linewidth=0.7)
        axis.set_title(ENV_LABEL[environment])
        axis.set_xlabel("Goal-reaching rate")
        axis.margins(x=0.08, y=0.12)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Reward gain over backbone")
    fig.savefig(output, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def plot_legacy(matched: pd.DataFrame, output: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    markers = {"legacy": "o", "horizon-normalized": "s"}
    for axis, environment in zip(axes, ["Pendulum-v1", "CartpoleSwingupEnvLong-v0"]):
        group = matched[matched.environment == environment]
        for family, family_rows in group.groupby("family"):
            axis.scatter(
                family_rows.goal_reaching_rate,
                family_rows.mean_reward,
                marker=markers[family],
                alpha=0.8,
                label=family.replace("horizon-normalized", "NAB modes"),
            )
        axis.set_title(ENV_LABEL[environment])
        axis.set_xlabel("Goal-reaching rate")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Episode return")
    axes[0].legend(frameon=False)
    fig.savefig(output, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def write_table(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Horizon-normalized CALF-Wrapper modes over the complete checkpoint sweep. Reward differences and 95\% confidence intervals are paired against the bare backbone.}",
        r"\label{tab:nab_modes}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Environment & Mode & $T$ & Target NAB & $p_0$ & $\lambda$ & Empirical NAB & Fallback & $\Delta R$ (95\% CI) \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{ENV_LABEL[row['environment']]} & {MODE_LABEL[row['mode']]} & {int(row['horizon'])} & "
            f"{row['target_nab']:.2f} & {row['p0']:.1f} & {row['lambda']:.6f} & "
            f"{row['empirical_nab']:.3f} & {row['fallback_fraction']:.3f} & "
            f"{row['reward_gain_mean']:.1f} $\\pm$ {row['reward_gain_ci95']:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    output.write_text("\n".join(lines) + "\n")


def write_macros(summary: pd.DataFrame, output: Path) -> None:
    env_names = {
        "Pendulum-v1": "Pendulum",
        "CartpoleSwingupEnvLong-v0": "Cartpole",
        "UnderwaterDrone-v0": "Underwater",
        "RobotNavigationConstSpeedCatch-v0": "Robot",
    }
    mode_names = {mode: MODE_LABEL[mode].replace(" ", "") for mode in MODE_ORDER}
    lines = ["% Generated from the verified complete checkpoint sweep."]
    for _, row in summary.iterrows():
        stem = f"Nab{env_names[row['environment']]}{mode_names[row['mode']]}"
        lines.extend([
            f"\\newcommand{{\\{stem}Gain}}{{{row['reward_gain_mean']:.1f}}}",
            f"\\newcommand{{\\{stem}GainCI}}{{{row['reward_gain_ci95']:.2f}}}",
            f"\\newcommand{{\\{stem}Goal}}{{{row['goal_reaching_rate']:.3f}}}",
        ])
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--legacy-pendulum", type=Path, required=True)
    parser.add_argument("--legacy-cartpole", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=9654)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = canonicalize(args.results, args.expected_rows)
    summary = summarize(data)
    matched = matched_legacy(data, args.legacy_pendulum, args.legacy_cartpole)
    shutil.copyfile(args.results, args.output_dir / "calf_matrix_all_tasks.csv")
    data.to_csv(args.output_dir / "calf_checkpoint_mode_results.csv", index=False)
    summary.to_csv(args.output_dir / "calf_mode_summary.csv", index=False)
    matched.to_csv(args.output_dir / "calf_legacy_new_matched.csv", index=False)
    plot_sensitivity(summary, args.output_dir / "calf_nab_sensitivity.pdf")
    plot_tradeoff(summary, args.output_dir / "calf_reward_reliability_tradeoff.pdf")
    plot_legacy(matched, args.output_dir / "calf_legacy_new_modes.pdf")
    write_table(summary, args.output_dir / "calf_mode_table.tex")
    write_macros(summary, args.output_dir / "calf_numbers.tex")
    output_records = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "format": "calf-wrapper-nab-analysis-v1",
        "source": str(args.results),
        "source_sha256": sha256(args.results),
        "source_rows": len(pd.read_csv(args.results)),
        "canonical_checkpoint_mode_rows": len(data),
        "summary_rows": len(summary),
        "matched_legacy_new_rows": len(matched),
        "outputs": output_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
