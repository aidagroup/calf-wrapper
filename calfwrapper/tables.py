"""LaTeX tables generated from reproduced evaluation results."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from calfwrapper.operating_modes import OPERATING_MODES
from calfwrapper.paths import ROOT
from calfwrapper.results import Result
from calfwrapper.statistics import STAGES, ReturnTest

ENVIRONMENTS = (
    ("pendulum", "Pendulum"),
    ("cartpole", "CartPole"),
    ("auv", "AUV"),
    ("robot", "Robot"),
)
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
MODE_LABELS = {method: label for method, label in METHODS}
RUNTIME_ENVIRONMENTS = (
    ("Pendulum-v1", r"\shortstack{Pendulum\\(PPO)}"),
    ("CartpoleSwingupEnvLong-v0", r"\shortstack{CartPole\\(PPO)}"),
    ("UnderwaterDrone-v0", r"\shortstack{AUV\\(TD3)}"),
    ("RobotNavigationConstSpeedCatch-v0", r"\shortstack{Robot\\(TD3)}"),
)
RUNTIME_MODE_LABELS = {
    "conservative": "Conserv.",
    "guarded": "Guarded",
    "moderate": "Moderate",
    "balanced": "Balanced",
    "high": "High",
    "almost_open": "Almost-open",
}


def _save(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _p_value(value: float) -> str:
    if value == 0:
        body = "0"
    elif value < 0.001:
        exponent = math.floor(math.log10(value))
        body = rf"{value / 10**exponent:.2f}\times10^{{{exponent}}}"
    else:
        body = f"{value:.3f}".rstrip("0").rstrip(".")
    return rf"\mathbf{{{body}}}" if value < 0.05 else body


def main_results(rows: list[Result], destination: Path) -> None:
    """Write the central episode-return and goal-reaching-rate table."""

    index = {(row["environment"], row["checkpoint"], row["method"]): row for row in rows}
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Episode returns and goal-reaching rates at the Low-, Mid-,",
        r"    and High-return checkpoints in each environment.}",
        r"  \label{tab:main_results}",
        r"  \scriptsize",
        r"  \begin{tabular}{llrrrrrrrrr}",
        r"    \toprule",
        r"    Environment & Checkpoint & \multicolumn{9}{c}{Method} \\",
        "    & & " + " & ".join(label for _, label in METHODS) + r" \\",
        r"    \midrule",
    ]
    for environment_index, (environment, label) in enumerate(ENVIRONMENTS):
        if environment_index:
            lines.append(r"    \midrule")
        for stage_index, stage in enumerate(STAGES):
            values = []
            for method, _ in METHODS:
                row = index[(environment, stage, method)]
                precision = 3 if environment == "robot" and stage == "High" else 2
                values.append(
                    rf"${row['mean_episode_return']:.{precision}f}"
                    rf"\pm{row['return_ci95_half_width']:.{precision}f}$; "
                    rf"${row['goal_reaching_rate']:.0f}\%$"
                )
            lines.append(
                "    " + " & ".join((label if stage_index == 0 else "", stage, *values)) + r" \\"
            )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"])
    _save(destination, lines)


def _describe(rows: list[ReturnTest]) -> str:
    parts = []
    for environment, label in ENVIRONMENTS:
        modes = [
            MODE_LABELS[mode]
            for mode in OPERATING_MODES
            if any(row["environment"] == environment and row["mode"] == mode for row in rows)
        ]
        if modes:
            parts.append(f"{label}: {', '.join(modes)}")
    return "; ".join(parts) + "." if parts else "None."


def comparison_summary(
    tests: list[ReturnTest],
    reference: str,
    destination: Path,
) -> None:
    """Write the compact paired-comparison table for one reference policy."""

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Paired return comparisons with the {reference} policy.}}",
        rf"  \label{{tab:comparisons_with_{reference}}}",
        r"  \scriptsize",
        r"  \begin{tabularx}{\columnwidth}{llcX}",
        r"    \toprule",
        r"    Checkpoint & Statistical result & Count & Environment and operating modes \\",
        r"    \midrule",
    ]
    for stage_index, stage in enumerate(STAGES):
        if stage_index:
            lines.append(r"    \midrule")
        selected = [
            row for row in tests if row["family"] == f"calfwrapper_vs_{reference}_{stage.lower()}"
        ]
        categories = (
            (
                "Significant; mean paired difference positive",
                [
                    row
                    for row in selected
                    if row["significant"] and row["mean_paired_difference"] > 0
                ],
            ),
            (
                "Significant; mean paired difference negative",
                [
                    row
                    for row in selected
                    if row["significant"] and row["mean_paired_difference"] < 0
                ],
            ),
            ("No significant difference", [row for row in selected if not row["significant"]]),
        )
        for description, category in categories:
            lines.append(
                f"    {stage} & {description} & {len(category)}/24 & {_describe(category)} " + r"\\"
            )
    lines.extend([r"    \bottomrule", r"  \end{tabularx}", r"\end{table}"])
    _save(destination, lines)


def adjusted_p_values(tests: list[ReturnTest], destination: Path) -> None:
    """Write all adjusted two-sided p-values reported in Table 10."""

    index = {(row["family"], row["environment"], row["stage"], row["mode"]): row for row in tests}
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Holm-adjusted p-values for the paired comparisons with",
        r"    the fallback policy and the base policy.}",
        r"  \label{tab:all_adjusted_p_values}",
        r"  \scriptsize",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c}{Fallback policy} & \multicolumn{3}{c}{Base policy} \\",
        r"    Environment & Mode & Low & Mid & High & Low & Mid & High \\",
        r"    \midrule",
    ]
    for environment_index, (environment, label) in enumerate(ENVIRONMENTS):
        if environment_index:
            lines.append(r"    \midrule")
        for mode_index, mode in enumerate(OPERATING_MODES):
            values = []
            for reference in ("fallback", "base"):
                for stage in STAGES:
                    row = index[
                        (
                            f"calfwrapper_vs_{reference}_{stage.lower()}",
                            environment,
                            stage,
                            mode,
                        )
                    ]
                    values.append(f"${_p_value(row['holm_adjusted_p_value'])}$")
            lines.append(
                "    "
                + " & ".join((label if mode_index == 0 else "", MODE_LABELS[mode], *values))
                + r" \\"
            )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"])
    _save(destination, lines)


def intersection_union_p_values(tests: list[ReturnTest], destination: Path) -> None:
    """Write all adjusted intersection-union p-values reported in Table 11."""

    index = {(row["family"], row["environment"], row["stage"], row["mode"]): row for row in tests}
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Holm-adjusted p-values for tests of a return higher than",
        r"    both constituent policies.}",
        r"  \label{tab:intersection_union_p_values}",
        r"  \scriptsize",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Environment & Mode & Low & Mid & High \\",
        r"    \midrule",
    ]
    for environment_index, (environment, label) in enumerate(ENVIRONMENTS):
        if environment_index:
            lines.append(r"    \midrule")
        for mode_index, mode in enumerate(OPERATING_MODES):
            values = []
            for stage in STAGES:
                row = index[
                    (
                        f"calfwrapper_above_both_{stage.lower()}",
                        environment,
                        stage,
                        mode,
                    )
                ]
                values.append(f"${_p_value(row['holm_adjusted_p_value'])}$")
            lines.append(
                "    "
                + " & ".join((label if mode_index == 0 else "", MODE_LABELS[mode], *values))
                + r" \\"
            )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    _save(destination, lines)


def runtime_overhead(source: Path, destination: Path) -> None:
    """Write the latency table from the committed native benchmark measurements."""

    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {(row["environment"], row["mode"]): row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Measured mean per-step latency of the native Rust implementation",
        r"    on an NVIDIA GeForce RTX 3090 GPU.}",
        r"  \label{tab:runtime_overhead}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{1.3pt}",
        r"  \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{llrrrrrr}",
        r"      \toprule",
        (
            r"      Env. & Mode & Fallback (\%) & Base ($\mu$s) & Critic ($\mu$s) & "
            r"Other ($\mu$s) & Wrapper ($\mu$s) & Added ($\mu$s; 95\% CI) \\"
        ),
        r"      \midrule",
    ]
    for environment_index, (environment, environment_label) in enumerate(RUNTIME_ENVIRONMENTS):
        if environment_index:
            lines.append(r"      \midrule")
        environment_rows = [index[(environment, mode)] for mode in OPERATING_MODES]
        base_mean = sum(float(row["base_latency_us"]) for row in environment_rows) / 6
        critic_mean = sum(float(row["critic_latency_us"]) for row in environment_rows) / 6
        for mode_index, mode in enumerate(OPERATING_MODES):
            row = index[(environment, mode)]
            label = rf"\multirow{{6}}{{*}}{{{environment_label}}}" if mode_index == 0 else ""
            base = rf"\multirow{{6}}{{*}}{{{base_mean:.1f}}}" if mode_index == 0 else ""
            critic = rf"\multirow{{6}}{{*}}{{{critic_mean:.1f}}}" if mode_index == 0 else ""
            fallback = 100 * int(row["fallback_calls"]) / int(row["calls_per_block"])
            lines.append(
                "      "
                + f"{label} & {RUNTIME_MODE_LABELS[mode]} & {fallback:.1f} & "
                + f"{base} & {critic} & {float(row['native_logic_latency_us']):.3f} & "
                + f"{float(row['wrapped_latency_us']):.1f} & "
                + f"{float(row['added_latency_us']):.1f} "
                + f"[{float(row['added_latency_ci95_low_us']):.1f}, "
                + f"{float(row['added_latency_ci95_high_us']):.1f}] \\\\"
            )
    lines.extend([r"      \bottomrule", r"    \end{tabular}%", r"  }", r"\end{table}"])
    _save(destination, lines)


def all_tables(
    results: list[Result],
    tests: list[ReturnTest],
    destination: Path,
) -> None:
    main_results(results, destination / "table-6-main-results.tex")
    comparison_summary(tests, "fallback", destination / "table-7-fallback-comparisons.tex")
    comparison_summary(tests, "base", destination / "table-8-base-comparisons.tex")
    adjusted_p_values(tests, destination / "table-10-adjusted-p-values.tex")
    intersection_union_p_values(tests, destination / "table-11-intersection-union-p-values.tex")
    runtime_overhead(
        ROOT / "reference" / "runtime" / "summary.csv",
        destination / "table-9-runtime-overhead.tex",
    )
