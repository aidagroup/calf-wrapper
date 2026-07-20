"""Aggregate SOOPER screening results and freeze the held-out protocol."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CHECKPOINT = re.compile(r"td3_UnderwaterDrone-v0_(\d+).+?checkpoint_(\d+)_steps")


def ci95(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(1.96 * array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def identify(path: str) -> tuple[int, int]:
    match = CHECKPOINT.search(path)
    if match is None:
        raise ValueError(f"Cannot identify underwater checkpoint: {path}")
    return int(match.group(1)), int(match.group(2))


def collect(result_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    summaries = []
    trials = []
    for path in sorted((result_root / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        config = summary["config"]
        checkpoint_seed, checkpoint_step = identify(config["model_path"])
        run_dir = path.parent
        raw = pd.read_csv(run_dir / "raw" / "evaluation_trials.csv")
        final_iteration = raw["iteration"].max()
        final = raw[raw["iteration"] == final_iteration]
        for row in final.to_dict("records"):
            row.update(
                {
                    "checkpoint_training_seed": checkpoint_seed,
                    "checkpoint_step": checkpoint_step,
                    "sooper_training_seed": config["seed"],
                    "online_iterations": config["online_iterations"],
                    "real_interactions": summary["real_interactions"],
                    "offline_prior_interactions": summary["offline_prior_interactions"],
                    "wall_clock_seconds": summary["wall_clock_seconds"],
                    "cost_budget": summary["cost_budget"],
                    "mlflow_run_id": summary["mlflow_run_id"],
                    "run_dir": str(run_dir.relative_to(result_root)),
                }
            )
            trials.append(row)
        summaries.append(summary)
    return pd.DataFrame(trials), summaries


def aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["checkpoint_training_seed", "checkpoint_step", "online_iterations"]
    for key, group in trials.groupby(keys):
        rows.append(
            {
                **dict(zip(keys, key)),
                "sooper_training_seeds": ",".join(map(str, sorted(group.sooper_training_seed.unique()))),
                "n_trials": len(group),
                "mean_reward": group.episode_return.mean(),
                "std_reward": group.episode_return.std(ddof=1),
                "reward_ci95_half_width": ci95(group.episode_return),
                "goal_reaching_rate": group.goal_reached.mean(),
                "constraint_satisfaction_rate": group.constraint_satisfied.mean(),
                "intervention_fraction": group.intervention_fraction.mean(),
                "cost_budget": group.cost_budget.iloc[0],
                "real_interactions_per_training_seed": group.real_interactions.iloc[0],
                "offline_interactions_per_training_seed": group.offline_prior_interactions.iloc[0],
                "mean_wall_clock_seconds": group.groupby("sooper_training_seed").wall_clock_seconds.first().mean(),
                "gpu_hours_per_training_seed": group.groupby("sooper_training_seed").wall_clock_seconds.first().mean() / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calf-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    shortlist = json.loads(args.shortlist.read_text())["candidates"]
    expected = (
        len(shortlist)
        * len(protocol["sooper_screening"]["training_seeds"])
        * len(protocol["sooper_screening"]["online_iterations"])
    )
    failures = list((args.result_root / "failures").glob("*.json"))
    if failures:
        raise SystemExit(f"Screening contains {len(failures)} failures")
    trials, summaries = collect(args.result_root)
    if len(summaries) != expected:
        raise SystemExit(f"Expected {expected} completed SOOPER tasks, found {len(summaries)}")
    grouped = aggregate(trials)
    feasible = grouped[
        (grouped.constraint_satisfaction_rate >= protocol["sooper_screening"]["constraint_satisfaction_rate_min"])
        & (grouped.goal_reaching_rate >= protocol["sooper_screening"]["goal_reaching_rate_min"])
    ]
    selected_settings = []
    for candidate in shortlist:
        subset = feasible[
            (feasible.checkpoint_training_seed == candidate["training_seed"])
            & (feasible.checkpoint_step == candidate["checkpoint_step"])
        ]
        if subset.empty:
            selected_setting = grouped[
                (grouped.checkpoint_training_seed == candidate["training_seed"])
                & (grouped.checkpoint_step == candidate["checkpoint_step"])
            ].sort_values(
                ["constraint_satisfaction_rate", "goal_reaching_rate", "mean_reward"],
                ascending=False,
            ).iloc[0]
        else:
            selected_setting = subset.sort_values("mean_reward", ascending=False).iloc[0]
        selected_settings.append(selected_setting.to_dict() | candidate)
    scenarios = pd.DataFrame(selected_settings)
    scenarios["calf_sooper_gap"] = scenarios.calf_reward - scenarios.mean_reward
    scenarios["sooper_backbone_gap"] = scenarios.mean_reward - scenarios.base_reward
    scenarios["minimum_adjacent_gap"] = scenarios[["calf_sooper_gap", "sooper_backbone_gap"]].min(axis=1)
    ordered = scenarios[
        (scenarios.calf_sooper_gap > 0)
        & (scenarios.sooper_backbone_gap > 0)
        & (scenarios.calf_goal_reaching_rate >= scenarios.goal_reaching_rate)
    ]
    if ordered.empty:
        chosen = scenarios.sort_values("minimum_adjacent_gap", ascending=False).iloc[0]
        positioning = "strongest transparent Pareto/efficiency comparison"
    else:
        chosen = ordered.sort_values("minimum_adjacent_gap", ascending=False).iloc[0]
        positioning = "CALF-Wrapper > SOOPER > bare backbone in development reward"
    calf = pd.read_csv(args.calf_results)
    calf_row = calf[
        (calf.environment == "UnderwaterDrone-v0")
        & (calf.training_seed == chosen.training_seed)
        & (calf.checkpoint_step == chosen.checkpoint_step)
        & (calf["mode"] == chosen.selected_calf_mode)
    ].iloc[0]
    selected_runs = trials[
        (trials.checkpoint_training_seed == chosen.checkpoint_training_seed)
        & (trials.checkpoint_step == chosen.checkpoint_step)
        & (trials.online_iterations == chosen.online_iterations)
    ][["sooper_training_seed", "run_dir", "mlflow_run_id"]].drop_duplicates().sort_values("sooper_training_seed")
    held_out = protocol["held_out_confirmation"]
    frozen = {
        "format": "calf-wrapper-sooper-frozen-held-out-protocol-v1",
        "source_protocol": str(args.protocol),
        "source_shortlist": str(args.shortlist),
        "source_screening_root": str(args.result_root),
        "positioning_selected_on_development_data": positioning,
        "environment": "underwater-drone",
        "env_id": "UnderwaterDrone-v0",
        "algorithm": "cleanrl_td3",
        "base_checkpoint": (
            f"run/artifacts/td3_UnderwaterDrone-v0_{int(chosen.checkpoint_training_seed)}/checkpoints/"
            f"td3_checkpoint_{int(chosen.checkpoint_step)}_steps.pt"
        ),
        "checkpoint_training_seed": int(chosen.checkpoint_training_seed),
        "checkpoint_step": int(chosen.checkpoint_step),
        "cost_budget": float(chosen.cost_budget),
        "calf": {
            "mode": chosen.selected_calf_mode,
            "target_nab": float(calf_row.calf_target_acceptance_budget),
            "p0": float(calf_row.calf_relaxprob_init),
            "lambda": float(calf_row.calf_relaxprob_factor),
            "development_reward": float(chosen.calf_reward),
            "development_goal_reaching_rate": float(chosen.calf_goal_reaching_rate),
            "development_mlflow_run_id": calf_row.mlflow_run_id,
        },
        "backbone": {
            "development_reward": float(chosen.base_reward),
            "development_goal_reaching_rate": float(chosen.base_goal_reaching_rate),
        },
        "sooper": {
            "online_iterations": int(chosen.online_iterations),
            "real_interactions_per_training_seed": int(chosen.real_interactions_per_training_seed),
            "offline_interactions_per_training_seed": int(chosen.offline_interactions_per_training_seed),
            "development_reward": float(chosen.mean_reward),
            "development_goal_reaching_rate": float(chosen.goal_reaching_rate),
            "development_constraint_satisfaction_rate": float(chosen.constraint_satisfaction_rate),
            "development_intervention_fraction": float(chosen.intervention_fraction),
            "runs": selected_runs.to_dict("records"),
        },
        "held_out_seeds": list(range(held_out["seed_start"], held_out["seed_stop_inclusive"] + 1)),
        "retuning_after_held_out_observation": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "sooper_screening_trials.csv", index=False)
    grouped.to_csv(args.output_dir / "sooper_screening_summary.csv", index=False)
    scenarios.to_csv(args.output_dir / "sooper_scenario_candidates.csv", index=False)
    (args.output_dir / "frozen_held_out_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
