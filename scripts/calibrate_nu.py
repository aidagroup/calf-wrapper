#!/usr/bin/env python3
"""Calibrate CALF's improvement threshold on fixed fallback trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from run.eval import goal_reaching_mask, load_model, make_env, presets, run_episode
from scripts.run_checkpoint_matrix import discover_checkpoints
from src.critic_values import critic_values
from src.nu_calibration import calibrate_nu


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PROTOCOL = REPO_ROOT / "experiments" / "checkpoint-sweep-v1.json"
DEFAULT_NU_PROTOCOL = REPO_ROOT / "experiments" / "nu-ablation-v1.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "run" / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_corrected_next(item: dict[str, Any]) -> np.ndarray:
    following = np.copy(item["next_obs"])
    for index, (done, info) in enumerate(zip(item["is_done"], item["info"])):
        if done and "terminal_observation" in info:
            following[index] = info["terminal_observation"]
    return following


def collect_fallback_transitions(
    preset: str,
    *,
    seed: int,
    trials: int,
) -> dict[str, np.ndarray]:
    config = replace(presets[preset][1], seed=seed, n_envs=trials)
    env = DummyVecEnv(
        [
            make_env(config.env_id, rank, seed, wrapper_class=None, wrapper_kwargs={})
            for rank in range(trials)
        ]
    )
    env.seed(seed)
    episode = run_episode(
        config.stabilizing_policy.get_action,
        env,
        config.n_steps,
    )

    current_parts = []
    next_parts = []
    reached = np.zeros(trials, dtype=bool)
    for item in episode:
        active = np.asarray(item["active"], dtype=bool) & ~reached
        current = np.asarray(item["obs"])
        following = terminal_corrected_next(item)
        next_goal = goal_reaching_mask(config.env_id, following)
        current_parts.append(current[active])
        next_parts.append(following[active])
        reached[active & next_goal] = True
        if np.all(reached):
            break
    current = np.concatenate(current_parts)
    following = np.concatenate(next_parts)
    return {
        "current_observations": current,
        "next_observations": following,
        "current_goal_mask": goal_reaching_mask(config.env_id, current),
        "next_goal_mask": goal_reaching_mask(config.env_id, following),
    }


def batched_critic_values(model: Any, observations: np.ndarray) -> np.ndarray:
    chunks = [
        critic_values(model, observations[start : start + 8192])
        for start in range(0, len(observations), 8192)
    ]
    return np.concatenate(chunks).astype(np.float64)


def selected_checkpoints(
    checkpoint_protocol: dict[str, Any],
    artifacts_root: Path,
    selection_csv: Path | None,
) -> list[tuple[str, int, int, Path, str]]:
    requested: set[tuple[str, int, int]] | None = None
    stages: dict[tuple[str, int, int], str] = {}
    if selection_csv is not None:
        requested = set()
        with selection_csv.open(newline="") as source:
            for row in csv.DictReader(source):
                key = (
                    row["environment"],
                    int(row["training_seed"]),
                    int(row["checkpoint_step"]),
                )
                requested.add(key)
                stages[key] = row["reward_stage"]

    selected = []
    for preset, config in checkpoint_protocol["environments"].items():
        for seed, step, path in discover_checkpoints(config, artifacts_root):
            key = (config["env_id"], seed, step)
            if requested is None or key in requested:
                selected.append((preset, seed, step, path, stages.get(key, "all")))
    if requested is not None:
        found = {
            (checkpoint_protocol["environments"][preset]["env_id"], seed, step)
            for preset, seed, step, _, _ in selected
        }
        missing = requested - found
        if missing:
            raise RuntimeError(
                f"missing {len(missing)} selected checkpoints: {sorted(missing)[:3]}"
            )
    return selected


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-protocol", type=Path, default=DEFAULT_CHECKPOINT_PROTOCOL
    )
    parser.add_argument("--nu-protocol", type=Path, default=DEFAULT_NU_PROTOCOL)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate-n",
        help="Optional comma-separated override used after development selection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_protocol = json.loads(args.checkpoint_protocol.read_text())
    nu_protocol = json.loads(args.nu_protocol.read_text())
    calibration = nu_protocol["calibration"]
    fallback_seed = int(calibration["fallback_seed"])
    fallback_trials = int(calibration["fallback_trials"])
    candidates = (
        [float(item) for item in args.candidate_n.split(",")]
        if args.candidate_n
        else [float(item) for item in calibration["candidate_n"]]
    )
    checkpoints = selected_checkpoints(
        checkpoint_protocol,
        args.artifacts_root,
        args.selection_csv,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_dir = args.output_dir / "fallback_transitions"
    transition_dir.mkdir(exist_ok=True)
    rows = []
    transition_manifest = {}
    by_preset: dict[str, list[tuple[str, int, int, Path, str]]] = {}
    for checkpoint in checkpoints:
        by_preset.setdefault(checkpoint[0], []).append(checkpoint)

    for preset, preset_checkpoints in by_preset.items():
        env_config = checkpoint_protocol["environments"][preset]
        transitions = collect_fallback_transitions(
            preset,
            seed=fallback_seed,
            trials=fallback_trials,
        )
        transition_path = transition_dir / f"{preset}.npz"
        np.savez_compressed(transition_path, **transitions)
        transition_manifest[preset] = {
            "path": str(transition_path.relative_to(args.output_dir)),
            "sha256": sha256(transition_path),
            "transitions": int(len(transitions["current_observations"])),
            "goal_neighborhood_transitions": int(
                np.sum(
                    ~transitions["current_goal_mask"] & transitions["next_goal_mask"]
                )
            ),
        }

        for index, (_, seed, step, path, stage) in enumerate(preset_checkpoints, 1):
            model_config = replace(
                presets[preset][1],
                model_path=path,
                device=args.device,
            )
            model = load_model(model_config)
            current_values = batched_critic_values(
                model, transitions["current_observations"]
            )
            next_values = batched_critic_values(model, transitions["next_observations"])
            for n in candidates:
                result = calibrate_nu(
                    current_values,
                    next_values,
                    transitions["current_goal_mask"],
                    transitions["next_goal_mask"],
                    horizon=int(env_config["episode_horizon"]),
                    n=n,
                )
                common = {
                    "environment": env_config["env_id"],
                    "preset": preset,
                    "algorithm": env_config["algorithm"],
                    "training_seed": seed,
                    "checkpoint_step": step,
                    "reward_stage": stage,
                    "checkpoint_path": str(path),
                    **asdict(result),
                }
                rows.append({**common, "rule_variant": "goal_guarded_max"})
                rows.append(
                    {
                        **common,
                        "nu": result.range_increment,
                        "rule_variant": "trajectory_scale",
                    }
                )
            print(
                f"{preset}: calibrated {index}/{len(preset_checkpoints)} "
                f"seed={seed} step={step}",
                flush=True,
            )
            del model

    output_csv = args.output_dir / "nu_calibration.csv"
    write_csv(rows, output_csv)
    manifest = {
        "protocol_id": nu_protocol["protocol_id"],
        "checkpoint_protocol": str(args.checkpoint_protocol),
        "nu_protocol": str(args.nu_protocol),
        "selection_csv": str(args.selection_csv) if args.selection_csv else None,
        "device": args.device,
        "checkpoints": len(checkpoints),
        "rows": len(rows),
        "fallback_seed": fallback_seed,
        "fallback_trials": fallback_trials,
        "nu_calibration_sha256": sha256(output_csv),
        "fallback_transitions": transition_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
