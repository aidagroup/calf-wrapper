#!/usr/bin/env python3
"""Compare aggregate reproduction metrics with reference JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPRODUCED_METHODS = {"fallback", "conservative", "balanced", "brave", "base"}


def compare(reference: dict, actual: dict, absolute_tolerance: float) -> list[dict]:
    rows = []
    for stage, expected in reference["stages"].items():
        observed = actual.get("stages", {}).get(stage)
        rows.append(
            {
                "section": "stages",
                "method": stage,
                "metric": "checkpoint_step",
                "reference": expected,
                "actual": observed,
                "absolute_delta": (
                    abs(observed - expected) if observed is not None else None
                ),
                "exact": observed == expected,
                "pass": observed == expected,
            }
        )
    for section, methods in reference.items():
        if section == "stages":
            continue
        methods = {"fallback": methods} if section == "fallback" else methods
        actual_methods = (
            {"fallback": actual[section]} if section == "fallback" else actual[section]
        )
        for method, reference_metrics in methods.items():
            # Residual RL is a plot-only external baseline. This repository has
            # neither its implementation nor checkpoints, so the exporter may
            # carry its archived values through, but it must never be counted
            # as a reproduced CALF-Wrapper result.
            if method not in REPRODUCED_METHODS:
                continue
            for metric, expected in reference_metrics.items():
                observed = actual_methods.get(method, {}).get(metric)
                if observed is None:
                    rows.append(
                        {
                            "section": section,
                            "method": method,
                            "metric": metric,
                            "reference": expected,
                            "actual": None,
                            "absolute_delta": None,
                            "exact": False,
                            "pass": False,
                        }
                    )
                    continue
                delta = observed - expected
                rows.append(
                    {
                        "section": section,
                        "method": method,
                        "metric": metric,
                        "reference": expected,
                        "actual": observed,
                        "absolute_delta": abs(delta),
                        "exact": observed == expected,
                        "pass": math.isclose(
                            observed, expected, rel_tol=0.0, abs_tol=absolute_tolerance
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    report = {}
    for env_name in ("pendulum", "cartpole"):
        reference = json.loads((args.reference_dir / f"{env_name}.json").read_text())
        actual = json.loads((args.actual_dir / f"{env_name}.json").read_text())
        rows = compare(reference, actual, args.absolute_tolerance)
        report[env_name] = {
            "verdict": "PASS" if all(row["pass"] for row in rows) else "FAIL",
            "exact": all(row["exact"] for row in rows),
            "comparison_count": len(rows),
            "excluded_reference_methods": {
                "residual": (
                    "External plot-only baseline; no implementation or checkpoint "
                    "is present in this repository."
                )
            },
            "comparisons": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({env: values["verdict"] for env, values in report.items()}))
    if any(values["verdict"] != "PASS" for values in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
