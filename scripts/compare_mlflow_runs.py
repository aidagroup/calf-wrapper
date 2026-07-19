#!/usr/bin/env python3
"""Compare deterministic MLflow metric histories from two training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from mlflow import MlflowClient

DEFAULT_IGNORED_METRICS = {"charts/SPS"}


def metric_history(
    client: MlflowClient,
    tracking_uri: str,
    run_id: str,
    key: str,
    max_step: int | None,
) -> list[tuple[int, float]]:
    if tracking_uri.startswith(("http://", "https://")):
        query = urlencode({"run_id": run_id, "metric_key": key})
        url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/metrics/get-history?{query}"
        with urlopen(url, timeout=180) as response:
            points = json.load(response).get("metrics", [])
        history = [
            (int(point["step"]), float(point["value"]))
            for point in points
            if max_step is None or int(point["step"]) <= max_step
        ]
    else:
        points = client.get_metric_history(run_id, key)
        history = [
            (point.step, point.value)
            for point in points
            if max_step is None or point.step <= max_step
        ]
    return sorted(history)


def compare_runs(
    *,
    reference_tracking_uri: str,
    reference_run_id: str,
    actual_tracking_uri: str,
    actual_run_id: str,
    max_step: int | None,
    ignored_metrics: set[str],
    selected_metrics: set[str] | None = None,
) -> dict:
    reference_client = MlflowClient(tracking_uri=reference_tracking_uri)
    actual_client = MlflowClient(tracking_uri=actual_tracking_uri)
    reference_run = reference_client.get_run(reference_run_id)
    actual_run = actual_client.get_run(actual_run_id)

    reference_keys = set(reference_run.data.metrics) - ignored_metrics
    actual_keys = set(actual_run.data.metrics) - ignored_metrics
    metric_keys = sorted(
        selected_metrics
        if selected_metrics is not None
        else reference_keys | actual_keys
    )
    comparisons = {}
    for key in metric_keys:
        reference = metric_history(
            reference_client,
            reference_tracking_uri,
            reference_run_id,
            key,
            max_step,
        )
        actual = metric_history(
            actual_client, actual_tracking_uri, actual_run_id, key, max_step
        )
        first_mismatch = None
        for index, (reference_point, actual_point) in enumerate(zip(reference, actual)):
            if reference_point != actual_point:
                first_mismatch = {
                    "index": index,
                    "reference": reference_point,
                    "actual": actual_point,
                }
                break
        if first_mismatch is None and len(reference) != len(actual):
            first_mismatch = {
                "index": min(len(reference), len(actual)),
                "reference": (
                    reference[len(actual)] if len(reference) > len(actual) else None
                ),
                "actual": (
                    actual[len(reference)] if len(actual) > len(reference) else None
                ),
            }
        comparisons[key] = {
            "reference_points": len(reference),
            "actual_points": len(actual),
            "exact": reference == actual,
            "first_mismatch": first_mismatch,
        }

    return {
        "reference": {
            "tracking_uri": reference_tracking_uri,
            "run_id": reference_run_id,
            "git_commit": reference_run.data.tags.get("mlflow.source.git.commit"),
        },
        "actual": {
            "tracking_uri": actual_tracking_uri,
            "run_id": actual_run_id,
            "git_commit": actual_run.data.tags.get("mlflow.source.git.commit"),
        },
        "max_step": max_step,
        "ignored_metrics": sorted(ignored_metrics),
        "selected_metrics": (
            sorted(selected_metrics) if selected_metrics is not None else None
        ),
        "all_exact": all(item["exact"] for item in comparisons.values()),
        "metrics": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tracking-uri", required=True)
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument("--actual-tracking-uri", required=True)
    parser.add_argument("--actual-run-id", required=True)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--metric", action="append")
    parser.add_argument("--ignore-metric", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_runs(
        reference_tracking_uri=args.reference_tracking_uri,
        reference_run_id=args.reference_run_id,
        actual_tracking_uri=args.actual_tracking_uri,
        actual_run_id=args.actual_run_id,
        max_step=args.max_step,
        ignored_metrics=DEFAULT_IGNORED_METRICS | set(args.ignore_metric),
        selected_metrics=set(args.metric) if args.metric else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_exact": result["all_exact"], "output": str(args.output)}))
    if not result["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
