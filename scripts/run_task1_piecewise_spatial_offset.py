#!/usr/bin/env python3
"""Evaluate deployable prediction-routed piecewise spatial offsets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_task1_hybrid_robustness as hybrid_slices  # noqa: E402
import run_task1_mean_offset as baseline  # noqa: E402
import run_task1_robust_offset as robust  # noqa: E402
import run_task1_stratified_target_coverage as stratified  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--robust-results", required=True, type=Path)
    parser.add_argument("--hybrid-slice-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def prediction_eccentricity_deg(item: baseline.Observation) -> float:
    vector = baseline.pitchyaw_to_vector(item.pred_pitch, item.pred_yaw)
    return math.degrees(math.acos(max(-1.0, min(1.0, vector[2]))))


def fit_region_offsets(
    selected: list[baseline.Observation],
    *,
    central_max: float,
    mid_max: float,
    delta: float,
    maximum_iterations: int,
    tolerance: float,
) -> dict[str, tuple[float, float]]:
    regions: dict[str, list[baseline.Observation]] = defaultdict(list)
    for item in selected:
        name = hybrid_slices.slice_name(
            hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
        )
        regions[name].append(item)
    offsets = {}
    for name in stratified.SLICE_ORDER:
        if not regions[name]:
            raise ValueError("A calibration region is empty")
        pitch, yaw, _ = robust.robust_offsets(
            regions[name],
            delta=delta,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        offsets[name] = (pitch, yaw)
    return offsets


def routed_error(
    item: baseline.Observation,
    offsets: dict[str, tuple[float, float]],
    *,
    central_max: float,
    mid_max: float,
) -> float:
    route = hybrid_slices.slice_name(
        prediction_eccentricity_deg(item), central_max, mid_max
    )
    pitch, yaw = offsets[route]
    return baseline.angular_error_deg(
        item.pred_pitch + pitch,
        item.pred_yaw + yaw,
        item.target_pitch,
        item.target_yaw,
    )


def evaluate(
    splits: dict[str, baseline.SubjectSplit],
    deltas: dict[tuple[str, int], float],
    hybrid_slice: dict[tuple[str, int, str], float],
    hybrid_overall: dict[tuple[str, int], float],
    *,
    budgets: list[int],
    central_max: float,
    mid_max: float,
    maximum_iterations: int,
    tolerance: float,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for subject, split in sorted(splits.items()):
        true_slices: dict[str, list[baseline.Observation]] = defaultdict(list)
        for item in split.evaluation:
            name = hybrid_slices.slice_name(
                hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
            )
            true_slices[name].append(item)
        for budget in budgets:
            selected = stratified.stratified_sample(
                split.candidate,
                budget,
                central_max=central_max,
                mid_max=mid_max,
            )
            offsets = fit_region_offsets(
                selected,
                central_max=central_max,
                mid_max=mid_max,
                delta=deltas[(subject, budget)],
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            all_error = statistics.fmean(
                routed_error(
                    item,
                    offsets,
                    central_max=central_max,
                    mid_max=mid_max,
                )
                for item in split.evaluation
            )
            rows.append(
                {
                    "subject": subject,
                    "budget": budget,
                    "slice": "overall",
                    "calibrated_mean_deg": all_error,
                    "huber_reference_deg": hybrid_overall[(subject, budget)],
                    "advantage_over_huber_deg": hybrid_overall[(subject, budget)]
                    - all_error,
                }
            )
            for name, observations in sorted(true_slices.items()):
                error = statistics.fmean(
                    routed_error(
                        item,
                        offsets,
                        central_max=central_max,
                        mid_max=mid_max,
                    )
                    for item in observations
                )
                reference = hybrid_slice[(subject, budget, name)]
                rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "slice": name,
                        "calibrated_mean_deg": error,
                        "huber_reference_deg": reference,
                        "advantage_over_huber_deg": reference - error,
                    }
                )

    aggregates = []
    for budget in budgets:
        for name in ("overall", *stratified.SLICE_ORDER):
            matching = [
                row
                for row in rows
                if row["budget"] == budget and row["slice"] == name
            ]
            differences = [
                float(row["advantage_over_huber_deg"]) for row in matching
            ]
            low, high = baseline.bootstrap_mean_ci(
                differences,
                replicates=replicates,
                seed=baseline.stable_seed(seed, "piecewise", budget, name),
            )
            aggregates.append(
                {
                    "budget": budget,
                    "slice": name,
                    "subjects": len(matching),
                    "subject_mean_calibrated_deg": statistics.fmean(
                        float(row["calibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_huber_reference_deg": statistics.fmean(
                        float(row["huber_reference_deg"]) for row in matching
                    ),
                    "advantage_over_huber_deg": statistics.fmean(differences),
                    "advantage_ci95_low_deg": low,
                    "advantage_ci95_high_deg": high,
                }
            )
    return rows, aggregates


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    slices_config = config["slices"]
    budgets = [int(value) for value in experiment["budgets"]]
    observations = baseline.load_unique_observations(args.predictions)
    splits = {
        subject: baseline.split_subject(
            subject,
            values,
            candidate_fraction=float(experiment["candidate_fraction"]),
            guard_fraction=float(experiment["guard_fraction"]),
        )
        for subject, values in observations.items()
    }
    hybrid_slice, hybrid_overall = stratified.load_hybrid_references(
        args.hybrid_slice_results / "per_subject_slice_metrics.csv",
        args.robust_results / "per_subject_metrics.csv",
    )
    rows, aggregates = evaluate(
        splits,
        stratified.load_selected_deltas(args.robust_results / "tuning.csv", budgets),
        hybrid_slice,
        hybrid_overall,
        budgets=budgets,
        central_max=float(slices_config["central_max_deg"]),
        mid_max=float(slices_config["mid_max_deg"]),
        maximum_iterations=int(experiment["maximum_iterations"]),
        tolerance=float(experiment["tolerance"]),
        replicates=int(experiment["bootstrap_replicates"]),
        seed=int(experiment["seed"]),
    )
    summary = {
        "status": "complete",
        "experiment": experiment["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "inference_router": config["protocol"]["inference_router"],
        "aggregate_results": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", rows)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
