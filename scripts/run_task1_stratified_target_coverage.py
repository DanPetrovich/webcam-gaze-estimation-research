#!/usr/bin/env python3
"""Evaluate region-stratified target coverage with robust offset fitting."""

from __future__ import annotations

import argparse
import csv
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
import run_task1_target_diversity as target_diversity  # noqa: E402

SLICE_ORDER = ("central_le_10", "mid_10_20", "peripheral_gt_20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--robust-results", required=True, type=Path)
    parser.add_argument("--hybrid-slice-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def region_quotas(budget: int) -> dict[str, int]:
    if budget < 3:
        raise ValueError("Stratified coverage requires at least three samples")
    base, remainder = divmod(budget, 3)
    quotas = {name: base for name in SLICE_ORDER}
    for name in ("mid_10_20", "peripheral_gt_20")[:remainder]:
        quotas[name] += 1
    return quotas


def stratified_sample(
    candidates: tuple[baseline.Observation, ...],
    budget: int,
    *,
    central_max: float,
    mid_max: float,
) -> list[baseline.Observation]:
    regions: dict[str, list[baseline.Observation]] = defaultdict(list)
    for item in candidates:
        name = hybrid_slices.slice_name(
            hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
        )
        regions[name].append(item)
    selected = []
    for name, quota in region_quotas(budget).items():
        if len(regions[name]) < quota:
            raise ValueError("A candidate region cannot satisfy its quota")
        selected.extend(
            target_diversity.target_diversity_sample(tuple(regions[name]), quota)
        )
    if len({item.group_id for item in selected}) != budget:
        raise ValueError("Stratified selection is incomplete or duplicated")
    return selected


def load_selected_deltas(
    path: Path, budgets: list[int]
) -> dict[tuple[str, int], float]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["selected"].lower() == "true" and int(row["budget"]) in budgets:
                result[(row["held_out_subject"], int(row["budget"]))] = math.radians(
                    float(row["delta_deg"])
                )
    return result


def load_hybrid_references(
    slice_path: Path, subject_path: Path
) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int], float]]:
    with slice_path.open(newline="", encoding="utf-8") as stream:
        slices = {
            (row["subject"], int(row["budget"]), row["slice"]): float(
                row["calibrated_mean_deg"]
            )
            for row in csv.DictReader(stream)
            if int(row["budget"]) > 1
        }
    with subject_path.open(newline="", encoding="utf-8") as stream:
        overall = {
            (row["subject"], int(row["budget"])): float(row["calibrated_mean_deg"])
            for row in csv.DictReader(stream)
            if row["selector"] == "random" and int(row["budget"]) > 1
        }
    return slices, overall


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
        evaluation_slices: dict[str, list[baseline.Observation]] = defaultdict(list)
        for item in split.evaluation:
            name = hybrid_slices.slice_name(
                hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
            )
            evaluation_slices[name].append(item)
        for budget in budgets:
            selected = stratified_sample(
                split.candidate,
                budget,
                central_max=central_max,
                mid_max=mid_max,
            )
            pitch, yaw, effective = robust.robust_offsets(
                selected,
                delta=deltas[(subject, budget)],
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            overall_error = baseline.summarize_evaluation(
                split.evaluation, pitch, yaw
            )[0]
            rows.append(
                {
                    "subject": subject,
                    "budget": budget,
                    "slice": "overall",
                    "quota": budget,
                    "calibrated_mean_deg": overall_error,
                    "huber_reference_deg": hybrid_overall[(subject, budget)],
                    "advantage_over_huber_deg": hybrid_overall[(subject, budget)]
                    - overall_error,
                    "effective_weight_fraction": effective,
                }
            )
            quotas = region_quotas(budget)
            for name, observations in sorted(evaluation_slices.items()):
                error = hybrid_slices.slice_error(observations, pitch, yaw)
                reference = hybrid_slice[(subject, budget, name)]
                rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "slice": name,
                        "quota": quotas[name],
                        "calibrated_mean_deg": error,
                        "huber_reference_deg": reference,
                        "advantage_over_huber_deg": reference - error,
                        "effective_weight_fraction": effective,
                    }
                )

    aggregates = []
    for budget in budgets:
        for name in ("overall", *SLICE_ORDER):
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
                seed=baseline.stable_seed(seed, "stratified", budget, name),
            )
            aggregates.append(
                {
                    "budget": budget,
                    "slice": name,
                    "quota": matching[0]["quota"],
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
                    "mean_effective_weight_fraction": statistics.fmean(
                        float(row["effective_weight_fraction"]) for row in matching
                    ),
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
    hybrid_slice, hybrid_overall = load_hybrid_references(
        args.hybrid_slice_results / "per_subject_slice_metrics.csv",
        args.robust_results / "per_subject_metrics.csv",
    )
    rows, aggregates = evaluate(
        splits,
        load_selected_deltas(args.robust_results / "tuning.csv", budgets),
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
        "quota_policy": slices_config["quota_policy"],
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
