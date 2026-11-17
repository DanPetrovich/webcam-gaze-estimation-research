#!/usr/bin/env python3
"""Evaluate leakage-safe Huber-weighted residual offset calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task1_mean_offset as baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--quality-selection-results", required=True, type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def huber_location(
    values: list[float],
    *,
    delta: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[float, list[float]]:
    if not values or delta <= 0:
        raise ValueError("Huber location requires values and a positive delta")
    location = statistics.median(values)
    weights = [1.0] * len(values)
    for _ in range(maximum_iterations):
        weights = [
            min(1.0, delta / max(abs(value - location), 1e-15))
            for value in values
        ]
        numerator = sum(
            weight * value
            for weight, value in zip(weights, values, strict=True)
        )
        updated = numerator / sum(weights)
        if abs(updated - location) <= tolerance:
            location = updated
            break
        location = updated
    return location, weights


def robust_offsets(
    selected: list[baseline.Observation],
    *,
    delta: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[float, float, float]:
    pitch, pitch_weights = huber_location(
        [item.pitch_residual for item in selected],
        delta=delta,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    yaw, yaw_weights = huber_location(
        [item.yaw_residual for item in selected],
        delta=delta,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    effective_fraction = statistics.fmean(
        (pitch_weight + yaw_weight) / 2
        for pitch_weight, yaw_weight in zip(
            pitch_weights, yaw_weights, strict=True
        )
    )
    return pitch, yaw, effective_fraction


def random_sample(
    split: baseline.SubjectSplit, seed: int, budget: int, draw: int
) -> list[baseline.Observation]:
    generator = random.Random(
        baseline.stable_seed(seed, split.subject, budget, "random", draw)
    )
    return generator.sample(list(split.candidate), budget)


def select_delta(
    splits: dict[str, baseline.SubjectSplit],
    held_out: str,
    budget: int,
    deltas: list[float],
    *,
    tuning_draws: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[float, dict[float, float]]:
    totals = {delta: 0.0 for delta in deltas}
    cases = 0
    for subject, split in sorted(splits.items()):
        if subject == held_out:
            continue
        for draw in range(tuning_draws):
            selected = random_sample(split, seed, budget, draw)
            for delta in deltas:
                pitch, yaw, _ = robust_offsets(
                    selected,
                    delta=delta,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                totals[delta] += baseline.summarize_evaluation(
                    split.guard, pitch, yaw
                )[0]
            cases += 1
    means = {delta: total / cases for delta, total in totals.items()}
    return min(deltas, key=lambda delta: (means[delta], -delta)), means


def load_quality_selections(
    path: Path,
    splits: dict[str, baseline.SubjectSplit],
) -> dict[str, list[baseline.Observation]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for subject, split in splits.items():
        indexed = {item.group_id: item for item in split.candidate}
        selected = []
        for record in value["selections"][subject]["quality_then_target_diversity"]:
            item = indexed.get(record["group_id"])
            if item is None or item.min_sample_index != record["min_sample_index"]:
                raise ValueError("Quality selection does not match frozen split")
            selected.append(item)
        result[subject] = selected
    return result


def load_reference(
    path: Path, selector: str
) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["subject"], int(row["budget"]), row["calibrator"]): float(
                row["calibrated_mean_deg"]
            )
            for row in csv.DictReader(stream)
            if row["selector"] == selector
        }


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    budgets = [int(value) for value in config["budgets"]]
    deltas = [math.radians(float(value)) for value in config["huber_delta_degrees"]]
    seed = int(config["seed"])
    iterations = int(config["maximum_iterations"])
    tolerance = float(config["tolerance"])
    observations = baseline.load_unique_observations(args.predictions)
    splits = {
        subject: baseline.split_subject(
            subject,
            values,
            candidate_fraction=float(config["candidate_fraction"]),
            guard_fraction=float(config["guard_fraction"]),
        )
        for subject, values in observations.items()
    }
    quality = load_quality_selections(args.selection_summary, splits)
    selected_deltas = {}
    tuning_rows = []
    for subject in sorted(splits):
        for budget in budgets:
            selected, means = select_delta(
                splits,
                subject,
                budget,
                deltas,
                tuning_draws=int(config["tuning_random_draws"]),
                seed=seed,
                maximum_iterations=iterations,
                tolerance=tolerance,
            )
            selected_deltas[(subject, budget)] = selected
            for delta, error in means.items():
                tuning_rows.append(
                    {
                        "held_out_subject": subject,
                        "budget": budget,
                        "delta_deg": math.degrees(delta),
                        "guard_mean_deg": error,
                        "selected": delta == selected,
                    }
                )
    draw_rows = []
    for subject, split in sorted(splits.items()):
        uncalibrated = statistics.fmean(
            item.uncalibrated_error_deg for item in split.evaluation
        )
        for budget in budgets:
            delta = selected_deltas[(subject, budget)]
            for selector in ("random", "quality_then_target_diversity"):
                draws = int(config["random_draws"]) if selector == "random" else 1
                for draw in range(draws):
                    selected = (
                        random_sample(split, seed, budget, draw)
                        if selector == "random"
                        else quality[subject][:budget]
                    )
                    pitch, yaw, effective = robust_offsets(
                        selected,
                        delta=delta,
                        maximum_iterations=iterations,
                        tolerance=tolerance,
                    )
                    mean_error, median_error, p95_error = baseline.summarize_evaluation(
                        split.evaluation, pitch, yaw
                    )
                    draw_rows.append(
                        {
                            "subject": subject,
                            "selector": selector,
                            "budget": budget,
                            "draw": draw,
                            "delta_deg": math.degrees(delta),
                            "effective_weight_fraction": effective,
                            "uncalibrated_mean_deg": uncalibrated,
                            "calibrated_mean_deg": mean_error,
                            "calibrated_median_deg": median_error,
                            "calibrated_p95_deg": p95_error,
                            "gain_deg": uncalibrated - mean_error,
                        }
                    )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in draw_rows:
        grouped[(row["subject"], row["selector"], row["budget"])].append(row)
    subject_rows = []
    for (subject, selector, budget), rows in sorted(grouped.items()):
        subject_rows.append(
            {
                "subject": subject,
                "selector": selector,
                "budget": budget,
                "calibrated_mean_deg": statistics.fmean(
                    float(row["calibrated_mean_deg"]) for row in rows
                ),
                "mean_gain_deg": statistics.fmean(
                    float(row["gain_deg"]) for row in rows
                ),
                "mean_effective_weight_fraction": statistics.fmean(
                    float(row["effective_weight_fraction"]) for row in rows
                ),
            }
        )
    random_mean = load_reference(
        args.mean_offset_results / "per_subject_metrics.csv", "random"
    )
    quality_mean = load_reference(
        args.quality_selection_results / "per_subject_metrics.csv",
        "quality_then_target_diversity",
    )
    aggregate_rows = []
    comparison_rows = []
    replicates = int(config["bootstrap_replicates"])
    for selector in ("random", "quality_then_target_diversity"):
        reference = random_mean if selector == "random" else quality_mean
        for budget in budgets:
            matching = [
                row
                for row in subject_rows
                if row["selector"] == selector and row["budget"] == budget
            ]
            errors = [float(row["calibrated_mean_deg"]) for row in matching]
            gains = [float(row["mean_gain_deg"]) for row in matching]
            aggregate_rows.append(
                {
                    "selector": selector,
                    "budget": budget,
                    "subject_mean_calibrated_deg": statistics.fmean(errors),
                    "subject_std_calibrated_deg": statistics.stdev(errors),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "harmed_subject_fraction": statistics.fmean(
                        gain < 0 for gain in gains
                    ),
                    "mean_effective_weight_fraction": statistics.fmean(
                        float(row["mean_effective_weight_fraction"]) for row in matching
                    ),
                }
            )
            differences = [
                reference[(row["subject"], budget, "mean_offset")]
                - float(row["calibrated_mean_deg"])
                for row in matching
            ]
            low, high = baseline.bootstrap_mean_ci(
                differences,
                replicates=replicates,
                seed=baseline.stable_seed(seed, selector, budget, "robust_vs_mean"),
            )
            comparison_rows.append(
                {
                    "comparison": "robust_advantage_over_mean_offset",
                    "selector": selector,
                    "budget": budget,
                    "mean_difference_deg": statistics.fmean(differences),
                    "difference_ci95_low_deg": low,
                    "difference_ci95_high_deg": high,
                }
            )
    summary = {
        "status": "complete",
        "experiment": config["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "best_aggregate_result": min(
            aggregate_rows, key=lambda row: float(row["subject_mean_calibrated_deg"])
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "tuning.csv", tuning_rows)
    baseline.write_csv(args.output_dir / "per_draw_metrics.csv", draw_rows)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", subject_rows)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    baseline.write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
