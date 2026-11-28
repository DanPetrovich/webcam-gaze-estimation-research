#!/usr/bin/env python3
"""Compare deterministic target-diversity selection with Task 1 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task1_mean_offset as baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def target_distance_deg(
    left: baseline.Observation,
    right: baseline.Observation,
) -> float:
    return baseline.angular_error_deg(
        left.target_pitch,
        left.target_yaw,
        right.target_pitch,
        right.target_yaw,
    )


def target_diversity_sample(
    candidates: tuple[baseline.Observation, ...],
    budget: int,
) -> list[baseline.Observation]:
    """Select a deterministic center followed by farthest-first target points."""
    if budget <= 0 or budget > len(candidates):
        raise ValueError("Budget must fit the non-empty candidate pool")
    mean_pitch = statistics.fmean(item.target_pitch for item in candidates)
    mean_yaw = statistics.fmean(item.target_yaw for item in candidates)
    center_vector = baseline.pitchyaw_to_vector(mean_pitch, mean_yaw)

    def distance_to_center(item: baseline.Observation) -> float:
        item_vector = baseline.pitchyaw_to_vector(item.target_pitch, item.target_yaw)
        cosine = sum(
            left * right for left, right in zip(item_vector, center_vector, strict=True)
        )
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    first = min(
        candidates, key=lambda item: (distance_to_center(item), item.min_sample_index)
    )
    selected = [first]
    selected_ids = {first.group_id}
    while len(selected) < budget:
        next_item = max(
            (item for item in candidates if item.group_id not in selected_ids),
            key=lambda item: (
                min(target_distance_deg(item, chosen) for chosen in selected),
                -item.min_sample_index,
            ),
        )
        selected.append(next_item)
        selected_ids.add(next_item.group_id)
    return selected


def coverage_metrics(
    candidates: tuple[baseline.Observation, ...],
    selected: list[baseline.Observation],
) -> tuple[float, float, float]:
    nearest = [
        min(target_distance_deg(candidate, chosen) for chosen in selected)
        for candidate in candidates
    ]
    pairwise = [
        target_distance_deg(left, right)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ]
    return (
        max(nearest),
        statistics.fmean(nearest),
        min(pairwise) if pairwise else 0.0,
    )


def load_baseline_subjects(
    path: Path,
) -> dict[tuple[str, str, int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["subject"],
                row["selector"],
                int(row["budget"]),
                row["calibrator"],
            ): row
            for row in csv.DictReader(handle)
        }


def aggregate(
    subject_rows: list[dict[str, Any]],
    *,
    baseline_subjects: dict[tuple[str, str, int, str], dict[str, str]],
    budgets: list[int],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for calibrator in ("mean_offset", "map_offset"):
            matching = [
                row
                for row in subject_rows
                if row["budget"] == budget and row["calibrator"] == calibrator
            ]
            errors = [float(row["calibrated_mean_deg"]) for row in matching]
            gains = [float(row["gain_deg"]) for row in matching]
            gain_low, gain_high = baseline.bootstrap_mean_ci(
                gains,
                replicates=bootstrap_replicates,
                seed=baseline.stable_seed(
                    seed, budget, calibrator, "target_diversity_gain"
                ),
            )
            aggregate_rows.append(
                {
                    "selector": "target_diversity",
                    "budget": budget,
                    "calibrator": calibrator,
                    "subjects": len(matching),
                    "subject_mean_calibrated_deg": statistics.fmean(errors),
                    "subject_std_calibrated_deg": statistics.stdev(errors),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "gain_ci95_low_deg": gain_low,
                    "gain_ci95_high_deg": gain_high,
                    "harmed_subject_fraction": statistics.fmean(
                        gain < 0 for gain in gains
                    ),
                    "mean_covering_radius_deg": statistics.fmean(
                        float(row["target_covering_radius_deg"]) for row in matching
                    ),
                }
            )
            for reference_selector in ("random", "fixed_uniform"):
                differences = []
                for row in matching:
                    key = (
                        row["subject"],
                        reference_selector,
                        budget,
                        calibrator,
                    )
                    reference = baseline_subjects.get(key)
                    if reference is None:
                        raise ValueError(f"Missing baseline subject result: {key}")
                    differences.append(
                        float(reference["calibrated_mean_deg"])
                        - float(row["calibrated_mean_deg"])
                    )
                low, high = baseline.bootstrap_mean_ci(
                    differences,
                    replicates=bootstrap_replicates,
                    seed=baseline.stable_seed(
                        seed,
                        budget,
                        calibrator,
                        reference_selector,
                        "target_diversity_comparison",
                    ),
                )
                comparison_rows.append(
                    {
                        "comparison": (
                            f"target_diversity_advantage_over_{reference_selector}"
                        ),
                        "budget": budget,
                        "calibrator": calibrator,
                        "mean_difference_deg": statistics.fmean(differences),
                        "difference_ci95_low_deg": low,
                        "difference_ci95_high_deg": high,
                    }
                )
    return aggregate_rows, comparison_rows


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    budgets = [int(value) for value in experiment["budgets"]]
    seed = int(experiment["seed"])
    bootstrap_replicates = int(experiment["bootstrap_replicates"])

    observations = baseline.load_unique_observations(args.predictions)
    if sorted(observations) != baseline.EXPECTED_SUBJECTS:
        raise ValueError("Predictions do not contain the expected 15 subjects")
    splits = {
        subject: baseline.split_subject(
            subject,
            values,
            candidate_fraction=float(experiment["candidate_fraction"]),
            guard_fraction=float(experiment["guard_fraction"]),
        )
        for subject, values in observations.items()
    }
    baseline_subjects = load_baseline_subjects(
        args.mean_offset_results / "per_subject_metrics.csv"
    )

    subject_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for subject, split in sorted(splits.items()):
        prior = baseline.estimate_map_prior(splits, subject)
        uncalibrated = statistics.fmean(
            item.uncalibrated_error_deg for item in split.evaluation
        )
        for budget in budgets:
            selected = target_diversity_sample(split.candidate, budget)
            covering_radius, mean_nearest, min_pairwise = coverage_metrics(
                split.candidate, selected
            )
            for rank, item in enumerate(selected):
                selection_rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "rank": rank,
                        "group_id": item.group_id,
                        "min_sample_index": item.min_sample_index,
                        "target_pitch_rad": item.target_pitch,
                        "target_yaw_rad": item.target_yaw,
                    }
                )
            for calibrator in ("mean_offset", "map_offset"):
                pitch_offset, yaw_offset, _, _ = baseline.calibration_offsets(
                    selected,
                    calibrator=calibrator,
                    prior=prior,
                )
                mean_error, median_error, p95_error = baseline.summarize_evaluation(
                    split.evaluation,
                    pitch_offset,
                    yaw_offset,
                )
                subject_rows.append(
                    {
                        "subject": subject,
                        "selector": "target_diversity",
                        "budget": budget,
                        "calibrator": calibrator,
                        "uncalibrated_mean_deg": uncalibrated,
                        "calibrated_mean_deg": mean_error,
                        "calibrated_median_deg": median_error,
                        "calibrated_p95_deg": p95_error,
                        "gain_deg": uncalibrated - mean_error,
                        "target_covering_radius_deg": covering_radius,
                        "target_mean_nearest_deg": mean_nearest,
                        "target_min_pairwise_deg": min_pairwise,
                    }
                )

    aggregate_rows, comparison_rows = aggregate(
        subject_rows,
        baseline_subjects=baseline_subjects,
        budgets=budgets,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary = {
        "status": "complete",
        "experiment": experiment["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "selector": "target_diversity_k_center",
        "best_aggregate_result": min(
            aggregate_rows,
            key=lambda row: float(row["subject_mean_calibrated_deg"]),
        ),
        "input_predictions": str(args.predictions),
        "mean_offset_results": str(args.mean_offset_results),
        "config": str(args.config),
        "limitations": [
            "Known target directions are available in guided calibration but not "
            "for passive unlabeled frame selection.",
            "Target diversity does not measure image validity, head pose, or "
            "feature-space redundancy.",
            "Temporal order is approximated by minimum saved sample index.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "selected_samples.csv", selection_rows)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", subject_rows)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    baseline.write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
