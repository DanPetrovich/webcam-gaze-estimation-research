#!/usr/bin/env python3
"""Evaluate residual-consistency abstention for calibration attempts."""

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
import run_task1_robust_offset as robust  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--robust-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("Invalid quantile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def residual_dispersion_deg(
    selected: list[baseline.Observation], pitch: float, yaw: float
) -> float:
    deviations = [
        math.degrees(
            math.hypot(item.pitch_residual - pitch, item.yaw_residual - yaw)
        )
        for item in selected
    ]
    return statistics.median(deviations)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation inputs must have equal non-trivial length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return correlation(average_ranks(left), average_ranks(right))


def load_random_draws(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["selector"] == "random"]
    return [
        {
            **row,
            "budget": int(row["budget"]),
            "draw": int(row["draw"]),
            "delta_deg": float(row["delta_deg"]),
            "calibrated_mean_deg": float(row["calibrated_mean_deg"]),
            "gain_deg": float(row["gain_deg"]),
        }
        for row in rows
    ]


def attach_reliability(
    rows: list[dict[str, Any]],
    splits: dict[str, baseline.SubjectSplit],
    *,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> None:
    for row in rows:
        split = splits[str(row["subject"])]
        selected = robust.random_sample(
            split, seed, int(row["budget"]), int(row["draw"])
        )
        pitch, yaw, _ = robust.robust_offsets(
            selected,
            delta=math.radians(float(row["delta_deg"])),
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        row["residual_dispersion_deg"] = residual_dispersion_deg(
            selected, pitch, yaw
        )


def evaluate(
    rows: list[dict[str, Any]],
    *,
    budgets: list[int],
    target_coverage: float,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted({str(row["subject"]) for row in rows})
    subject_rows = []
    for subject in subjects:
        for budget in budgets:
            training_scores = [
                float(row["residual_dispersion_deg"])
                for row in rows
                if row["subject"] != subject and row["budget"] == budget
            ]
            threshold = quantile(training_scores, target_coverage)
            matching = [
                row
                for row in rows
                if row["subject"] == subject and row["budget"] == budget
            ]
            accepted = [
                row
                for row in matching
                if float(row["residual_dispersion_deg"]) <= threshold
            ]
            if not accepted:
                raise ValueError("Reliability gate rejected every draw")
            all_errors = [float(row["calibrated_mean_deg"]) for row in matching]
            accepted_errors = [
                float(row["calibrated_mean_deg"]) for row in accepted
            ]
            subject_rows.append(
                {
                    "subject": subject,
                    "budget": budget,
                    "threshold_deg": threshold,
                    "coverage": len(accepted) / len(matching),
                    "all_mean_error_deg": statistics.fmean(all_errors),
                    "accepted_mean_error_deg": statistics.fmean(accepted_errors),
                    "accepted_advantage_deg": statistics.fmean(all_errors)
                    - statistics.fmean(accepted_errors),
                    "all_harmed_draw_fraction": statistics.fmean(
                        float(row["gain_deg"]) < 0.0 for row in matching
                    ),
                    "accepted_harmed_draw_fraction": statistics.fmean(
                        float(row["gain_deg"]) < 0.0 for row in accepted
                    ),
                    "spearman_dispersion_vs_error": spearman(
                        [float(row["residual_dispersion_deg"]) for row in matching],
                        all_errors,
                    ),
                }
            )

    aggregates = []
    for budget in budgets:
        matching = [row for row in subject_rows if row["budget"] == budget]
        advantages = [float(row["accepted_advantage_deg"]) for row in matching]
        low, high = baseline.bootstrap_mean_ci(
            advantages,
            replicates=replicates,
            seed=baseline.stable_seed(seed, "reliability", budget),
        )
        aggregates.append(
            {
                "budget": budget,
                "subjects": len(matching),
                "mean_coverage": statistics.fmean(
                    float(row["coverage"]) for row in matching
                ),
                "mean_all_error_deg": statistics.fmean(
                    float(row["all_mean_error_deg"]) for row in matching
                ),
                "mean_accepted_error_deg": statistics.fmean(
                    float(row["accepted_mean_error_deg"]) for row in matching
                ),
                "mean_accepted_advantage_deg": statistics.fmean(advantages),
                "advantage_ci95_low_deg": low,
                "advantage_ci95_high_deg": high,
                "mean_all_harmed_draw_fraction": statistics.fmean(
                    float(row["all_harmed_draw_fraction"]) for row in matching
                ),
                "mean_accepted_harmed_draw_fraction": statistics.fmean(
                    float(row["accepted_harmed_draw_fraction"]) for row in matching
                ),
                "mean_spearman_dispersion_vs_error": statistics.fmean(
                    float(row["spearman_dispersion_vs_error"]) for row in matching
                ),
            }
        )
    return subject_rows, aggregates


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    budgets = [int(value) for value in config["budgets"]]
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
    rows = [
        row
        for row in load_random_draws(args.robust_results / "per_draw_metrics.csv")
        if row["budget"] in budgets
    ]
    attach_reliability(
        rows,
        splits,
        seed=int(config["seed"]),
        maximum_iterations=int(config["maximum_iterations"]),
        tolerance=float(config["tolerance"]),
    )
    subject_rows, aggregates = evaluate(
        rows,
        budgets=budgets,
        target_coverage=float(config["target_coverage"]),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    expected = int(config["expected_subjects"]) * int(config["expected_draws"]) * len(
        budgets
    )
    if len(rows) != expected:
        raise ValueError("Unexpected random draw count")
    summary = {
        "status": "complete",
        "experiment": config["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "target_coverage": config["target_coverage"],
        "aggregate_results": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "per_draw_reliability.csv", rows)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", subject_rows)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
