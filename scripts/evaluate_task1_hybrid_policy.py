#!/usr/bin/env python3
"""Evaluate the prespecified simple Task 1 deployment policy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task1_mean_offset as baseline  # noqa: E402

ONE_SHOT_SELECTOR = "soft_quality25_target_diversity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-quality-results", required=True, type=Path)
    parser.add_argument("--robust-results", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def assemble_policy(
    soft_rows: list[dict[str, str]],
    robust_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    *,
    budgets: list[int],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    soft = {
        (row["subject"], int(row["budget"])): row
        for row in soft_rows
        if row["selector"] == ONE_SHOT_SELECTOR and row["calibrator"] == "map_offset"
    }
    robust = {
        (row["subject"], int(row["budget"])): row
        for row in robust_rows
        if row["selector"] == "random"
    }
    reference = {
        (row["subject"], int(row["budget"]), row["calibrator"]): row
        for row in reference_rows
        if row["selector"] == "random"
    }
    subjects = sorted({subject for subject, _ in soft})
    if not subjects:
        raise ValueError("Soft one-shot results are missing")

    selected = []
    for subject in subjects:
        for budget in budgets:
            if budget == 1:
                source = soft[(subject, budget)]
                method = "soft_quality25_map_offset"
                reference_calibrator = "map_offset"
                gain = float(source["gain_deg"])
            else:
                source = robust[(subject, budget)]
                method = "random_huber_offset"
                reference_calibrator = "mean_offset"
                gain = float(source["mean_gain_deg"])
            reference_error = float(
                reference[(subject, budget, reference_calibrator)][
                    "calibrated_mean_deg"
                ]
            )
            error = float(source["calibrated_mean_deg"])
            selected.append(
                {
                    "subject": subject,
                    "budget": budget,
                    "method": method,
                    "reference_calibrator": reference_calibrator,
                    "calibrated_mean_deg": error,
                    "gain_deg": gain,
                    "advantage_over_reference_deg": reference_error - error,
                }
            )

    aggregates = []
    comparisons = []
    for budget in budgets:
        matching = [row for row in selected if row["budget"] == budget]
        errors = [float(row["calibrated_mean_deg"]) for row in matching]
        gains = [float(row["gain_deg"]) for row in matching]
        differences = [
            float(row["advantage_over_reference_deg"]) for row in matching
        ]
        gain_low, gain_high = baseline.bootstrap_mean_ci(
            gains,
            replicates=replicates,
            seed=baseline.stable_seed(seed, "hybrid", budget, "gain"),
        )
        difference_low, difference_high = baseline.bootstrap_mean_ci(
            differences,
            replicates=replicates,
            seed=baseline.stable_seed(seed, "hybrid", budget, "reference"),
        )
        aggregates.append(
            {
                "budget": budget,
                "method": matching[0]["method"],
                "subjects": len(matching),
                "subject_mean_calibrated_deg": statistics.fmean(errors),
                "subject_std_calibrated_deg": (
                    statistics.stdev(errors) if len(errors) > 1 else 0.0
                ),
                "subject_mean_gain_deg": statistics.fmean(gains),
                "gain_ci95_low_deg": gain_low,
                "gain_ci95_high_deg": gain_high,
                "harmed_subject_fraction": statistics.fmean(
                    value < 0 for value in gains
                ),
            }
        )
        comparisons.append(
            {
                "comparison": "hybrid_policy_advantage_over_random_reference",
                "budget": budget,
                "reference_calibrator": matching[0]["reference_calibrator"],
                "mean_difference_deg": statistics.fmean(differences),
                "difference_ci95_low_deg": difference_low,
                "difference_ci95_high_deg": difference_high,
            }
        )
    return selected, aggregates, comparisons


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    budgets = [int(value) for value in config["budgets"]]
    selected, aggregates, comparisons = assemble_policy(
        read_rows(args.soft_quality_results / "per_subject_metrics.csv"),
        read_rows(args.robust_results / "per_subject_metrics.csv"),
        read_rows(args.mean_offset_results / "per_subject_metrics.csv"),
        budgets=budgets,
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    summary = {
        "status": "complete",
        "experiment": "task1_hybrid_simple_policy",
        "subjects": sorted({str(row["subject"]) for row in selected}),
        "budgets": budgets,
        "policy": {
            "k1": "soft_quality25_target_diversity_plus_map_offset",
            "k_ge_3": "random_plus_loso_tuned_huber_offset",
        },
        "aggregate_results": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", selected)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    baseline.write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
