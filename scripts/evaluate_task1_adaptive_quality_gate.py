#!/usr/bin/env python3
"""Evaluate a leave-one-subject-out quality-gate policy."""

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
import evaluate_task1_quality_selection as quality_evaluation  # noqa: E402
import run_task1_mean_offset as baseline  # noqa: E402

GATE_FRACTIONS = {
    "quality_then_target_diversity": 0.50,
    "quality75_then_target_diversity": 0.75,
    "quality90_then_target_diversity": 0.90,
    "quality95_then_target_diversity": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-results", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["budget"] = int(row["budget"])
        for field in (
            "uncalibrated_mean_deg",
            "calibrated_mean_deg",
            "calibrated_median_deg",
            "calibrated_p95_deg",
            "gain_deg",
        ):
            row[field] = float(row[field])
    return rows


def choose_gate(
    rows: list[dict[str, Any]],
    *,
    held_subject: str,
    budget: int,
    calibrator: str,
) -> str:
    """Choose the lowest-error gate using only non-held subjects."""
    candidates = []
    for selector, fraction in GATE_FRACTIONS.items():
        errors = [
            float(row["calibrated_mean_deg"])
            for row in rows
            if row["subject"] != held_subject
            and row["selector"] == selector
            and row["budget"] == budget
            and row["calibrator"] == calibrator
        ]
        if not errors:
            raise ValueError(f"Missing training rows for {selector}")
        candidates.append((statistics.fmean(errors), -fraction, selector))
    return min(candidates)[2]


def evaluate_policy(
    rows: list[dict[str, Any]],
    reference: dict[tuple[str, int, str], float],
    *,
    budgets: list[int],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted({str(row["subject"]) for row in rows})
    selected_rows = []
    for subject in subjects:
        for budget in budgets:
            for calibrator in ("mean_offset", "map_offset"):
                selector = choose_gate(
                    rows,
                    held_subject=subject,
                    budget=budget,
                    calibrator=calibrator,
                )
                matches = [
                    row
                    for row in rows
                    if row["subject"] == subject
                    and row["selector"] == selector
                    and row["budget"] == budget
                    and row["calibrator"] == calibrator
                ]
                if len(matches) != 1:
                    raise ValueError("Held-subject gate row is missing or duplicated")
                source = matches[0]
                selected_rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "calibrator": calibrator,
                        "selected_selector": selector,
                        "selected_gate_fraction": GATE_FRACTIONS[selector],
                        "uncalibrated_mean_deg": source["uncalibrated_mean_deg"],
                        "calibrated_mean_deg": source["calibrated_mean_deg"],
                        "calibrated_median_deg": source["calibrated_median_deg"],
                        "calibrated_p95_deg": source["calibrated_p95_deg"],
                        "gain_deg": source["gain_deg"],
                    }
                )

    aggregates = []
    comparisons = []
    for budget in budgets:
        for calibrator in ("mean_offset", "map_offset"):
            matching = [
                row
                for row in selected_rows
                if row["budget"] == budget and row["calibrator"] == calibrator
            ]
            errors = [float(row["calibrated_mean_deg"]) for row in matching]
            gains = [float(row["gain_deg"]) for row in matching]
            gain_low, gain_high = baseline.bootstrap_mean_ci(
                gains,
                replicates=replicates,
                seed=baseline.stable_seed(seed, budget, calibrator, "adaptive_gain"),
            )
            aggregates.append(
                {
                    "budget": budget,
                    "calibrator": calibrator,
                    "subjects": len(matching),
                    "subject_mean_calibrated_deg": statistics.fmean(errors),
                    "subject_std_calibrated_deg": statistics.stdev(errors),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "gain_ci95_low_deg": gain_low,
                    "gain_ci95_high_deg": gain_high,
                    "harmed_subject_fraction": statistics.fmean(
                        value < 0 for value in gains
                    ),
                    "mean_selected_gate_fraction": statistics.fmean(
                        float(row["selected_gate_fraction"]) for row in matching
                    ),
                }
            )
            differences = [
                reference[(str(row["subject"]), budget, calibrator)]
                - float(row["calibrated_mean_deg"])
                for row in matching
            ]
            low, high = baseline.bootstrap_mean_ci(
                differences,
                replicates=replicates,
                seed=baseline.stable_seed(seed, budget, calibrator, "adaptive_random"),
            )
            comparisons.append(
                {
                    "comparison": "adaptive_gate_advantage_over_random",
                    "budget": budget,
                    "calibrator": calibrator,
                    "mean_difference_deg": statistics.fmean(differences),
                    "difference_ci95_low_deg": low,
                    "difference_ci95_high_deg": high,
                }
            )
    return selected_rows, aggregates, comparisons


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    budgets = [int(value) for value in config["experiment"]["budgets"]]
    rows = load_rows(args.quality_results / "per_subject_metrics.csv")
    reference = quality_evaluation.load_random_reference(
        args.mean_offset_results / "per_subject_metrics.csv"
    )
    selected, aggregates, comparisons = evaluate_policy(
        rows,
        reference,
        budgets=budgets,
        replicates=int(config["experiment"]["bootstrap_replicates"]),
        seed=int(config["experiment"]["seed"]),
    )
    summary = {
        "status": "complete",
        "experiment": "task1_adaptive_quality_gate_loso",
        "subjects": sorted({str(row["subject"]) for row in selected}),
        "budgets": budgets,
        "candidate_gate_fractions": sorted(GATE_FRACTIONS.values()),
        "best_aggregate_result": min(
            aggregates,
            key=lambda row: float(row["subject_mean_calibrated_deg"]),
        ),
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
