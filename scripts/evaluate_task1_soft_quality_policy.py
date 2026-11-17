#!/usr/bin/env python3
"""Evaluate LOSO selection of the soft quality-coverage weight."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_task1_adaptive_quality_gate as adaptive  # noqa: E402
import evaluate_task1_quality_selection as quality_evaluation  # noqa: E402
import run_task1_mean_offset as baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-results", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    budgets = [int(value) for value in config["experiment"]["budgets"]]
    rows = adaptive.load_rows(args.quality_results / "per_subject_metrics.csv")
    reference = quality_evaluation.load_random_reference(
        args.mean_offset_results / "per_subject_metrics.csv"
    )
    selected, aggregates, comparisons = adaptive.evaluate_policy(
        rows,
        reference,
        budgets=budgets,
        replicates=int(config["experiment"]["bootstrap_replicates"]),
        seed=int(config["experiment"]["seed"]),
        selector_values=adaptive.SOFT_QUALITY_WEIGHTS,
        selected_value_field="selected_quality_weight",
        comparison_name="adaptive_soft_quality_advantage_over_random",
    )
    summary = {
        "status": "complete",
        "experiment": "task1_soft_quality_policy_loso",
        "subjects": sorted({str(row["subject"]) for row in selected}),
        "budgets": budgets,
        "candidate_quality_weights": sorted(adaptive.SOFT_QUALITY_WEIGHTS.values()),
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
