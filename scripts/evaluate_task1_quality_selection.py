#!/usr/bin/env python3
"""Evaluate server-selected calibration groups on frozen Task 0 predictions."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_random_reference(path: Path) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["subject"], int(row["budget"]), row["calibrator"]): float(
            row["calibrated_mean_deg"]
        )
        for row in rows
        if row["selector"] == "random"
    }


def aggregate(
    rows: list[dict[str, Any]],
    reference: dict[tuple[str, int, str], float],
    *,
    selectors: list[str],
    budgets: list[int],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates = []
    comparisons = []
    for selector in selectors:
        for budget in budgets:
            for calibrator in ("mean_offset", "map_offset"):
                matching = [
                    row
                    for row in rows
                    if row["selector"] == selector
                    and row["budget"] == budget
                    and row["calibrator"] == calibrator
                ]
                errors = [float(row["calibrated_mean_deg"]) for row in matching]
                gains = [float(row["gain_deg"]) for row in matching]
                gain_low, gain_high = baseline.bootstrap_mean_ci(
                    gains,
                    replicates=replicates,
                    seed=baseline.stable_seed(seed, selector, budget, calibrator, "gain"),
                )
                aggregates.append(
                    {
                        "selector": selector,
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
                    }
                )
                differences = [
                    reference[(row["subject"], budget, calibrator)]
                    - float(row["calibrated_mean_deg"])
                    for row in matching
                ]
                low, high = baseline.bootstrap_mean_ci(
                    differences,
                    replicates=replicates,
                    seed=baseline.stable_seed(
                        seed, selector, budget, calibrator, "vs_random"
                    ),
                )
                comparisons.append(
                    {
                        "comparison": f"{selector}_advantage_over_random",
                        "budget": budget,
                        "calibrator": calibrator,
                        "mean_difference_deg": statistics.fmean(differences),
                        "difference_ci95_low_deg": low,
                        "difference_ci95_high_deg": high,
                    }
                )
    return aggregates, comparisons


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    budgets = [int(value) for value in config["experiment"]["budgets"]]
    selectors = list(config["selection"]["selectors"])
    if selection.get("status") != "complete" or selection.get("selectors") != selectors:
        raise ValueError("Selection summary does not match the fixed protocol")
    observations = baseline.load_unique_observations(args.predictions)
    splits = {
        subject: baseline.split_subject(
            subject, values, candidate_fraction=0.20, guard_fraction=0.10
        )
        for subject, values in observations.items()
    }
    rows: list[dict[str, Any]] = []
    for subject, split in sorted(splits.items()):
        indexed = {item.group_id: item for item in split.candidate}
        prior = baseline.estimate_map_prior(splits, subject)
        uncalibrated = statistics.fmean(
            item.uncalibrated_error_deg for item in split.evaluation
        )
        for selector in selectors:
            selected = []
            for record in selection["selections"][subject][selector]:
                item = indexed.get(record["group_id"])
                if item is None or item.min_sample_index != record["min_sample_index"]:
                    raise ValueError("Selected group does not match the frozen split")
                selected.append(item)
            if len({item.group_id for item in selected}) != max(budgets):
                raise ValueError("Selected groups are not unique")
            for budget in budgets:
                for calibrator in ("mean_offset", "map_offset"):
                    pitch, yaw, _, _ = baseline.calibration_offsets(
                        selected[:budget], calibrator=calibrator, prior=prior
                    )
                    mean_error, median_error, p95_error = baseline.summarize_evaluation(
                        split.evaluation, pitch, yaw
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "selector": selector,
                            "budget": budget,
                            "calibrator": calibrator,
                            "uncalibrated_mean_deg": uncalibrated,
                            "calibrated_mean_deg": mean_error,
                            "calibrated_median_deg": median_error,
                            "calibrated_p95_deg": p95_error,
                            "gain_deg": uncalibrated - mean_error,
                        }
                    )
    reference = load_random_reference(
        args.mean_offset_results / "per_subject_metrics.csv"
    )
    aggregates, comparisons = aggregate(
        rows,
        reference,
        selectors=selectors,
        budgets=budgets,
        replicates=int(config["experiment"]["bootstrap_replicates"]),
        seed=int(config["experiment"]["seed"]),
    )
    summary = {
        "status": "complete",
        "experiment": config["experiment"]["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "selectors": selectors,
        "quality_gate_fraction_mean": selection["quality_gate_fraction_mean"],
        "best_aggregate_result": min(
            aggregates,
            key=lambda row: float(row["subject_mean_calibrated_deg"]),
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", rows)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    baseline.write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
