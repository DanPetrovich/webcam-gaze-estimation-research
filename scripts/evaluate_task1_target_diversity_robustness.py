#!/usr/bin/env python3
"""Evaluate target-diversity calibration across eccentricity slices."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_task1_hybrid_robustness as hybrid_slices  # noqa: E402
import run_task1_mean_offset as baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--selected-samples", required=True, type=Path)
    parser.add_argument("--hybrid-slice-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_selections(
    path: Path,
    splits: dict[str, baseline.SubjectSplit],
    budgets: list[int],
) -> dict[tuple[str, int], list[baseline.Observation]]:
    indexed = {
        subject: {item.group_id: item for item in split.candidate}
        for subject, split in splits.items()
    }
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            budget = int(row["budget"])
            if budget in budgets:
                grouped[(row["subject"], budget)].append(row)
    result = {}
    for key, rows in grouped.items():
        subject, budget = key
        ordered = sorted(rows, key=lambda row: int(row["rank"]))
        selected = []
        for row in ordered:
            item = indexed[subject].get(row["group_id"])
            if item is None or item.min_sample_index != int(row["min_sample_index"]):
                raise ValueError("Target-diversity selection does not match split")
            selected.append(item)
        if len(selected) != budget:
            raise ValueError("Target-diversity budget is incomplete")
        result[key] = selected
    return result


def load_hybrid_reference(path: Path) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["subject"], int(row["budget"]), row["slice"]): float(
                row["calibrated_mean_deg"]
            )
            for row in csv.DictReader(stream)
            if int(row["budget"]) > 1
        }


def evaluate(
    splits: dict[str, baseline.SubjectSplit],
    selections: dict[tuple[str, int], list[baseline.Observation]],
    hybrid_reference: dict[tuple[str, int, str], float],
    *,
    budgets: list[int],
    central_max: float,
    mid_max: float,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows = []
    for subject, split in sorted(splits.items()):
        slices: dict[str, list[baseline.Observation]] = defaultdict(list)
        for item in split.evaluation:
            name = hybrid_slices.slice_name(
                hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
            )
            slices[name].append(item)
        prior = baseline.estimate_map_prior(splits, subject)
        for budget in budgets:
            selected = selections[(subject, budget)]
            for calibrator in ("mean_offset", "map_offset"):
                pitch, yaw, _, _ = baseline.calibration_offsets(
                    selected, calibrator=calibrator, prior=prior
                )
                for name, observations in sorted(slices.items()):
                    uncalibrated = statistics.fmean(
                        item.uncalibrated_error_deg for item in observations
                    )
                    calibrated = hybrid_slices.slice_error(observations, pitch, yaw)
                    huber_error = hybrid_reference[(subject, budget, name)]
                    subject_rows.append(
                        {
                            "subject": subject,
                            "budget": budget,
                            "calibrator": calibrator,
                            "slice": name,
                            "uncalibrated_mean_deg": uncalibrated,
                            "calibrated_mean_deg": calibrated,
                            "gain_deg": uncalibrated - calibrated,
                            "huber_mean_deg": huber_error,
                            "advantage_over_huber_deg": huber_error - calibrated,
                        }
                    )

    aggregates = []
    for budget in budgets:
        for calibrator in ("mean_offset", "map_offset"):
            for name in ("central_le_10", "mid_10_20", "peripheral_gt_20"):
                matching = [
                    row
                    for row in subject_rows
                    if row["budget"] == budget
                    and row["calibrator"] == calibrator
                    and row["slice"] == name
                ]
                differences = [
                    float(row["advantage_over_huber_deg"]) for row in matching
                ]
                low, high = baseline.bootstrap_mean_ci(
                    differences,
                    replicates=replicates,
                    seed=baseline.stable_seed(
                        seed, "target_diversity_slice", budget, calibrator, name
                    ),
                )
                aggregates.append(
                    {
                        "budget": budget,
                        "calibrator": calibrator,
                        "slice": name,
                        "subjects": len(matching),
                        "subject_mean_calibrated_deg": statistics.fmean(
                            float(row["calibrated_mean_deg"]) for row in matching
                        ),
                        "subject_mean_huber_deg": statistics.fmean(
                            float(row["huber_mean_deg"]) for row in matching
                        ),
                        "advantage_over_huber_deg": statistics.fmean(differences),
                        "advantage_ci95_low_deg": low,
                        "advantage_ci95_high_deg": high,
                        "harmed_subject_fraction": statistics.fmean(
                            float(row["gain_deg"]) < 0.0 for row in matching
                        ),
                    }
                )
    return subject_rows, aggregates


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
    subject_rows, aggregates = evaluate(
        splits,
        load_selections(args.selected_samples, splits, budgets),
        load_hybrid_reference(
            args.hybrid_slice_results / "per_subject_slice_metrics.csv"
        ),
        budgets=budgets,
        central_max=float(slices_config["central_max_deg"]),
        mid_max=float(slices_config["mid_max_deg"]),
        replicates=int(experiment["bootstrap_replicates"]),
        seed=int(experiment["seed"]),
    )
    summary = {
        "status": "complete",
        "experiment": experiment["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "slices": slices_config["names"],
        "aggregate_results": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "per_subject_slice_metrics.csv", subject_rows)
    baseline.write_csv(args.output_dir / "aggregate_slice_metrics.csv", aggregates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
