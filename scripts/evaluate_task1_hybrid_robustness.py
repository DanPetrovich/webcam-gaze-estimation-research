#!/usr/bin/env python3
"""Evaluate the hybrid policy across target-eccentricity slices."""

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
import run_task1_mean_offset as baseline  # noqa: E402
import run_task1_robust_offset as robust  # noqa: E402

ONE_SHOT_SELECTOR = "soft_quality25_target_diversity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--robust-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def target_eccentricity_deg(item: baseline.Observation) -> float:
    vector = baseline.pitchyaw_to_vector(item.target_pitch, item.target_yaw)
    return math.degrees(math.acos(max(-1.0, min(1.0, vector[2]))))


def slice_name(eccentricity: float, central_max: float, mid_max: float) -> str:
    if eccentricity <= central_max:
        return "central_le_10"
    if eccentricity <= mid_max:
        return "mid_10_20"
    return "peripheral_gt_20"


def load_one_shot(
    path: Path, splits: dict[str, baseline.SubjectSplit]
) -> dict[str, baseline.Observation]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for subject, split in splits.items():
        indexed = {item.group_id: item for item in split.candidate}
        record = value["selections"][subject][ONE_SHOT_SELECTOR][0]
        selected = indexed.get(record["group_id"])
        if selected is None or selected.min_sample_index != record["min_sample_index"]:
            raise ValueError("One-shot selection does not match frozen split")
        result[subject] = selected
    return result


def load_robust_draws(path: Path, budgets: list[int]) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["selector"] == "random" and int(row["budget"]) in budgets
        ]
    return [
        {
            "subject": row["subject"],
            "budget": int(row["budget"]),
            "draw": int(row["draw"]),
            "delta_deg": float(row["delta_deg"]),
        }
        for row in rows
    ]


def slice_error(
    observations: list[baseline.Observation], pitch: float, yaw: float
) -> float:
    return statistics.fmean(
        baseline.angular_error_deg(
            item.pred_pitch + pitch,
            item.pred_yaw + yaw,
            item.target_pitch,
            item.target_yaw,
        )
        for item in observations
    )


def evaluate(
    splits: dict[str, baseline.SubjectSplit],
    one_shot: dict[str, baseline.Observation],
    robust_draws: list[dict[str, Any]],
    *,
    budgets: list[int],
    central_max: float,
    mid_max: float,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
    replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    draws_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in robust_draws:
        draws_by_key[(str(row["subject"]), int(row["budget"]))].append(row)

    subject_rows = []
    for subject, split in sorted(splits.items()):
        slices: dict[str, list[baseline.Observation]] = defaultdict(list)
        for item in split.evaluation:
            name = slice_name(target_eccentricity_deg(item), central_max, mid_max)
            slices[name].append(item)
        expected_slices = ("central_le_10", "mid_10_20", "peripheral_gt_20")
        if any(not slices[name] for name in expected_slices):
            raise ValueError("A target-eccentricity slice is empty")
        prior = baseline.estimate_map_prior(splits, subject)
        for budget in budgets:
            if budget == 1:
                pitch, yaw, _, _ = baseline.calibration_offsets(
                    [one_shot[subject]], calibrator="map_offset", prior=prior
                )
                offsets = [(pitch, yaw)]
                method = "soft_quality25_map_offset"
            else:
                offsets = []
                for row in draws_by_key[(subject, budget)]:
                    selected = robust.random_sample(split, seed, budget, row["draw"])
                    pitch, yaw, _ = robust.robust_offsets(
                        selected,
                        delta=math.radians(float(row["delta_deg"])),
                        maximum_iterations=maximum_iterations,
                        tolerance=tolerance,
                    )
                    offsets.append((pitch, yaw))
                method = "random_huber_offset"
            if not offsets:
                raise ValueError("Policy offsets are missing")
            for name, observations in sorted(slices.items()):
                uncalibrated = statistics.fmean(
                    item.uncalibrated_error_deg for item in observations
                )
                calibrated = statistics.fmean(
                    slice_error(observations, pitch, yaw) for pitch, yaw in offsets
                )
                subject_rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "method": method,
                        "slice": name,
                        "unique_groups": len(observations),
                        "uncalibrated_mean_deg": uncalibrated,
                        "calibrated_mean_deg": calibrated,
                        "gain_deg": uncalibrated - calibrated,
                    }
                )

    aggregates = []
    for budget in budgets:
        for name in ("central_le_10", "mid_10_20", "peripheral_gt_20"):
            matching = [
                row
                for row in subject_rows
                if row["budget"] == budget and row["slice"] == name
            ]
            gains = [float(row["gain_deg"]) for row in matching]
            low, high = baseline.bootstrap_mean_ci(
                gains,
                replicates=replicates,
                seed=baseline.stable_seed(seed, "hybrid_slice", budget, name),
            )
            aggregates.append(
                {
                    "budget": budget,
                    "method": matching[0]["method"],
                    "slice": name,
                    "subjects": len(matching),
                    "mean_unique_groups": statistics.fmean(
                        int(row["unique_groups"]) for row in matching
                    ),
                    "subject_mean_uncalibrated_deg": statistics.fmean(
                        float(row["uncalibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_calibrated_deg": statistics.fmean(
                        float(row["calibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "gain_ci95_low_deg": low,
                    "gain_ci95_high_deg": high,
                    "harmed_subject_fraction": statistics.fmean(
                        value < 0.0 for value in gains
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
        load_one_shot(args.selection_summary, splits),
        load_robust_draws(
            args.robust_results / "per_draw_metrics.csv",
            [budget for budget in budgets if budget > 1],
        ),
        budgets=budgets,
        central_max=float(slices_config["central_max_deg"]),
        mid_max=float(slices_config["mid_max_deg"]),
        seed=int(experiment["seed"]),
        maximum_iterations=int(experiment["maximum_iterations"]),
        tolerance=float(experiment["tolerance"]),
        replicates=int(experiment["bootstrap_replicates"]),
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
