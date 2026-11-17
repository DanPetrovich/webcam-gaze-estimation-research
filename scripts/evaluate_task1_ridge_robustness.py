#!/usr/bin/env python3
"""Compare ridge-affine and hybrid Huber calibration by target eccentricity."""

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
import run_task1_ridge_affine as ridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ridge-results", required=True, type=Path)
    parser.add_argument("--hybrid-slice-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_ridge_draws(path: Path, budgets: list[int]) -> list[dict[str, Any]]:
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
            "ridge_lambda": float(row["ridge_lambda"]),
        }
        for row in rows
    ]


def load_hybrid_reference(path: Path) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["subject"], int(row["budget"]), row["slice"]): float(
                row["calibrated_mean_deg"]
            )
            for row in csv.DictReader(stream)
            if int(row["budget"]) > 1
        }


def affine_slice_error(
    observations: list[baseline.Observation],
    coefficients: ridge.Matrix,
    *,
    angle_scale_rad: float,
) -> float:
    errors = []
    for item in observations:
        pitch, yaw = ridge.predict_affine(
            item,
            coefficients,
            angle_scale_rad=angle_scale_rad,
        )
        errors.append(
            baseline.angular_error_deg(
                pitch,
                yaw,
                item.target_pitch,
                item.target_yaw,
            )
        )
    return statistics.fmean(errors)


def evaluate(
    splits: dict[str, baseline.SubjectSplit],
    ridge_draws: list[dict[str, Any]],
    hybrid_reference: dict[tuple[str, int, str], float],
    *,
    budgets: list[int],
    central_max: float,
    mid_max: float,
    seed: int,
    angle_scale_rad: float,
    replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    draws_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ridge_draws:
        draws_by_key[(str(row["subject"]), int(row["budget"]))].append(row)

    subject_rows = []
    for subject, split in sorted(splits.items()):
        slices: dict[str, list[baseline.Observation]] = defaultdict(list)
        for item in split.evaluation:
            name = hybrid_slices.slice_name(
                hybrid_slices.target_eccentricity_deg(item), central_max, mid_max
            )
            slices[name].append(item)
        for budget in budgets:
            errors: dict[str, list[float]] = defaultdict(list)
            for row in draws_by_key[(subject, budget)]:
                selected = ridge.random_sample(
                    split,
                    seed=seed,
                    budget=budget,
                    draw=int(row["draw"]),
                )
                coefficients = ridge.fit_ridge_affine(
                    selected,
                    ridge_lambda=float(row["ridge_lambda"]),
                    angle_scale_rad=angle_scale_rad,
                )
                for name, observations in slices.items():
                    errors[name].append(
                        affine_slice_error(
                            observations,
                            coefficients,
                            angle_scale_rad=angle_scale_rad,
                        )
                    )
            for name, observations in sorted(slices.items()):
                if not errors[name]:
                    raise ValueError("Ridge slice draws are missing")
                uncalibrated = statistics.fmean(
                    item.uncalibrated_error_deg for item in observations
                )
                calibrated = statistics.fmean(errors[name])
                huber_error = hybrid_reference[(subject, budget, name)]
                subject_rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "slice": name,
                        "unique_groups": len(observations),
                        "uncalibrated_mean_deg": uncalibrated,
                        "ridge_mean_deg": calibrated,
                        "ridge_gain_deg": uncalibrated - calibrated,
                        "huber_mean_deg": huber_error,
                        "ridge_advantage_over_huber_deg": huber_error - calibrated,
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
            differences = [
                float(row["ridge_advantage_over_huber_deg"]) for row in matching
            ]
            low, high = baseline.bootstrap_mean_ci(
                differences,
                replicates=replicates,
                seed=baseline.stable_seed(seed, "ridge_slice", budget, name),
            )
            aggregates.append(
                {
                    "budget": budget,
                    "slice": name,
                    "subjects": len(matching),
                    "subject_mean_uncalibrated_deg": statistics.fmean(
                        float(row["uncalibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_ridge_deg": statistics.fmean(
                        float(row["ridge_mean_deg"]) for row in matching
                    ),
                    "subject_mean_huber_deg": statistics.fmean(
                        float(row["huber_mean_deg"]) for row in matching
                    ),
                    "ridge_advantage_over_huber_deg": statistics.fmean(differences),
                    "advantage_ci95_low_deg": low,
                    "advantage_ci95_high_deg": high,
                    "ridge_harmed_subject_fraction": statistics.fmean(
                        float(row["ridge_gain_deg"]) < 0.0 for row in matching
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
        load_ridge_draws(args.ridge_results / "per_draw_metrics.csv", budgets),
        load_hybrid_reference(
            args.hybrid_slice_results / "per_subject_slice_metrics.csv"
        ),
        budgets=budgets,
        central_max=float(slices_config["central_max_deg"]),
        mid_max=float(slices_config["mid_max_deg"]),
        seed=int(experiment["seed"]),
        angle_scale_rad=math.radians(float(experiment["angle_scale_deg"])),
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
