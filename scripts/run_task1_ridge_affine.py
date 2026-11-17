#!/usr/bin/env python3
"""Evaluate leakage-safe ridge-affine calibration on frozen Task 0 outputs."""

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


Matrix = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--mean-offset-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def solve_3x3(matrix: list[list[float]], values: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise ValueError("Singular ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * reference
                for current, reference in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][3] for row in range(3)]


def fit_ridge_affine(
    selected: list[baseline.Observation],
    *,
    ridge_lambda: float,
    angle_scale_rad: float,
) -> Matrix:
    """Fit normalized target angles from predictions, shrunk to identity."""
    if not selected:
        raise ValueError("Cannot fit ridge-affine calibration without samples")
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    if angle_scale_rad <= 0:
        raise ValueError("angle_scale_rad must be positive")

    count = len(selected)
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [[0.0] * 2 for _ in range(3)]
    for item in selected:
        features = (
            item.pred_pitch / angle_scale_rad,
            item.pred_yaw / angle_scale_rad,
            1.0,
        )
        targets = (
            item.target_pitch / angle_scale_rad,
            item.target_yaw / angle_scale_rad,
        )
        for row in range(3):
            for column in range(3):
                xtx[row][column] += features[row] * features[column] / count
            for output in range(2):
                xty[row][output] += features[row] * targets[output] / count

    identity_target = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    for index in range(3):
        xtx[index][index] += ridge_lambda
        for output in range(2):
            xty[index][output] += ridge_lambda * identity_target[index][output]

    columns = [
        solve_3x3(xtx, [xty[row][output] for row in range(3)])
        for output in range(2)
    ]
    return tuple(
        (columns[0][row], columns[1][row]) for row in range(3)
    )  # type: ignore[return-value]


def predict_affine(
    item: baseline.Observation,
    coefficients: Matrix,
    *,
    angle_scale_rad: float,
) -> tuple[float, float]:
    features = (
        item.pred_pitch / angle_scale_rad,
        item.pred_yaw / angle_scale_rad,
        1.0,
    )
    pitch = sum(features[row] * coefficients[row][0] for row in range(3))
    yaw = sum(features[row] * coefficients[row][1] for row in range(3))
    return pitch * angle_scale_rad, yaw * angle_scale_rad


def mean_affine_error(
    evaluation: tuple[baseline.Observation, ...],
    coefficients: Matrix,
    *,
    angle_scale_rad: float,
) -> float:
    errors = []
    for item in evaluation:
        pitch, yaw = predict_affine(
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


def summarize_affine_error(
    evaluation: tuple[baseline.Observation, ...],
    coefficients: Matrix,
    *,
    angle_scale_rad: float,
) -> tuple[float, float, float]:
    errors = []
    for item in evaluation:
        pitch, yaw = predict_affine(
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
    return (
        statistics.fmean(errors),
        statistics.median(errors),
        baseline.quantile(errors, 0.95),
    )


def random_sample(
    split: baseline.SubjectSplit,
    *,
    seed: int,
    budget: int,
    draw: int,
) -> list[baseline.Observation]:
    generator = random.Random(
        baseline.stable_seed(seed, split.subject, budget, "random", draw)
    )
    return generator.sample(list(split.candidate), budget)


def select_ridge_lambda(
    splits: dict[str, baseline.SubjectSplit],
    *,
    held_out_subject: str,
    budget: int,
    lambdas: list[float],
    tuning_random_draws: int,
    seed: int,
    angle_scale_rad: float,
) -> tuple[float, dict[float, float]]:
    """Select lambda using only candidate/guard data from other subjects."""
    totals = {value: 0.0 for value in lambdas}
    cases = 0
    for subject, split in sorted(splits.items()):
        if subject == held_out_subject:
            continue
        for draw in range(tuning_random_draws):
            selected = random_sample(split, seed=seed, budget=budget, draw=draw)
            for ridge_lambda in lambdas:
                coefficients = fit_ridge_affine(
                    selected,
                    ridge_lambda=ridge_lambda,
                    angle_scale_rad=angle_scale_rad,
                )
                totals[ridge_lambda] += mean_affine_error(
                    split.guard,
                    coefficients,
                    angle_scale_rad=angle_scale_rad,
                )
            cases += 1
    means = {value: total / cases for value, total in totals.items()}
    return min(lambdas, key=lambda value: (means[value], -value)), means


def load_baseline_draws(path: Path) -> dict[tuple[str, str, int, str, int], float]:
    result: dict[tuple[str, str, int, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["subject"],
                row["selector"],
                int(row["budget"]),
                row["calibrator"],
                int(row["draw"]),
            )
            result[key] = float(row["calibrated_mean_deg"])
    return result


def aggregate_results(
    draw_rows: list[dict[str, Any]],
    *,
    baseline_draws: dict[tuple[str, str, int, str, int], float],
    budgets: list[int],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in draw_rows:
        grouped[(row["subject"], row["selector"], int(row["budget"]))].append(row)

    per_subject: list[dict[str, Any]] = []
    for (subject, selector, budget), rows in sorted(grouped.items()):
        ridge_errors = [float(row["calibrated_mean_deg"]) for row in rows]
        gains = [float(row["gain_deg"]) for row in rows]
        item: dict[str, Any] = {
            "subject": subject,
            "selector": selector,
            "budget": budget,
            "draws": len(rows),
            "ridge_lambda": rows[0]["ridge_lambda"],
            "uncalibrated_mean_deg": rows[0]["uncalibrated_mean_deg"],
            "calibrated_mean_deg": statistics.fmean(ridge_errors),
            "calibrated_draw_std_deg": (
                statistics.stdev(ridge_errors) if len(ridge_errors) > 1 else 0.0
            ),
            "mean_gain_deg": statistics.fmean(gains),
            "harmed_draw_fraction": statistics.fmean(gain < 0 for gain in gains),
        }
        for calibrator in ("mean_offset", "map_offset"):
            differences = []
            for row in rows:
                key = (subject, selector, budget, calibrator, int(row["draw"]))
                if key not in baseline_draws:
                    raise ValueError(f"Missing paired baseline draw: {key}")
                differences.append(
                    baseline_draws[key] - float(row["calibrated_mean_deg"])
                )
            item[f"advantage_over_{calibrator}_deg"] = statistics.fmean(differences)
        per_subject.append(item)

    aggregates: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for selector in ("random", "fixed_uniform"):
        for budget in budgets:
            matching = [
                row
                for row in per_subject
                if row["selector"] == selector and row["budget"] == budget
            ]
            errors = [float(row["calibrated_mean_deg"]) for row in matching]
            gains = [float(row["mean_gain_deg"]) for row in matching]
            gain_low, gain_high = baseline.bootstrap_mean_ci(
                gains,
                replicates=bootstrap_replicates,
                seed=baseline.stable_seed(seed, selector, budget, "ridge_gain"),
            )
            aggregates.append(
                {
                    "selector": selector,
                    "budget": budget,
                    "calibrator": "ridge_affine",
                    "subjects": len(matching),
                    "subject_mean_calibrated_deg": statistics.fmean(errors),
                    "subject_std_calibrated_deg": statistics.stdev(errors),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "gain_ci95_low_deg": gain_low,
                    "gain_ci95_high_deg": gain_high,
                    "harmed_subject_fraction": statistics.fmean(
                        gain < 0 for gain in gains
                    ),
                    "mean_within_subject_draw_std_deg": statistics.fmean(
                        float(row["calibrated_draw_std_deg"]) for row in matching
                    ),
                }
            )
            for calibrator in ("mean_offset", "map_offset"):
                differences = [
                    float(row[f"advantage_over_{calibrator}_deg"])
                    for row in matching
                ]
                low, high = baseline.bootstrap_mean_ci(
                    differences,
                    replicates=bootstrap_replicates,
                    seed=baseline.stable_seed(
                        seed, selector, budget, calibrator, "ridge_comparison"
                    ),
                )
                comparisons.append(
                    {
                        "comparison": f"ridge_advantage_over_{calibrator}",
                        "selector": selector,
                        "budget": budget,
                        "mean_difference_deg": statistics.fmean(differences),
                        "difference_ci95_low_deg": low,
                        "difference_ci95_high_deg": high,
                    }
                )
    return per_subject, aggregates, comparisons


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    budgets = [int(value) for value in experiment["budgets"]]
    random_draws = int(experiment["random_draws"])
    tuning_random_draws = int(experiment["tuning_random_draws"])
    seed = int(experiment["seed"])
    bootstrap_replicates = int(experiment["bootstrap_replicates"])
    lambdas = [float(value) for value in experiment["ridge_lambdas"]]
    angle_scale_rad = math.radians(float(experiment["angle_scale_deg"]))

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
    baseline_draws = load_baseline_draws(
        args.mean_offset_results / "per_draw_metrics.csv"
    )

    tuning_rows: list[dict[str, Any]] = []
    selected_lambdas: dict[tuple[str, int], float] = {}
    for subject in sorted(splits):
        for budget in budgets:
            selected, means = select_ridge_lambda(
                splits,
                held_out_subject=subject,
                budget=budget,
                lambdas=lambdas,
                tuning_random_draws=tuning_random_draws,
                seed=seed,
                angle_scale_rad=angle_scale_rad,
            )
            selected_lambdas[(subject, budget)] = selected
            for ridge_lambda in lambdas:
                tuning_rows.append(
                    {
                        "held_out_subject": subject,
                        "budget": budget,
                        "ridge_lambda": ridge_lambda,
                        "guard_mean_deg": means[ridge_lambda],
                        "selected": ridge_lambda == selected,
                        "tuning_subjects": len(splits) - 1,
                        "tuning_random_draws": tuning_random_draws,
                    }
                )

    draw_rows: list[dict[str, Any]] = []
    for subject, split in sorted(splits.items()):
        uncalibrated = statistics.fmean(
            item.uncalibrated_error_deg for item in split.evaluation
        )
        for budget in budgets:
            ridge_lambda = selected_lambdas[(subject, budget)]
            for selector in ("random", "fixed_uniform"):
                draws = random_draws if selector == "random" else 1
                for draw in range(draws):
                    selected = (
                        random_sample(split, seed=seed, budget=budget, draw=draw)
                        if selector == "random"
                        else baseline.fixed_uniform_sample(split.candidate, budget)
                    )
                    coefficients = fit_ridge_affine(
                        selected,
                        ridge_lambda=ridge_lambda,
                        angle_scale_rad=angle_scale_rad,
                    )
                    mean_error, median_error, p95_error = summarize_affine_error(
                        split.evaluation,
                        coefficients,
                        angle_scale_rad=angle_scale_rad,
                    )
                    draw_rows.append(
                        {
                            "subject": subject,
                            "selector": selector,
                            "budget": budget,
                            "calibrator": "ridge_affine",
                            "draw": draw,
                            "ridge_lambda": ridge_lambda,
                            "uncalibrated_mean_deg": uncalibrated,
                            "calibrated_mean_deg": mean_error,
                            "calibrated_median_deg": median_error,
                            "calibrated_p95_deg": p95_error,
                            "gain_deg": uncalibrated - mean_error,
                        }
                    )

    per_subject, aggregates, comparisons = aggregate_results(
        draw_rows,
        baseline_draws=baseline_draws,
        budgets=budgets,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary = {
        "status": "complete",
        "experiment": experiment["name"],
        "subjects": sorted(splits),
        "budgets": budgets,
        "random_draws": random_draws,
        "tuning_random_draws": tuning_random_draws,
        "ridge_lambdas": lambdas,
        "angle_scale_deg": float(experiment["angle_scale_deg"]),
        "hyperparameter_protocol": (
            "For each held-out subject and budget, lambda is selected by mean "
            "guard error over random draws from the other 14 subjects only."
        ),
        "best_aggregate_result": min(
            aggregates,
            key=lambda row: float(row["subject_mean_calibrated_deg"]),
        ),
        "input_predictions": str(args.predictions),
        "mean_offset_results": str(args.mean_offset_results),
        "config": str(args.config),
        "limitations": [
            "The temporal split inherits the minimum-sample-index approximation.",
            "Ridge tuning reuses fixed Task 0 predictions from the other subjects.",
            "The affine map uses prediction angles only; head pose is not available.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.write_csv(args.output_dir / "ridge_tuning.csv", tuning_rows)
    baseline.write_csv(args.output_dir / "per_draw_metrics.csv", draw_rows)
    baseline.write_csv(args.output_dir / "per_subject_metrics.csv", per_subject)
    baseline.write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    baseline.write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
