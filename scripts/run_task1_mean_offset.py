#!/usr/bin/env python3
"""Run duplicate-safe few-shot mean/MAP offset calibration on Task 0 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_SUBJECTS = [f"p{index:02d}" for index in range(15)]


@dataclass(frozen=True)
class Observation:
    group_id: str
    min_sample_index: int
    multiplicity: int
    target_pitch: float
    target_yaw: float
    pred_pitch: float
    pred_yaw: float
    uncalibrated_error_deg: float

    @property
    def pitch_residual(self) -> float:
        return self.target_pitch - self.pred_pitch

    @property
    def yaw_residual(self) -> float:
        return self.target_yaw - self.pred_yaw


@dataclass(frozen=True)
class SubjectSplit:
    subject: str
    candidate: tuple[Observation, ...]
    guard: tuple[Observation, ...]
    evaluation: tuple[Observation, ...]


@dataclass(frozen=True)
class MapPrior:
    pitch_mean: float
    yaw_mean: float
    pitch_between_variance: float
    yaw_between_variance: float
    pitch_noise_variance: float
    yaw_noise_variance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate quantile of an empty collection")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def bootstrap_mean_ci(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(values)
    means = [
        sum(generator.choice(values) for _ in range(size)) / size
        for _ in range(replicates)
    ]
    return quantile(means, 0.025), quantile(means, 0.975)


def pitchyaw_to_vector(pitch: float, yaw: float) -> tuple[float, float, float]:
    return (
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    )


def angular_error_deg(
    pred_pitch: float,
    pred_yaw: float,
    target_pitch: float,
    target_yaw: float,
) -> float:
    prediction = pitchyaw_to_vector(pred_pitch, pred_yaw)
    target = pitchyaw_to_vector(target_pitch, target_yaw)
    cosine = sum(a * b for a, b in zip(prediction, target, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def load_unique_observations(path: Path) -> dict[str, tuple[Observation, ...]]:
    grouped: dict[str, dict[tuple[str, ...], list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["target_pitch_rad"],
                row["target_yaw_rad"],
                row["pred_pitch_rad"],
                row["pred_yaw_rad"],
            )
            grouped[row["subject"]][key].append(row)

    result: dict[str, tuple[Observation, ...]] = {}
    for subject, subject_groups in sorted(grouped.items()):
        observations: list[Observation] = []
        for key, rows in subject_groups.items():
            errors = {row["angular_error_deg"] for row in rows}
            if len(errors) != 1:
                raise ValueError(f"Duplicate group has inconsistent errors: {subject}")
            reported_error = float(next(iter(errors)))
            recomputed_error = angular_error_deg(
                float(key[2]),
                float(key[3]),
                float(key[0]),
                float(key[1]),
            )
            # Predictions are serialized from mixed precision, so recomputation
            # from rounded CSV angles can differ by a few thousandths of a degree.
            if not math.isclose(reported_error, recomputed_error, abs_tol=1e-2):
                raise ValueError(f"Angular error mismatch for {subject}")
            group_id = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:16]
            observations.append(
                Observation(
                    group_id=group_id,
                    min_sample_index=min(int(row["sample_index"]) for row in rows),
                    multiplicity=len(rows),
                    target_pitch=float(key[0]),
                    target_yaw=float(key[1]),
                    pred_pitch=float(key[2]),
                    pred_yaw=float(key[3]),
                    uncalibrated_error_deg=reported_error,
                )
            )
        result[subject] = tuple(
            sorted(observations, key=lambda observation: observation.min_sample_index)
        )
    return result


def split_subject(
    subject: str,
    observations: tuple[Observation, ...],
    *,
    candidate_fraction: float,
    guard_fraction: float,
) -> SubjectSplit:
    if candidate_fraction <= 0 or guard_fraction < 0:
        raise ValueError("Split fractions must be non-negative and candidate > 0")
    if candidate_fraction + guard_fraction >= 1:
        raise ValueError("Candidate and guard fractions must leave evaluation data")
    candidate_end = math.floor(len(observations) * candidate_fraction)
    guard_end = candidate_end + math.floor(len(observations) * guard_fraction)
    if candidate_end == 0 or guard_end >= len(observations):
        raise ValueError(f"Invalid split sizes for {subject}")
    return SubjectSplit(
        subject=subject,
        candidate=observations[:candidate_end],
        guard=observations[candidate_end:guard_end],
        evaluation=observations[guard_end:],
    )


def variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def estimate_map_prior(
    splits: dict[str, SubjectSplit],
    held_out_subject: str,
) -> MapPrior:
    pitch_means: list[float] = []
    yaw_means: list[float] = []
    pitch_variances: list[float] = []
    yaw_variances: list[float] = []
    candidate_counts: list[int] = []
    for subject, split in splits.items():
        if subject == held_out_subject:
            continue
        pitch = [observation.pitch_residual for observation in split.candidate]
        yaw = [observation.yaw_residual for observation in split.candidate]
        pitch_means.append(statistics.fmean(pitch))
        yaw_means.append(statistics.fmean(yaw))
        pitch_variances.append(variance(pitch))
        yaw_variances.append(variance(yaw))
        candidate_counts.append(len(split.candidate))

    pitch_noise = statistics.fmean(pitch_variances)
    yaw_noise = statistics.fmean(yaw_variances)
    pitch_measurement_noise = statistics.fmean(
        item / count
        for item, count in zip(pitch_variances, candidate_counts, strict=True)
    )
    yaw_measurement_noise = statistics.fmean(
        item / count
        for item, count in zip(yaw_variances, candidate_counts, strict=True)
    )
    epsilon = 1e-12
    return MapPrior(
        pitch_mean=statistics.fmean(pitch_means),
        yaw_mean=statistics.fmean(yaw_means),
        pitch_between_variance=max(
            variance(pitch_means) - pitch_measurement_noise,
            epsilon,
        ),
        yaw_between_variance=max(
            variance(yaw_means) - yaw_measurement_noise,
            epsilon,
        ),
        pitch_noise_variance=max(pitch_noise, epsilon),
        yaw_noise_variance=max(yaw_noise, epsilon),
    )


def fixed_uniform_sample(
    candidates: tuple[Observation, ...], budget: int
) -> list[Observation]:
    if budget > len(candidates):
        raise ValueError("Budget exceeds candidate pool")
    indices = [
        min(len(candidates) - 1, math.floor((index + 0.5) * len(candidates) / budget))
        for index in range(budget)
    ]
    if len(set(indices)) != budget:
        raise RuntimeError("Fixed selector produced duplicate indices")
    return [candidates[index] for index in indices]


def calibration_offsets(
    selected: list[Observation],
    *,
    calibrator: str,
    prior: MapPrior,
) -> tuple[float, float, float, float]:
    pitch_mean = statistics.fmean(item.pitch_residual for item in selected)
    yaw_mean = statistics.fmean(item.yaw_residual for item in selected)
    if calibrator == "mean_offset":
        return pitch_mean, yaw_mean, 1.0, 1.0
    if calibrator != "map_offset":
        raise ValueError(f"Unknown calibrator: {calibrator}")
    budget = len(selected)
    pitch_alpha = (
        budget
        * prior.pitch_between_variance
        / (budget * prior.pitch_between_variance + prior.pitch_noise_variance)
    )
    yaw_alpha = (
        budget
        * prior.yaw_between_variance
        / (budget * prior.yaw_between_variance + prior.yaw_noise_variance)
    )
    pitch_offset = pitch_alpha * pitch_mean + (1 - pitch_alpha) * prior.pitch_mean
    yaw_offset = yaw_alpha * yaw_mean + (1 - yaw_alpha) * prior.yaw_mean
    return pitch_offset, yaw_offset, pitch_alpha, yaw_alpha


def summarize_evaluation(
    evaluation: tuple[Observation, ...],
    pitch_offset: float,
    yaw_offset: float,
) -> tuple[float, float, float]:
    errors = [
        angular_error_deg(
            item.pred_pitch + pitch_offset,
            item.pred_yaw + yaw_offset,
            item.target_pitch,
            item.target_yaw,
        )
        for item in evaluation
    ]
    return statistics.fmean(errors), statistics.median(errors), quantile(errors, 0.95)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_rows(
    draw_rows: list[dict[str, Any]],
    *,
    budgets: list[int],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_subject_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in draw_rows:
        key = (
            row["subject"],
            row["selector"],
            int(row["budget"]),
            row["calibrator"],
        )
        grouped[key].append(row)

    for key, rows in sorted(grouped.items()):
        subject, selector, budget, calibrator = key
        calibrated = [float(row["calibrated_mean_deg"]) for row in rows]
        gains = [float(row["gain_deg"]) for row in rows]
        per_subject_rows.append(
            {
                "subject": subject,
                "selector": selector,
                "budget": budget,
                "calibrator": calibrator,
                "draws": len(rows),
                "uncalibrated_mean_deg": rows[0]["uncalibrated_mean_deg"],
                "calibrated_mean_deg": statistics.fmean(calibrated),
                "calibrated_draw_std_deg": (
                    statistics.stdev(calibrated) if len(calibrated) > 1 else 0.0
                ),
                "mean_gain_deg": statistics.fmean(gains),
                "harmed_draw_fraction": statistics.fmean(gain < 0 for gain in gains),
            }
        )

    for selector in ("random", "fixed_uniform"):
        for budget in budgets:
            for calibrator in ("mean_offset", "map_offset"):
                matching = [
                    row
                    for row in per_subject_rows
                    if row["selector"] == selector
                    and row["budget"] == budget
                    and row["calibrator"] == calibrator
                ]
                if not matching:
                    continue
                errors = [float(row["calibrated_mean_deg"]) for row in matching]
                gains = [float(row["mean_gain_deg"]) for row in matching]
                ci_low, ci_high = bootstrap_mean_ci(
                    gains,
                    replicates=bootstrap_replicates,
                    seed=stable_seed(seed, selector, budget, calibrator, "bootstrap"),
                )
                aggregate_rows.append(
                    {
                        "selector": selector,
                        "budget": budget,
                        "calibrator": calibrator,
                        "subjects": len(matching),
                        "subject_mean_calibrated_deg": statistics.fmean(errors),
                        "subject_std_calibrated_deg": statistics.stdev(errors),
                        "subject_mean_gain_deg": statistics.fmean(gains),
                        "gain_ci95_low_deg": ci_low,
                        "gain_ci95_high_deg": ci_high,
                        "harmed_subject_fraction": statistics.fmean(
                            gain < 0 for gain in gains
                        ),
                        "mean_within_subject_draw_std_deg": statistics.fmean(
                            float(row["calibrated_draw_std_deg"]) for row in matching
                        ),
                    }
                )
    return per_subject_rows, aggregate_rows


def build_comparison_rows(
    per_subject_rows: list[dict[str, Any]],
    *,
    budgets: list[int],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["subject"],
            row["selector"],
            int(row["budget"]),
            row["calibrator"],
        ): row
        for row in per_subject_rows
    }
    subjects = sorted({str(row["subject"]) for row in per_subject_rows})
    comparisons: list[dict[str, Any]] = []

    for selector in ("random", "fixed_uniform"):
        for budget in budgets:
            differences = [
                float(
                    indexed[(subject, selector, budget, "mean_offset")][
                        "calibrated_mean_deg"
                    ]
                )
                - float(
                    indexed[(subject, selector, budget, "map_offset")][
                        "calibrated_mean_deg"
                    ]
                )
                for subject in subjects
            ]
            low, high = bootstrap_mean_ci(
                differences,
                replicates=bootstrap_replicates,
                seed=stable_seed(seed, selector, budget, "map_vs_mean"),
            )
            comparisons.append(
                {
                    "comparison": "map_advantage_over_mean",
                    "selector": selector,
                    "budget": budget,
                    "calibrator": "map_offset",
                    "mean_difference_deg": statistics.fmean(differences),
                    "difference_ci95_low_deg": low,
                    "difference_ci95_high_deg": high,
                }
            )

    for calibrator in ("mean_offset", "map_offset"):
        for budget in budgets:
            differences = [
                float(
                    indexed[(subject, "random", budget, calibrator)][
                        "calibrated_mean_deg"
                    ]
                )
                - float(
                    indexed[(subject, "fixed_uniform", budget, calibrator)][
                        "calibrated_mean_deg"
                    ]
                )
                for subject in subjects
            ]
            low, high = bootstrap_mean_ci(
                differences,
                replicates=bootstrap_replicates,
                seed=stable_seed(seed, calibrator, budget, "fixed_vs_random"),
            )
            comparisons.append(
                {
                    "comparison": "fixed_advantage_over_random",
                    "selector": "fixed_uniform",
                    "budget": budget,
                    "calibrator": calibrator,
                    "mean_difference_deg": statistics.fmean(differences),
                    "difference_ci95_low_deg": low,
                    "difference_ci95_high_deg": high,
                }
            )
    return comparisons


def save_curve_svg(
    aggregate_rows: list[dict[str, Any]],
    uncalibrated_mean: float,
    output_path: Path,
) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 75, 260, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    budgets = sorted({int(row["budget"]) for row in aggregate_rows})
    maximum = (
        max(
            uncalibrated_mean,
            *(float(row["subject_mean_calibrated_deg"]) for row in aggregate_rows),
        )
        * 1.12
    )
    x_position = {
        budget: left + index * plot_width / (len(budgets) - 1)
        for index, budget in enumerate(budgets)
    }

    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#1e293b}</style>",
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            'font-size="20" font-weight="700">Task 1: few-shot offset '
            "calibration</text>"
        ),
    ]
    for tick in range(0, math.ceil(maximum) + 1):
        y = top + plot_height * (1 - tick / maximum)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12">{tick}</text>'
        )
    uncalibrated_y = top + plot_height * (1 - uncalibrated_mean / maximum)
    elements.append(
        f'<line x1="{left}" y1="{uncalibrated_y:.2f}" x2="{width - right}" '
        f'y2="{uncalibrated_y:.2f}" stroke="#64748b" stroke-dasharray="7 5"/>'
    )

    styles = {
        ("random", "mean_offset"): ("#2563eb", "none"),
        ("random", "map_offset"): ("#16a34a", "none"),
        ("fixed_uniform", "mean_offset"): ("#2563eb", "6 4"),
        ("fixed_uniform", "map_offset"): ("#16a34a", "6 4"),
    }
    for key, (color, dash) in styles.items():
        selector, calibrator = key
        rows = sorted(
            (
                row
                for row in aggregate_rows
                if row["selector"] == selector and row["calibrator"] == calibrator
            ),
            key=lambda row: int(row["budget"]),
        )
        points = []
        for row in rows:
            x = x_position[int(row["budget"])]
            value = float(row["subject_mean_calibrated_deg"])
            y = top + plot_height * (1 - value / maximum)
            points.append(f"{x:.2f},{y:.2f}")
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        dash_attribute = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"{dash_attribute}/>'
        )
    legend_labels = {
        ("random", "mean_offset"): "random + mean",
        ("random", "map_offset"): "random + MAP",
        ("fixed_uniform", "mean_offset"): "fixed + mean",
        ("fixed_uniform", "map_offset"): "fixed + MAP",
    }
    legend_x = width - right + 30
    for index, (key, (color, dash)) in enumerate(styles.items()):
        y = top + 25 + index * 28
        dash_attribute = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 38}" y2="{y}" '
            f'stroke="{color}" stroke-width="2.5"{dash_attribute}/>'
        )
        elements.append(
            f'<text x="{legend_x + 48}" y="{y + 4}" font-size="13">'
            f"{legend_labels[key]}</text>"
        )
    legend_y = top + 25 + len(styles) * 28
    elements.append(
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 38}" '
        f'y2="{legend_y}" stroke="#64748b" stroke-dasharray="7 5"/>'
    )
    elements.append(
        f'<text x="{legend_x + 48}" y="{legend_y + 4}" font-size="13">'
        "uncalibrated</text>"
    )
    for budget, x in x_position.items():
        elements.append(
            f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" '
            f'font-size="13">{budget}</text>'
        )
    elements.append(
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" '
        'font-size="14">Calibration budget k</text>'
    )
    output_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    budgets = [int(value) for value in experiment["budgets"]]
    random_draws = int(experiment["random_draws"])
    seed = int(experiment["seed"])
    bootstrap_replicates = int(experiment["bootstrap_replicates"])

    observations = load_unique_observations(args.predictions)
    if sorted(observations) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected subjects {EXPECTED_SUBJECTS}, got {sorted(observations)}"
        )
    splits = {
        subject: split_subject(
            subject,
            values,
            candidate_fraction=float(experiment["candidate_fraction"]),
            guard_fraction=float(experiment["guard_fraction"]),
        )
        for subject, values in observations.items()
    }
    if max(budgets) > min(len(split.candidate) for split in splits.values()):
        raise ValueError("Largest budget exceeds a candidate pool")

    split_rows: list[dict[str, Any]] = []
    for subject, split in splits.items():
        for partition, items in (
            ("candidate", split.candidate),
            ("guard", split.guard),
            ("evaluation", split.evaluation),
        ):
            for item in items:
                split_rows.append(
                    {
                        "subject": subject,
                        "group_id": item.group_id,
                        "partition": partition,
                        "min_sample_index": item.min_sample_index,
                        "multiplicity": item.multiplicity,
                    }
                )

    draw_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    uncalibrated_subject_means: dict[str, float] = {}
    for subject, split in splits.items():
        prior = estimate_map_prior(splits, subject)
        uncalibrated_mean = statistics.fmean(
            item.uncalibrated_error_deg for item in split.evaluation
        )
        uncalibrated_subject_means[subject] = uncalibrated_mean
        for budget in budgets:
            pitch_alpha = (
                budget
                * prior.pitch_between_variance
                / (budget * prior.pitch_between_variance + prior.pitch_noise_variance)
            )
            yaw_alpha = (
                budget
                * prior.yaw_between_variance
                / (budget * prior.yaw_between_variance + prior.yaw_noise_variance)
            )
            prior_rows.append(
                {
                    "subject": subject,
                    "budget": budget,
                    **prior.__dict__,
                    "pitch_alpha": pitch_alpha,
                    "yaw_alpha": yaw_alpha,
                }
            )
            for selector in ("random", "fixed_uniform"):
                draws = random_draws if selector == "random" else 1
                for draw in range(draws):
                    if selector == "random":
                        generator = random.Random(
                            stable_seed(seed, subject, budget, selector, draw)
                        )
                        selected = generator.sample(list(split.candidate), budget)
                    else:
                        selected = fixed_uniform_sample(split.candidate, budget)
                    for calibrator in ("mean_offset", "map_offset"):
                        pitch_offset, yaw_offset, p_alpha, y_alpha = (
                            calibration_offsets(
                                selected,
                                calibrator=calibrator,
                                prior=prior,
                            )
                        )
                        mean_error, median_error, p95_error = summarize_evaluation(
                            split.evaluation,
                            pitch_offset,
                            yaw_offset,
                        )
                        draw_rows.append(
                            {
                                "subject": subject,
                                "selector": selector,
                                "budget": budget,
                                "calibrator": calibrator,
                                "draw": draw,
                                "uncalibrated_mean_deg": uncalibrated_mean,
                                "calibrated_mean_deg": mean_error,
                                "calibrated_median_deg": median_error,
                                "calibrated_p95_deg": p95_error,
                                "gain_deg": uncalibrated_mean - mean_error,
                                "pitch_offset_deg": math.degrees(pitch_offset),
                                "yaw_offset_deg": math.degrees(yaw_offset),
                                "pitch_alpha": p_alpha,
                                "yaw_alpha": y_alpha,
                            }
                        )

    per_subject_rows, aggregate_rows = build_aggregate_rows(
        draw_rows,
        budgets=budgets,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    comparison_rows = build_comparison_rows(
        per_subject_rows,
        budgets=budgets,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    uncalibrated_mean = statistics.fmean(uncalibrated_subject_means.values())
    random_one_shot_map = next(
        row
        for row in comparison_rows
        if row["comparison"] == "map_advantage_over_mean"
        and row["selector"] == "random"
        and row["budget"] == 1
    )
    summary = {
        "status": "complete",
        "experiment": experiment["name"],
        "protocol": (
            "duplicate-grouped temporal 20% candidate / 10% guard / 70% "
            "evaluation; leave-one-subject-out MAP prior"
        ),
        "subjects": sorted(splits),
        "budgets": budgets,
        "random_draws": random_draws,
        "uncalibrated_evaluation_subject_mean_deg": uncalibrated_mean,
        "raw_prediction_rows": sum(
            observation.multiplicity
            for values in observations.values()
            for observation in values
        ),
        "unique_observation_groups": sum(map(len, observations.values())),
        "unique_observations": {
            subject: len(values) for subject, values in observations.items()
        },
        "split_sizes": {
            subject: {
                "candidate": len(split.candidate),
                "guard": len(split.guard),
                "evaluation": len(split.evaluation),
            }
            for subject, split in splits.items()
        },
        "best_aggregate_result": min(
            aggregate_rows,
            key=lambda row: float(row["subject_mean_calibrated_deg"]),
        ),
        "random_one_shot_map_advantage_over_mean": random_one_shot_map,
        "input_predictions": str(args.predictions),
        "config": str(args.config),
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "limitations": [
            "Temporal order is approximated by the minimum saved sample index.",
            "Exact duplicates are grouped, but adjacent near-duplicates cannot "
            "be identified without raw session metadata.",
            "This reuses fixed Task 0 predictions and calibrates output angles; "
            "it does not retrain UniGaze.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_manifest.csv", split_rows)
    write_csv(args.output_dir / "map_priors.csv", prior_rows)
    write_csv(args.output_dir / "per_draw_metrics.csv", draw_rows)
    write_csv(args.output_dir / "per_subject_metrics.csv", per_subject_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    save_curve_svg(
        aggregate_rows,
        uncalibrated_mean,
        args.output_dir / "calibration_curve.svg",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
