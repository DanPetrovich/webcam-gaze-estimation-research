#!/usr/bin/env python3
"""Aggregate completed Task 0 LOPO shards into one reproducible baseline report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXPECTED_SUBJECTS = [f"p{index:02d}" for index in range(15)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Shard result directory containing metrics directly or in metrics/.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_path(run_dir: Path, filename: str) -> Path:
    candidates = (run_dir / "metrics" / filename, run_dir / filename)
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} in {run_dir} or metrics/, got {matches}"
        )
    return matches[0]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty collection")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def bootstrap_mean_ci(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Bootstrap requires at least two values")
    generator = random.Random(seed)
    sample_size = len(values)
    means = [
        sum(generator.choice(values) for _ in range(sample_size)) / sample_size
        for _ in range(replicates)
    ]
    return quantile(means, 0.025), quantile(means, 0.975)


def distribution_metrics(values: list[float], *, sample_std: bool) -> dict[str, float]:
    return {
        "mean_deg": statistics.fmean(values),
        "median_deg": statistics.median(values),
        "std_deg": statistics.stdev(values)
        if sample_std
        else statistics.pstdev(values),
        "p90_deg": quantile(values, 0.90),
        "p95_deg": quantile(values, 0.95),
    }


def validate_inputs(
    per_subject: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    observed_subjects = sorted(row["test_subject"] for row in per_subject)
    if observed_subjects != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected subjects {EXPECTED_SUBJECTS}, observed {observed_subjects}"
        )
    if any(int(row["samples"]) != 3000 for row in per_subject):
        raise ValueError("Every fold must contain exactly 3000 test samples")
    if len(predictions) != 45_000:
        raise ValueError(f"Expected 45000 predictions, observed {len(predictions)}")

    keys = [(row["subject"], int(row["sample_index"])) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (subject, sample_index) prediction rows")

    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_subject[row["subject"]].append(row)
    if sorted(by_subject) != EXPECTED_SUBJECTS:
        raise ValueError("Prediction subjects do not match the 15 LOPO folds")
    if any(len(rows) != 3000 for rows in by_subject.values()):
        raise ValueError("Every subject must have exactly 3000 prediction rows")

    reported = {row["test_subject"]: float(row["mean_deg"]) for row in per_subject}
    for subject, rows in by_subject.items():
        recalculated = statistics.fmean(float(row["angular_error_deg"]) for row in rows)
        if not math.isclose(recalculated, reported[subject], abs_tol=1e-5):
            raise ValueError(f"Reported mean does not match predictions for {subject}")


def build_hard_subject_analysis(
    predictions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_subject[row["subject"]].append(row)

    results: list[dict[str, Any]] = []
    for subject, rows in sorted(by_subject.items()):
        errors = [float(row["angular_error_deg"]) for row in rows]
        pitch_targets = [math.degrees(float(row["target_pitch_rad"])) for row in rows]
        yaw_targets = [math.degrees(float(row["target_yaw_rad"])) for row in rows]
        pitch_residuals = [
            math.degrees(float(row["pred_pitch_rad"]) - float(row["target_pitch_rad"]))
            for row in rows
        ]
        yaw_residuals = [
            math.degrees(float(row["pred_yaw_rad"]) - float(row["target_yaw_rad"]))
            for row in rows
        ]
        magnitudes = [
            math.hypot(pitch, yaw)
            for pitch, yaw in zip(pitch_targets, yaw_targets, strict=True)
        ]
        results.append(
            {
                "subject": subject,
                "mean_error_deg": statistics.fmean(errors),
                "median_error_deg": statistics.median(errors),
                "p95_error_deg": quantile(errors, 0.95),
                "fraction_error_gt_5deg": statistics.fmean(
                    error > 5.0 for error in errors
                ),
                "fraction_error_gt_10deg": statistics.fmean(
                    error > 10.0 for error in errors
                ),
                "mean_abs_target_pitch_deg": statistics.fmean(map(abs, pitch_targets)),
                "mean_abs_target_yaw_deg": statistics.fmean(map(abs, yaw_targets)),
                "mean_target_magnitude_deg": statistics.fmean(magnitudes),
                "mean_pitch_bias_deg": statistics.fmean(pitch_residuals),
                "mean_yaw_bias_deg": statistics.fmean(yaw_residuals),
                "pitch_mae_deg": statistics.fmean(map(abs, pitch_residuals)),
                "yaw_mae_deg": statistics.fmean(map(abs, yaw_residuals)),
            }
        )
    return sorted(results, key=lambda row: row["mean_error_deg"], reverse=True)


def build_magnitude_slices(predictions: list[dict[str, str]]) -> list[dict[str, Any]]:
    slices: dict[str, list[float]] = {"<10deg": [], "10-20deg": [], ">=20deg": []}
    for row in predictions:
        magnitude = math.hypot(
            math.degrees(float(row["target_pitch_rad"])),
            math.degrees(float(row["target_yaw_rad"])),
        )
        label = (
            "<10deg" if magnitude < 10 else "10-20deg" if magnitude < 20 else ">=20deg"
        )
        slices[label].append(float(row["angular_error_deg"]))
    return [
        {
            "target_magnitude_bin": label,
            "samples": len(values),
            "mean_deg": statistics.fmean(values),
            "median_deg": statistics.median(values),
            "std_deg": statistics.stdev(values),
        }
        for label, values in slices.items()
        if values
    ]


def save_subject_plot(per_subject: list[dict[str, str]], output_path: Path) -> None:
    ordered = sorted(per_subject, key=lambda row: row["test_subject"])
    width, height = 1200, 560
    left, right, top, bottom = 85, 25, 55, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(float(row["constant_baseline_mean_deg"]) for row in ordered) * 1.08
    group_width = plot_width / len(ordered)
    bar_width = group_width * 0.33
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        (
            "<style>text{font-family:Arial,sans-serif;fill:#1e293b}"
            ".label{font-size:13px}.tick{font-size:12px}"
            ".title{font-size:20px;font-weight:700}</style>"
        ),
        (
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            'class="title">Task 0: identity-disjoint 15-fold LOPO</text>'
        ),
    ]
    for tick in range(0, math.ceil(maximum) + 1, 2):
        y = top + plot_height * (1 - tick / maximum)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="tick">{tick}</text>'
        )
    for index, row in enumerate(ordered):
        center = left + group_width * (index + 0.5)
        model = float(row["mean_deg"])
        baseline = float(row["constant_baseline_mean_deg"])
        for value, offset, color in (
            (model, -bar_width, "#3B82F6"),
            (baseline, 0, "#CBD5E1"),
        ):
            bar_height = plot_height * value / maximum
            elements.append(
                f'<rect x="{center + offset:.2f}" '
                f'y="{top + plot_height - bar_height:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{color}"/>'
            )
        elements.append(
            f'<text x="{center:.2f}" y="{height - 45}" '
            f'text-anchor="middle" class="label">{row["test_subject"]}</text>'
        )
    elements.extend(
        [
            (
                f'<text x="20" y="{top + plot_height / 2}" '
                f'transform="rotate(-90 20 {top + plot_height / 2})" '
                'text-anchor="middle" class="label">Mean angular error '
                "(degrees; lower is better)</text>"
            ),
            (
                f'<rect x="{left}" y="{height - 25}" width="14" height="14" '
                f'fill="#3B82F6"/><text x="{left + 20}" y="{height - 13}" '
                'class="label">UniGaze fine-tuned</text>'
            ),
            (
                f'<rect x="{left + 190}" y="{height - 25}" width="14" '
                f'height="14" fill="#CBD5E1"/><text x="{left + 210}" '
                f'y="{height - 13}" class="label">constant median-angle '
                "baseline</text>"
            ),
        ]
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
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    per_subject_fields: list[str] | None = None
    prediction_fields: list[str] | None = None
    history_fields: list[str] | None = None
    per_subject: list[dict[str, str]] = []
    predictions: list[dict[str, str]] = []
    histories: list[dict[str, str]] = []
    for run_dir in args.run_dir:
        fields, rows = read_csv(metric_path(run_dir, "per_subject_metrics.csv"))
        per_subject_fields = per_subject_fields or fields
        per_subject.extend(rows)
        fields, rows = read_csv(metric_path(run_dir, "all_predictions.csv"))
        prediction_fields = prediction_fields or fields
        predictions.extend(rows)
        fields, rows = read_csv(metric_path(run_dir, "all_history.csv"))
        history_fields = history_fields or fields
        histories.extend(rows)

    per_subject.sort(key=lambda row: row["test_subject"])
    predictions.sort(key=lambda row: (row["subject"], int(row["sample_index"])))
    histories.sort(key=lambda row: (row["test_subject"], int(row["epoch"])))
    validate_inputs(per_subject, predictions)

    subject_errors = [float(row["mean_deg"]) for row in per_subject]
    baseline_errors = [float(row["constant_baseline_mean_deg"]) for row in per_subject]
    improvements = [
        baseline - model
        for baseline, model in zip(baseline_errors, subject_errors, strict=True)
    ]
    pooled_errors = [float(row["angular_error_deg"]) for row in predictions]
    subject_ci = bootstrap_mean_ci(
        subject_errors, replicates=args.bootstrap_replicates, seed=args.seed
    )
    improvement_ci = bootstrap_mean_ci(
        improvements, replicates=args.bootstrap_replicates, seed=args.seed + 1
    )
    hard_subjects = build_hard_subject_analysis(predictions)
    magnitude_slices = build_magnitude_slices(predictions)

    best_row = min(per_subject, key=lambda row: float(row["mean_deg"]))
    worst_row = max(per_subject, key=lambda row: float(row["mean_deg"]))
    summary = {
        "status": "complete",
        "baseline_status": "reference_baseline_not_final_model",
        "protocol": "15-fold identity-disjoint LOPO; 13 train / 1 validation / 1 test",
        "subjects": EXPECTED_SUBJECTS,
        "subject_count": len(EXPECTED_SUBJECTS),
        "samples_per_subject": 3000,
        "total_test_samples": len(predictions),
        "subject_mean_metrics": {
            **distribution_metrics(subject_errors, sample_std=True),
            "mean_ci95_low_deg": subject_ci[0],
            "mean_ci95_high_deg": subject_ci[1],
        },
        "pooled_sample_metrics": distribution_metrics(pooled_errors, sample_std=False),
        "constant_baseline_subject_mean_deg": statistics.fmean(baseline_errors),
        "paired_absolute_improvement_subject_mean_deg": statistics.fmean(improvements),
        "paired_absolute_improvement_ci95_low_deg": improvement_ci[0],
        "paired_absolute_improvement_ci95_high_deg": improvement_ci[1],
        "relative_improvement_vs_constant": statistics.fmean(improvements)
        / statistics.fmean(baseline_errors),
        "subjects_better_than_constant": sum(
            improvement > 0 for improvement in improvements
        ),
        "best_subject": best_row["test_subject"],
        "worst_subject": worst_row["test_subject"],
        "bootstrap": {
            "unit": "subject",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "interval": "percentile_95",
        },
        "input_run_dirs": [str(path) for path in args.run_dir],
        "limitations": [
            "The pooled sample distribution is descriptive; frames within a "
            "subject are not independent.",
            "The prediction tables contain gaze angles and errors, but no "
            "illumination, glasses, image-quality, or head-pose fields.",
            "Direct comparison with papers requires a matched dataset, "
            "preprocessing, split, and metric protocol.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert per_subject_fields is not None
    assert prediction_fields is not None
    assert history_fields is not None
    write_csv(
        args.output_dir / "per_subject_metrics.csv", per_subject_fields, per_subject
    )
    write_csv(args.output_dir / "all_predictions.csv", prediction_fields, predictions)
    write_csv(args.output_dir / "all_history.csv", history_fields, histories)
    write_csv(
        args.output_dir / "hard_subject_analysis.csv",
        list(hard_subjects[0]),
        hard_subjects,
    )
    write_csv(
        args.output_dir / "error_by_target_magnitude.csv",
        list(magnitude_slices[0]),
        magnitude_slices,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    save_subject_plot(per_subject, args.output_dir / "per_subject_error.svg")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
