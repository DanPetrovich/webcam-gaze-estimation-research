#!/usr/bin/env python3
"""Compare MAE->MPII and GazeGene->MPII on identical LOPO subjects."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

EXPECTED_SUBJECTS = tuple(f"p{index:02d}" for index in range(15))


def read_subject_means(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"test_subject", "mean_deg"} <= set(rows[0]):
        raise ValueError(f"Missing test_subject/mean_deg columns: {path}")
    values = {row["test_subject"]: float(row["mean_deg"]) for row in rows}
    if len(values) != len(rows):
        raise ValueError(f"Duplicate test_subject rows: {path}")
    if tuple(sorted(values)) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected all 15 LOPO subjects, got {sorted(values)}")
    return values


def paired_bootstrap_ci(
    differences: list[float], replicates: int, seed: int
) -> tuple[float, float]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = random.Random(seed)
    sample_size = len(differences)
    means = sorted(
        statistics.fmean(generator.choice(differences) for _ in range(sample_size))
        for _ in range(replicates)
    )
    low_index = round(0.025 * (replicates - 1))
    high_index = round(0.975 * (replicates - 1))
    return means[low_index], means[high_index]


def compare(
    baseline: dict[str, float],
    gazegene: dict[str, float],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, float | str]], dict[str, float | int | str]]:
    rows: list[dict[str, float | str]] = []
    for subject in EXPECTED_SUBJECTS:
        difference = baseline[subject] - gazegene[subject]
        rows.append(
            {
                "test_subject": subject,
                "mae_init_mean_deg": baseline[subject],
                "gazegene_init_mean_deg": gazegene[subject],
                "improvement_deg": difference,
            }
        )
    differences = [float(row["improvement_deg"]) for row in rows]
    low, high = paired_bootstrap_ci(differences, replicates, seed)
    summary: dict[str, float | int | str] = {
        "comparison": "MAE-to-MPII minus GazeGene-to-MPII",
        "subjects": len(rows),
        "mae_init_subject_macro_mean_deg": statistics.fmean(baseline.values()),
        "gazegene_init_subject_macro_mean_deg": statistics.fmean(gazegene.values()),
        "mean_improvement_deg": statistics.fmean(differences),
        "median_improvement_deg": statistics.median(differences),
        "improvement_std_deg": statistics.stdev(differences),
        "improvement_ci95_low_deg": low,
        "improvement_ci95_high_deg": high,
        "subjects_improved": sum(value > 0 for value in differences),
        "seed": seed,
        "bootstrap_replicates": replicates,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mae", required=True, type=Path)
    parser.add_argument("--gazegene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()

    rows, summary = compare(
        read_subject_means(args.mae),
        read_subject_means(args.gazegene),
        args.bootstrap_replicates,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_subject_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
