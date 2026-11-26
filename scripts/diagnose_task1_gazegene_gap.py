#!/usr/bin/env python3
"""Measure frozen prediction error structure and offline calibration windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as stats
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from scripts import run_task1_gazegene_calibration as gg

Offset = tuple[float, float]
METHODS = ("random_center", "random_median", "stable_median")


def weighted_sum(weights: list[float], values: list[float]) -> float:
    return math.fsum(w * x for w, x in zip(weights, values, strict=True))


def eccentricity(pitch: float, yaw: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(pitch) * math.cos(yaw)))))


def profile(
    rows: list[tuple[str, float, float]], edges: list[int]
) -> tuple[list[float], list[float]]:
    # Equal weight per subject, followed by equal weight per row within subject.
    counts = Counter(subject for subject, _, _ in rows)
    mass = [0.0] * (len(edges) + 1)
    total = mass.copy()
    for subject, angle, error in rows:
        index = sum(angle >= edge for edge in edges)
        weight = 1 / (len(counts) * counts[subject])
        mass[index] += weight
        total[index] += weight * error
    return mass, [
        value / weight if weight else 0.0
        for value, weight in zip(total, mass, strict=True)
    ]


def load_mpii(path: Path) -> list[tuple[str, float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            (
                row["subject"],
                eccentricity(
                    float(row["target_pitch_rad"]), float(row["target_yaw_rad"])
                ),
                float(row["angular_error_deg"]),
            )
            for row in csv.DictReader(stream)
        ]
    if Counter(row[0] for row in rows) != {f"p{i:02d}": 3000 for i in range(15)}:
        raise ValueError("mpii_coverage_invalid")
    if not all(math.isfinite(angle) and 0 <= error <= 180 for _, angle, error in rows):
        raise ValueError("mpii_nonfinite_or_invalid_error")
    return rows


def summarize(values: list[float], seed: int) -> dict[str, object]:
    return {
        "mean": stats.fmean(values),
        "ci95": gg.bootstrap_ci(values, replicates=10000, seed=seed),
    }


def offset_error(rows: tuple[gg.Observation, ...], offset: Offset) -> float:
    return gg.mean_error(rows, offset)


def windows(
    rows: tuple[gg.Observation, ...], width: int, target_limit: float
) -> list[tuple[Offset, Offset, float]]:
    result = []
    for start in range(0, len(rows) - width + 1, width):
        block = rows[start : start + width]
        center = block[width // 2]
        if any(
            math.hypot(
                row.target_pitch - center.target_pitch,
                row.target_yaw - center.target_yaw,
            )
            > target_limit
            for row in block
        ):
            continue
        pitch = stats.median(row.pred_pitch for row in block)
        yaw = stats.median(row.pred_yaw for row in block)
        jitter = stats.median(
            math.hypot(row.pred_pitch - pitch, row.pred_yaw - yaw) for row in block
        )
        result.append(
            (
                (center.pitch_residual, center.yaw_residual),
                (center.target_pitch - pitch, center.target_yaw - yaw),
                jitter,
            )
        )
    return result


def fit_huber(offsets: list[Offset]) -> Offset:
    pitch, yaw = zip(*offsets, strict=True)
    return (
        gg.huber_location(
            list(pitch), delta=math.radians(3), iterations=100, tolerance=1e-10
        ),
        gg.huber_location(
            list(yaw), delta=math.radians(3), iterations=100, tolerance=1e-10
        ),
    )


def diagnose(
    sequences: dict[tuple[int, int], tuple[gg.Observation, ...]],
    *,
    seed: int,
    draws: int,
    width: int,
    budget: int,
    target_limit: float,
) -> dict[str, object]:
    offsets: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    temporal: dict[int, list[list[float]]] = defaultdict(list)
    window_counts = []
    for (subject, camera), rows in sorted(sequences.items()):
        candidate, evaluation = rows[:400], rows[600:]
        late_fit, late_score = evaluation[::2], evaluation[1::2]
        early = gg.mean_offset(list(candidate))
        late = gg.mean_offset(list(late_fit))
        baseline = offset_error(late_score, (0.0, 0.0))
        offsets[subject].append(
            (
                baseline,
                offset_error(late_score, early),
                offset_error(late_score, late),
                math.degrees(math.dist(early, late)),
            )
        )
        pool = windows(candidate, width, target_limit)
        window_counts.append(len(pool))
        if len(pool) < budget:
            continue
        stable = sorted(pool, key=lambda window: window[2])[:budget]
        stable_error = offset_error(evaluation, fit_huber([item[1] for item in stable]))
        scores: list[list[float]] = [[], []]
        generator = random.Random(gg.stable_seed(seed, subject, camera))
        for _ in range(draws):
            chosen = generator.sample(pool, budget)
            for index, results in zip((0, 1), scores, strict=True):
                results.append(
                    offset_error(
                        evaluation,
                        fit_huber(
                            [item[0] if index == 0 else item[1] for item in chosen]
                        ),
                    )
                )
        temporal[subject].append(
            [
                offset_error(evaluation, (0.0, 0.0)),
                *(stats.fmean(results) for results in scores),
                stable_error,
            ]
        )
    split_metrics = {}
    for index, name in enumerate(
        ("baseline", "early_offset", "late_offset", "offset_shift")
    ):
        values = [
            stats.fmean(row[index] for row in group) for group in offsets.values()
        ]
        split_metrics[name] = summarize(values, seed)
    for index, name in ((1, "early_gain"), (2, "late_gain")):
        values = [
            stats.fmean(row[0] - row[index] for row in group)
            for group in offsets.values()
        ]
        split_metrics[name] = summarize(values, seed)
    temporal_result: dict[str, object] = {
        "eligible_sequences": sum(map(len, temporal.values())),
        "eligible_subjects": len(temporal),
        "window_counts_min_median_max": [
            min(window_counts),
            stats.median(window_counts),
            max(window_counts),
        ],
    }
    if temporal:
        subject_rows = [
            [stats.fmean(column) for column in zip(*group, strict=True)]
            for group in temporal.values()
        ]
        temporal_result["uncalibrated"] = stats.fmean(row[0] for row in subject_rows)
        for index, name in enumerate(METHODS, start=1):
            temporal_result[name] = {
                "error": stats.fmean(row[index] for row in subject_rows),
                "gain_vs_random_center": summarize(
                    [row[1] - row[index] for row in subject_rows], seed
                ),
                "gain_vs_random_median": summarize(
                    [row[2] - row[index] for row in subject_rows], seed
                ),
            }
    return {"offset_drift": split_metrics, "temporal": temporal_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gazegene", required=True, type=Path)
    parser.add_argument("--mpii", required=True, type=Path)
    parser.add_argument("--split-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    split = tomllib.loads(args.split_config.read_text(encoding="utf-8"))["experiment"]
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    sequences, audit = gg.load_predictions(
        args.gazegene,
        expected_subjects=split["test_subjects"],
        cameras=split["cameras"],
        frames=split["frames_per_camera"],
        angular_error_tolerance_deg=split["angular_error_tolerance_deg"],
        enforce_angular_error_tolerance=True,
    )
    edges = config["target_eccentricity_edges_deg"]
    gazegene = [
        (
            str(row.subject),
            eccentricity(row.target_pitch, row.target_yaw),
            row.uncalibrated_error_deg,
        )
        for rows in sequences.values()
        for row in rows
    ]
    mpii = load_mpii(args.mpii)
    gm, ge = profile(gazegene, edges)
    mm, me = profile(mpii, edges)
    common = [min(g, m) for g, m in zip(gm, mm, strict=True)]
    result = {
        "status": "complete",
        "mode": "diagnose",
        "protocol": config,
        "gazegene_audit": audit,
        "same_checkpoint": {
            "gazegene_mean": weighted_sum(gm, ge),
            "mpii_mean": weighted_sum(mm, me),
            "gazegene_bin_mass": gm,
            "mpii_bin_mass": mm,
            "gazegene_bin_error": ge,
            "mpii_bin_error": me,
            "common_mass": sum(common),
            "gazegene_common_mix_error": weighted_sum(common, ge) / sum(common),
            "mpii_common_mix_error": weighted_sum(common, me) / sum(common),
        },
        **diagnose(
            sequences,
            seed=config["seed"],
            draws=config["draws"],
            width=config["window_frames"],
            budget=config["budget"],
            target_limit=math.radians(config["target_radius_deg"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
        print(json.dumps({"status": "failed", "reason": gg.bounded_reason(error)}))
        raise SystemExit(1) from None
