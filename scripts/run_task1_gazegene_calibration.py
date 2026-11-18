#!/usr/bin/env python3
"""Audit and evaluate few-shot calibration on frozen GazeGene predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Observation:
    subject: int
    camera: int
    frame: int
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
class SequenceSplit:
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
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--test-predictions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "smoke", "full"), default="full")
    return parser.parse_args()


def bounded_reason(error: Exception) -> str:
    reason = str(error)
    if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", reason):
        return reason
    return f"unexpected_{type(error).__name__}"


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def pitchyaw_to_vector(pitch: float, yaw: float) -> tuple[float, float, float]:
    return (
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    )


def angular_error_deg(observation: Observation, pitch: float, yaw: float) -> float:
    prediction = pitchyaw_to_vector(
        observation.pred_pitch + pitch, observation.pred_yaw + yaw
    )
    target = pitchyaw_to_vector(observation.target_pitch, observation.target_yaw)
    cosine = sum(a * b for a, b in zip(prediction, target, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty_quantile")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def bootstrap_ci(
    values: list[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(replicates)
    ]
    return quantile(means, 0.025), quantile(means, 0.975)


def load_predictions(
    path: Path,
    *,
    expected_subjects: list[int],
    cameras: int,
    frames: int,
) -> tuple[dict[tuple[int, int], tuple[Observation, ...]], dict[str, Any]]:
    required = {
        "subject",
        "camera",
        "frame",
        "target_pitch_rad",
        "target_yaw_rad",
        "pred_pitch_rad",
        "pred_yaw_rad",
        "angular_error_deg",
    }
    grouped: dict[tuple[int, int], list[Observation]] = defaultdict(list)
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("prediction_columns_invalid")
        for row in reader:
            values = [
                float(row[key])
                for key in (
                    "target_pitch_rad",
                    "target_yaw_rad",
                    "pred_pitch_rad",
                    "pred_yaw_rad",
                    "angular_error_deg",
                )
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite_prediction")
            subject, camera, frame = (
                int(row["subject"]),
                int(row["camera"]),
                int(row["frame"]),
            )
            sample_key = (subject, camera, frame)
            if sample_key in seen:
                raise ValueError("duplicate_subject_camera_frame")
            seen.add(sample_key)
            observation = Observation(subject, camera, frame, *values)
            recomputed = angular_error_deg(observation, 0.0, 0.0)
            if not math.isclose(
                recomputed, observation.uncalibrated_error_deg, abs_tol=1e-2
            ):
                raise ValueError("angular_error_mismatch")
            grouped[(subject, camera)].append(observation)

    expected_keys = {
        (subject, camera) for subject in expected_subjects for camera in range(cameras)
    }
    if set(grouped) != expected_keys:
        raise ValueError("subject_camera_coverage_invalid")
    result: dict[tuple[int, int], tuple[Observation, ...]] = {}
    for sequence_key, observations in grouped.items():
        ordered = tuple(sorted(observations, key=lambda item: item.frame))
        if len(ordered) != frames or [item.frame for item in ordered] != list(
            range(frames)
        ):
            raise ValueError("frame_coverage_invalid")
        result[sequence_key] = ordered
    return result, {
        "subjects": len(expected_subjects),
        "cameras_per_subject": cameras,
        "frames_per_camera": frames,
        "samples": len(seen),
        "duplicate_subject_camera_frames": 0,
        "finite": True,
        "angular_error_verified": True,
    }


def split_sequences(
    sequences: dict[tuple[int, int], tuple[Observation, ...]],
    *,
    candidate_fraction: float,
    guard_fraction: float,
) -> dict[tuple[int, int], SequenceSplit]:
    if (
        candidate_fraction <= 0
        or guard_fraction <= 0
        or candidate_fraction + guard_fraction >= 1
    ):
        raise ValueError("split_fractions_invalid")
    result = {}
    for key, observations in sequences.items():
        candidate_end = math.floor(len(observations) * candidate_fraction)
        guard_end = candidate_end + math.floor(len(observations) * guard_fraction)
        result[key] = SequenceSplit(
            candidate=observations[:candidate_end],
            guard=observations[candidate_end:guard_end],
            evaluation=observations[guard_end:],
        )
    return result


def variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def estimate_map_prior(splits: dict[tuple[int, int], SequenceSplit]) -> MapPrior:
    pitch_means, yaw_means, pitch_variances, yaw_variances, counts = [], [], [], [], []
    for split in splits.values():
        pitch = [item.pitch_residual for item in split.candidate]
        yaw = [item.yaw_residual for item in split.candidate]
        pitch_means.append(statistics.fmean(pitch))
        yaw_means.append(statistics.fmean(yaw))
        pitch_variances.append(variance(pitch))
        yaw_variances.append(variance(yaw))
        counts.append(len(pitch))
    pitch_noise, yaw_noise = (
        statistics.fmean(pitch_variances),
        statistics.fmean(yaw_variances),
    )
    epsilon = 1e-12
    return MapPrior(
        pitch_mean=statistics.fmean(pitch_means),
        yaw_mean=statistics.fmean(yaw_means),
        pitch_between_variance=max(
            variance(pitch_means)
            - statistics.fmean(
                v / n for v, n in zip(pitch_variances, counts, strict=True)
            ),
            epsilon,
        ),
        yaw_between_variance=max(
            variance(yaw_means)
            - statistics.fmean(
                v / n for v, n in zip(yaw_variances, counts, strict=True)
            ),
            epsilon,
        ),
        pitch_noise_variance=max(pitch_noise, epsilon),
        yaw_noise_variance=max(yaw_noise, epsilon),
    )


def mean_offset(selected: list[Observation]) -> tuple[float, float]:
    return (
        statistics.fmean(item.pitch_residual for item in selected),
        statistics.fmean(item.yaw_residual for item in selected),
    )


def map_offset(selected: list[Observation], prior: MapPrior) -> tuple[float, float]:
    pitch, yaw = mean_offset(selected)
    count = len(selected)
    pitch_alpha = (
        count
        * prior.pitch_between_variance
        / (count * prior.pitch_between_variance + prior.pitch_noise_variance)
    )
    yaw_alpha = (
        count
        * prior.yaw_between_variance
        / (count * prior.yaw_between_variance + prior.yaw_noise_variance)
    )
    return (
        pitch_alpha * pitch + (1 - pitch_alpha) * prior.pitch_mean,
        yaw_alpha * yaw + (1 - yaw_alpha) * prior.yaw_mean,
    )


def huber_location(
    values: list[float], *, delta: float, iterations: int, tolerance: float
) -> float:
    location = statistics.median(values)
    for _ in range(iterations):
        weights = [
            min(1.0, delta / max(abs(value - location), 1e-15)) for value in values
        ]
        updated = sum(w * v for w, v in zip(weights, values, strict=True)) / sum(
            weights
        )
        if abs(updated - location) <= tolerance:
            return updated
        location = updated
    return location


def huber_offset(
    selected: list[Observation], *, delta: float, iterations: int, tolerance: float
) -> tuple[float, float]:
    return (
        huber_location(
            [item.pitch_residual for item in selected],
            delta=delta,
            iterations=iterations,
            tolerance=tolerance,
        ),
        huber_location(
            [item.yaw_residual for item in selected],
            delta=delta,
            iterations=iterations,
            tolerance=tolerance,
        ),
    )


def sample(
    split: SequenceSplit, *, seed: int, key: tuple[int, int], budget: int, draw: int
) -> list[Observation]:
    generator = random.Random(stable_seed(seed, *key, budget, draw))
    return generator.sample(list(split.candidate), budget)


def mean_error(
    observations: tuple[Observation, ...], offset: tuple[float, float]
) -> float:
    return statistics.fmean(angular_error_deg(item, *offset) for item in observations)


def tune_huber(
    splits: dict[tuple[int, int], SequenceSplit],
    *,
    budgets: list[int],
    deltas: list[float],
    draws: int,
    seed: int,
    iterations: int,
    tolerance: float,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    selected: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        scores = {}
        for delta in deltas:
            errors = []
            for key, split in sorted(splits.items()):
                for draw in range(draws):
                    chosen = sample(split, seed=seed, key=key, budget=budget, draw=draw)
                    offset = huber_offset(
                        chosen, delta=delta, iterations=iterations, tolerance=tolerance
                    )
                    errors.append(mean_error(split.guard, offset))
            scores[delta] = statistics.fmean(errors)
        winner = min(deltas, key=lambda value: (scores[value], -value))
        selected[budget] = winner
        rows.extend(
            {
                "budget": budget,
                "delta_deg": math.degrees(delta),
                "validation_guard_mean_deg": scores[delta],
                "selected": delta == winner,
            }
            for delta in deltas
        )
    return selected, rows


def evaluate(
    splits: dict[tuple[int, int], SequenceSplit],
    *,
    prior: MapPrior,
    huber_deltas: dict[int, float],
    budgets: list[int],
    draws: int,
    bootstrap_replicates: int,
    seed: int,
    iterations: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted({key[0] for key in splits})
    uncalibrated = {
        subject: statistics.fmean(
            mean_error(splits[(subject, camera)].evaluation, (0.0, 0.0))
            for camera in sorted(key[1] for key in splits if key[0] == subject)
        )
        for subject in subjects
    }
    draw_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for draw in range(draws):
            by_subject: dict[tuple[int, str], list[float]] = defaultdict(list)
            for key, split in sorted(splits.items()):
                chosen = sample(split, seed=seed, key=key, budget=budget, draw=draw)
                offsets = {
                    "mean_offset": mean_offset(chosen),
                    "map_offset": map_offset(chosen, prior),
                    "huber_offset": huber_offset(
                        chosen,
                        delta=huber_deltas[budget],
                        iterations=iterations,
                        tolerance=tolerance,
                    ),
                }
                for calibrator, offset in offsets.items():
                    by_subject[(key[0], calibrator)].append(
                        mean_error(split.evaluation, offset)
                    )
            for (subject, calibrator), camera_errors in sorted(by_subject.items()):
                calibrated = statistics.fmean(camera_errors)
                draw_rows.append(
                    {
                        "subject": subject,
                        "budget": budget,
                        "draw": draw,
                        "calibrator": calibrator,
                        "uncalibrated_mean_deg": uncalibrated[subject],
                        "calibrated_mean_deg": calibrated,
                        "gain_deg": uncalibrated[subject] - calibrated,
                    }
                )

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in draw_rows:
        grouped[
            (int(row["subject"]), int(row["budget"]), str(row["calibrator"]))
        ].append(row)
    subject_rows = []
    for (subject, budget, calibrator), rows in sorted(grouped.items()):
        gains = [float(row["gain_deg"]) for row in rows]
        subject_rows.append(
            {
                "subject": subject,
                "budget": budget,
                "calibrator": calibrator,
                "draws": len(rows),
                "uncalibrated_mean_deg": rows[0]["uncalibrated_mean_deg"],
                "calibrated_mean_deg": statistics.fmean(
                    float(row["calibrated_mean_deg"]) for row in rows
                ),
                "calibrated_draw_std_deg": statistics.stdev(
                    float(row["calibrated_mean_deg"]) for row in rows
                )
                if len(rows) > 1
                else 0.0,
                "mean_gain_deg": statistics.fmean(gains),
                "harmed_draw_fraction": statistics.fmean(gain < 0 for gain in gains),
            }
        )

    aggregate_rows = []
    for budget in budgets:
        for calibrator in ("mean_offset", "map_offset", "huber_offset"):
            matching = [
                row
                for row in subject_rows
                if row["budget"] == budget and row["calibrator"] == calibrator
            ]
            gains = [float(row["mean_gain_deg"]) for row in matching]
            low, high = bootstrap_ci(
                gains,
                replicates=bootstrap_replicates,
                seed=stable_seed(seed, budget, calibrator, "bootstrap"),
            )
            aggregate_rows.append(
                {
                    "budget": budget,
                    "calibrator": calibrator,
                    "subjects": len(matching),
                    "subject_mean_uncalibrated_deg": statistics.fmean(
                        float(row["uncalibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_calibrated_deg": statistics.fmean(
                        float(row["calibrated_mean_deg"]) for row in matching
                    ),
                    "subject_std_calibrated_deg": statistics.stdev(
                        float(row["calibrated_mean_deg"]) for row in matching
                    ),
                    "subject_mean_gain_deg": statistics.fmean(gains),
                    "gain_ci95_low_deg": low,
                    "gain_ci95_high_deg": high,
                    "harmed_subject_fraction": statistics.fmean(
                        gain < 0 for gain in gains
                    ),
                    "mean_within_subject_draw_std_deg": statistics.fmean(
                        float(row["calibrated_draw_std_deg"]) for row in matching
                    ),
                    "huber_delta_deg": math.degrees(huber_deltas[budget])
                    if calibrator == "huber_offset"
                    else None,
                }
            )
    return subject_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    validation, validation_audit = load_predictions(
        args.validation_predictions,
        expected_subjects=[int(value) for value in config["validation_subjects"]],
        cameras=int(config["cameras"]),
        frames=int(config["frames_per_camera"]),
    )
    test, test_audit = load_predictions(
        args.test_predictions,
        expected_subjects=[int(value) for value in config["test_subjects"]],
        cameras=int(config["cameras"]),
        frames=int(config["frames_per_camera"]),
    )
    if set(validation) & set(test):
        raise ValueError("validation_test_identity_overlap")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "complete",
        "mode": args.mode,
        "identity_disjoint": True,
        "validation": validation_audit,
        "test": test_audit,
        "checkpoint_sha256": str(config["expected_checkpoint_sha256"]),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.mode == "audit":
        print(json.dumps(audit, separators=(",", ":")))
        return

    candidate_fraction = float(config["candidate_fraction"])
    guard_fraction = float(config["guard_fraction"])
    validation_splits = split_sequences(
        validation, candidate_fraction=candidate_fraction, guard_fraction=guard_fraction
    )
    test_splits = split_sequences(
        test, candidate_fraction=candidate_fraction, guard_fraction=guard_fraction
    )
    budgets = [int(value) for value in config["budgets"]]
    prior = estimate_map_prior(validation_splits)
    iterations, tolerance = (
        int(config["maximum_iterations"]),
        float(config["tolerance"]),
    )
    huber_deltas, tuning_rows = tune_huber(
        validation_splits,
        budgets=budgets,
        deltas=[math.radians(float(value)) for value in config["huber_delta_degrees"]],
        draws=int(
            config["huber_tuning_draws"]
            if args.mode == "full"
            else config["smoke_draws"]
        ),
        seed=int(config["seed"]),
        iterations=iterations,
        tolerance=tolerance,
    )
    subject_rows, aggregate_rows = evaluate(
        test_splits,
        prior=prior,
        huber_deltas=huber_deltas,
        budgets=budgets,
        draws=int(
            config["random_draws"] if args.mode == "full" else config["smoke_draws"]
        ),
        bootstrap_replicates=int(
            config["bootstrap_replicates"] if args.mode == "full" else 1000
        ),
        seed=int(config["seed"]),
        iterations=iterations,
        tolerance=tolerance,
    )
    write_csv(output / "huber_tuning.csv", tuning_rows)
    write_csv(output / "per_subject_metrics.csv", subject_rows)
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    summary = {
        **audit,
        "candidate_frames_per_camera": len(next(iter(test_splits.values())).candidate),
        "guard_frames_per_camera": len(next(iter(test_splits.values())).guard),
        "evaluation_frames_per_camera": len(
            next(iter(test_splits.values())).evaluation
        ),
        "budgets": budgets,
        "draws": int(
            config["random_draws"] if args.mode == "full" else config["smoke_draws"]
        ),
        "huber_delta_degrees": {
            str(key): math.degrees(value) for key, value in huber_deltas.items()
        },
        "aggregate_metrics": aggregate_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(
            json.dumps(
                {"status": "failed", "reason": bounded_reason(error)},
                separators=(",", ":"),
            )
        )
        raise SystemExit(1) from None
