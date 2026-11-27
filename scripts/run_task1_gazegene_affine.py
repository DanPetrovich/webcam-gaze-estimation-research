#!/usr/bin/env python3
"""Compare identity-regularized affine calibration on frozen GazeGene predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts import run_task1_gazegene_calibration as gg
from scripts import run_task1_ridge_affine as ridge

FloatArray = NDArray[np.float64]
METHODS = ("uncalibrated", "mean", "map", "huber", "ridge")


def matrix(rows: Sequence[gg.Observation]) -> tuple[FloatArray, FloatArray]:
    pred = np.array([(item.pred_pitch, item.pred_yaw) for item in rows])
    target = np.array([(item.target_pitch, item.target_yaw) for item in rows])
    return pred, target


def angular_error(pred: FloatArray, target: FloatArray) -> FloatArray:
    pitch, yaw = pred.T
    gt_pitch, gt_yaw = target.T
    dot = np.sin(pitch) * np.sin(gt_pitch) + np.cos(pitch) * np.cos(gt_pitch) * np.cos(
        yaw - gt_yaw
    )
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))  # type: ignore[no-any-return]


def affine(
    chosen: list[gg.Observation], pred: FloatArray, scale: float, penalty: float
) -> FloatArray:
    coef = ridge.fit_ridge_affine(chosen, ridge_lambda=penalty, angle_scale_rad=scale)
    return np.asarray(
        pred @ np.asarray(coef[:2]) + scale * np.asarray(coef[2]), dtype=np.float64
    )


def tune(
    splits: dict[tuple[int, int], gg.SequenceSplit],
    budget: int,
    penalties: list[float],
    scale: float,
    seed: int,
    draws: int,
) -> tuple[float, list[float]]:
    totals = np.zeros(len(penalties))
    for key, split in sorted(splits.items()):
        pred, target = matrix(split.guard)
        for draw in range(draws):
            chosen = gg.sample(split, seed=seed, key=key, budget=budget, draw=draw)
            for i, penalty in enumerate(penalties):
                errors = angular_error(affine(chosen, pred, scale, penalty), target)
                totals[i] += errors.mean()
    scores = (totals / (len(splits) * draws)).tolist()
    winner = min(range(len(scores)), key=lambda i: (scores[i], -penalties[i]))
    return penalties[winner], scores


def evaluate(
    splits: dict[tuple[int, int], gg.SequenceSplit],
    prior: gg.MapPrior,
    budget: int,
    penalty: float,
    scale: float,
    delta: float,
    edges: list[float],
    protocol: dict[str, Any],
    draws: int,
) -> tuple[FloatArray, list[int]]:
    subjects = sorted({key[0] for key in splits})
    by_subject = np.zeros((len(subjects), len(METHODS), 4))
    counts = np.zeros(4, dtype=int)
    for key, split in sorted(splits.items()):
        pred, target = matrix(split.evaluation)
        eccentricity = np.degrees(
            np.arccos(np.clip(np.cos(target[:, 0]) * np.cos(target[:, 1]), -1, 1))
        )
        bins = np.searchsorted(edges, eccentricity, side="right")
        masks = [np.ones(len(bins), dtype=bool), *(bins == i for i in range(3))]
        sizes = np.array([mask.sum() for mask in masks])
        if not sizes.all():
            raise ValueError("empty_evaluation_slice")
        counts += sizes
        values = np.zeros((len(METHODS), 4))
        baseline = angular_error(pred, target)
        values[0] = [baseline[mask].mean() for mask in masks]
        for draw in range(draws):
            chosen = gg.sample(
                split, seed=protocol["seed"], key=key, budget=budget, draw=draw
            )
            corrections = (
                pred + gg.mean_offset(chosen),
                pred + gg.map_offset(chosen, prior),
                pred
                + gg.huber_offset(
                    chosen,
                    delta=delta,
                    iterations=protocol["maximum_iterations"],
                    tolerance=protocol["tolerance"],
                ),
                affine(chosen, pred, scale, penalty),
            )
            for index, adjusted in enumerate(corrections, start=1):
                errors = angular_error(adjusted, target)
                values[index] += [errors[mask].mean() for mask in masks]
        values[1:] /= draws
        by_subject[subjects.index(key[0])] += values / protocol["cameras"]
    return by_subject, counts.tolist()


def comparison(values: FloatArray, seed: int, replicates: int) -> dict[str, Any]:
    indices = np.random.default_rng(seed).integers(
        len(values), size=(replicates, len(values))
    )
    ci = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {
        "gain": float(values.mean()),
        "ci95": ci.tolist(),
        "worse_identities": int(np.sum(values < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "test", "config", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = tomllib.loads(args.protocol.read_text(encoding="utf-8"))["experiment"]
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["experiment"]
    splits = []
    for path, subject_key in (
        (args.validation, "validation_subjects"),
        (args.test, "test_subjects"),
    ):
        sequences, _ = gg.load_predictions(
            path,
            expected_subjects=protocol[subject_key],
            cameras=protocol["cameras"],
            frames=protocol["frames_per_camera"],
            angular_error_tolerance_deg=protocol["angular_error_tolerance_deg"],
            enforce_angular_error_tolerance=True,
        )
        splits.append(
            gg.split_sequences(
                sequences,
                candidate_fraction=protocol["candidate_fraction"],
                guard_fraction=protocol["guard_fraction"],
            )
        )
    validation, test = splits
    if {s for s, _ in validation} & {s for s, _ in test}:
        raise ValueError("validation_test_identity_overlap")
    prior = gg.estimate_map_prior(validation)
    scale = math.radians(config["angle_scale_deg"])
    tune_draws = protocol["smoke_draws" if args.smoke else "huber_tuning_draws"]
    draws = protocol["smoke_draws" if args.smoke else "random_draws"]
    replicates = 1000 if args.smoke else protocol["bootstrap_replicates"]
    rows = []
    for budget, delta in zip(
        protocol["budgets"], config["huber_delta_degrees"], strict=True
    ):
        penalty, scores = tune(
            validation,
            budget,
            config["ridge_lambdas"],
            scale,
            protocol["seed"],
            tune_draws,
        )
        values, counts = evaluate(
            test,
            prior,
            budget,
            penalty,
            scale,
            math.radians(delta),
            config["slice_edges_deg"],
            protocol,
            draws,
        )
        rows.append(
            {
                "budget": budget,
                "lambda": penalty,
                "validation_scores": scores,
                "slices": [
                    {
                        "samples": count,
                        "errors": dict(
                            zip(
                                METHODS,
                                values[:, :, i].mean(axis=0).tolist(),
                                strict=True,
                            )
                        ),
                        "ridge_vs": {
                            name: comparison(
                                values[:, j, i] - values[:, -1, i],
                                protocol["seed"],
                                replicates,
                            )
                            for j, name in enumerate(METHODS[:-1])
                        },
                    }
                    for i, count in enumerate(counts)
                ],
            }
        )
    input_hashes = {}
    for name in ("validation", "test"):
        with getattr(args, name).open("rb") as stream:
            input_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "status": "complete",
        "mode": "affine-smoke" if args.smoke else "affine",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input_sha256": input_hashes,
        "config": config,
        "draws": draws,
        "tuning_draws": tune_draws,
        "bootstrap_replicates": replicates,
        "aggregate_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(json.dumps({"status": "failed", "reason": gg.bounded_reason(error)}))
        raise SystemExit(1) from None
