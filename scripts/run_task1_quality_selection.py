#!/usr/bin/env python3
"""Select quality/pose-aware calibration samples on private normalized data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task1_quality_audit as quality_audit  # noqa: E402

EXPECTED_SUBJECTS = tuple(f"p{index:02d}" for index in range(15))


@dataclass(frozen=True)
class Candidate:
    subject: str
    group_id: str
    sample_index: int
    quality: dict[str, float]
    head_pitch: float
    head_yaw: float
    target_pitch: float
    target_yaw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--quality-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, list[tuple[str, int]]]:
    grouped = {subject: [] for subject in EXPECTED_SUBJECTS}
    seen_groups: set[tuple[str, str]] = set()
    seen_indices: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"subject", "group_id", "min_sample_index"}:
        raise ValueError("Unexpected candidate manifest schema")
    for row in rows:
        subject = row["subject"]
        group_id = row["group_id"]
        index = int(row["min_sample_index"])
        if subject not in grouped or not 0 <= index < 3000:
            raise ValueError("Invalid candidate manifest entry")
        if (subject, group_id) in seen_groups or (subject, index) in seen_indices:
            raise ValueError("Duplicate candidate manifest entry")
        seen_groups.add((subject, group_id))
        seen_indices.add((subject, index))
        grouped[subject].append((group_id, index))
    if any(len(values) < 16 for values in grouped.values()):
        raise ValueError("Candidate manifest does not cover every subject")
    return grouped


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def quality_reference(candidates: list[Candidate]) -> dict[str, float]:
    def q(key: str, probability: float) -> float:
        return quality_audit.quantile(
            [item.quality[key] for item in candidates], probability
        )

    return {
        "brightness_low": q("brightness", 0.05),
        "brightness_median": q("brightness", 0.50),
        "brightness_high": q("brightness", 0.95),
        "contrast_low": q("contrast", 0.05),
        "contrast_high": q("contrast", 0.95),
        "sharpness_low": q("log_laplacian_variance", 0.05),
        "sharpness_high": q("log_laplacian_variance", 0.95),
        "dark_high": q("dark_fraction", 0.95),
        "bright_high": q("bright_fraction", 0.95),
    }


def quality_score(item: Candidate, reference: dict[str, float]) -> float:
    epsilon = 1e-12
    brightness = item.quality["brightness"]
    median = reference["brightness_median"]
    brightness_scale = max(
        median - reference["brightness_low"],
        reference["brightness_high"] - median,
        epsilon,
    )
    components = [
        clamp(1.0 - abs(brightness - median) / brightness_scale),
        clamp(
            (item.quality["contrast"] - reference["contrast_low"])
            / max(reference["contrast_high"] - reference["contrast_low"], epsilon)
        ),
        clamp(
            (item.quality["log_laplacian_variance"] - reference["sharpness_low"])
            / max(reference["sharpness_high"] - reference["sharpness_low"], epsilon)
        ),
        clamp(
            1.0
            - item.quality["dark_fraction"] / max(reference["dark_high"], epsilon)
        ),
        clamp(
            1.0
            - item.quality["bright_fraction"]
            / max(reference["bright_high"], epsilon)
        ),
    ]
    return statistics.fmean(components)


def rank_by_score(
    candidates: list[Candidate], scores: dict[str, float], *, reverse: bool
) -> list[Candidate]:
    multiplier = -1.0 if reverse else 1.0
    return sorted(
        candidates,
        key=lambda item: (multiplier * scores[item.group_id], item.sample_index),
    )


def farthest_first(
    candidates: list[Candidate],
    count: int,
    *,
    coordinates: Callable[[Candidate], tuple[float, float]],
    center: tuple[float, float],
) -> list[Candidate]:
    if count <= 0 or count > len(candidates):
        raise ValueError("Invalid farthest-first selection size")

    def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        return math.hypot(left[0] - right[0], left[1] - right[1])

    first = min(
        candidates,
        key=lambda item: (distance(coordinates(item), center), item.sample_index),
    )
    selected = [first]
    selected_ids = {first.group_id}
    while len(selected) < count:
        item = max(
            (value for value in candidates if value.group_id not in selected_ids),
            key=lambda value: (
                min(
                    distance(coordinates(value), coordinates(chosen))
                    for chosen in selected
                ),
                -value.sample_index,
            ),
        )
        selected.append(item)
        selected_ids.add(item.group_id)
    return selected


def target_vector(item: Candidate) -> tuple[float, float, float]:
    return (
        math.cos(item.target_pitch) * math.sin(item.target_yaw),
        math.sin(item.target_pitch),
        math.cos(item.target_pitch) * math.cos(item.target_yaw),
    )


def angular_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    cosine = sum(a * b for a, b in zip(left, right, strict=True))
    return math.acos(max(-1.0, min(1.0, cosine)))


def soft_quality_target_farthest_first(
    candidates: list[Candidate],
    count: int,
    scores: dict[str, float],
    quality_weight: float,
) -> list[Candidate]:
    """Balance target coverage and quality without rejecting candidates."""
    if count <= 0 or count > len(candidates):
        raise ValueError("Invalid soft-selection size")
    if not 0.0 <= quality_weight <= 1.0:
        raise ValueError("Quality weight must be between zero and one")
    if set(scores) != {item.group_id for item in candidates}:
        raise ValueError("Quality scores do not cover the candidates")
    pitch = statistics.fmean(item.target_pitch for item in candidates)
    yaw = statistics.fmean(item.target_yaw for item in candidates)
    center = (
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    )
    vectors = {item.group_id: target_vector(item) for item in candidates}
    center_distances = {
        item.group_id: angular_distance(vectors[item.group_id], center)
        for item in candidates
    }
    maximum_center_distance = max(center_distances.values(), default=0.0)

    def centrality(item: Candidate) -> float:
        if maximum_center_distance <= 0.0:
            return 1.0
        return 1.0 - center_distances[item.group_id] / maximum_center_distance

    first = max(
        candidates,
        key=lambda item: (
            (1.0 - quality_weight) * centrality(item)
            + quality_weight * scores[item.group_id],
            -item.sample_index,
        ),
    )
    selected = [first]
    selected_ids = {first.group_id}
    while len(selected) < count:
        remaining = [
            value for value in candidates if value.group_id not in selected_ids
        ]
        minimum_distances = {
            value.group_id: min(
                angular_distance(vectors[value.group_id], vectors[chosen.group_id])
                for chosen in selected
            )
            for value in remaining
        }
        maximum_distance = max(minimum_distances.values(), default=0.0)
        item = max(
            remaining,
            key=lambda value: (
                (1.0 - quality_weight)
                * (
                    minimum_distances[value.group_id] / maximum_distance
                    if maximum_distance > 0.0
                    else 0.0
                )
                + quality_weight * scores[value.group_id],
                -value.sample_index,
            ),
        )
        selected.append(item)
        selected_ids.add(item.group_id)
    return selected


def target_farthest_first(candidates: list[Candidate], count: int) -> list[Candidate]:
    return soft_quality_target_farthest_first(
        candidates,
        count,
        {item.group_id: 0.0 for item in candidates},
        0.0,
    )


def extract_candidates(
    manifest: dict[str, list[tuple[str, int]]],
    dataset_root: Path,
    quality_config: dict[str, Any],
) -> dict[str, list[Candidate]]:
    import h5py
    import numpy as np

    result: dict[str, list[Candidate]] = {}
    for subject, entries in manifest.items():
        with h5py.File(dataset_root / f"{subject}.h5", "r", swmr=True) as handle:
            required = {"face_patch", "face_head_pose", "face_gaze"}
            if not required.issubset(handle.keys()):
                raise ValueError("Normalized subject file has missing keys")
            items = []
            for group_id, index in entries:
                pose = np.asarray(handle["face_head_pose"][index], dtype=np.float64)
                target = np.asarray(handle["face_gaze"][index], dtype=np.float64)
                if pose.shape != (2,) or target.shape != (2,):
                    raise ValueError("Unexpected private label shape")
                items.append(
                    Candidate(
                        subject,
                        group_id,
                        index,
                        quality_audit.quality_metrics(
                            handle["face_patch"][index], quality_config
                        ),
                        float(pose[0]),
                        float(pose[1]),
                        float(target[0]),
                        float(target[1]),
                    )
                )
            result[subject] = items
    return result


def select(
    candidates: dict[str, list[Candidate]], config: dict[str, Any]
) -> dict[str, Any]:
    maximum_budget = max(int(value) for value in config["experiment"]["budgets"])
    gate_fractions = [
        float(value) for value in config["selection"]["quality_gate_fractions"]
    ]
    soft_weights = [
        float(value) for value in config["selection"].get("soft_quality_weights", [])
    ]
    selections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    accepted_fractions: dict[float, list[float]] = {
        value: [] for value in gate_fractions
    }
    for subject, held_out in sorted(candidates.items()):
        training = [
            item
            for other, values in candidates.items()
            if other != subject
            for item in values
        ]
        reference = quality_reference(training)
        scores = {item.group_id: quality_score(item, reference) for item in held_out}
        high = rank_by_score(held_out, scores, reverse=True)
        low = rank_by_score(held_out, scores, reverse=False)
        train_pitch = [item.head_pitch for item in training]
        train_yaw = [item.head_yaw for item in training]
        pitch_mean = statistics.fmean(train_pitch)
        yaw_mean = statistics.fmean(train_yaw)
        pitch_std, yaw_std = statistics.stdev(train_pitch), statistics.stdev(train_yaw)
        def pose_coordinates(
            item: Candidate,
            pitch_center: float = pitch_mean,
            pitch_scale: float = pitch_std,
            yaw_center: float = yaw_mean,
            yaw_scale: float = yaw_std,
        ) -> tuple[float, float]:
            return (
                (item.head_pitch - pitch_center) / pitch_scale,
                (item.head_yaw - yaw_center) / yaw_scale,
            )

        pose = farthest_first(
            held_out,
            maximum_budget,
            coordinates=pose_coordinates,
            center=(0.0, 0.0),
        )
        methods = {
            "high_quality": high[:maximum_budget],
            "low_quality": low[:maximum_budget],
            "pose_diversity": pose,
        }
        gate_names = {
            0.50: "quality_then_target_diversity",
            0.75: "quality75_then_target_diversity",
            0.90: "quality90_then_target_diversity",
            0.95: "quality95_then_target_diversity",
        }
        for fraction in gate_fractions:
            accepted_count = max(
                maximum_budget, math.ceil(len(held_out) * fraction)
            )
            accepted_fractions[fraction].append(accepted_count / len(held_out))
            methods[gate_names[fraction]] = target_farthest_first(
                high[:accepted_count], maximum_budget
            )
        for weight in soft_weights:
            suffix = int(round(weight * 100))
            methods[f"soft_quality{suffix}_target_diversity"] = (
                soft_quality_target_farthest_first(
                    held_out,
                    maximum_budget,
                    scores,
                    weight,
                )
            )
        selections[subject] = {
            method: [
                {"group_id": item.group_id, "min_sample_index": item.sample_index}
                for item in values
            ]
            for method, values in methods.items()
        }
    return {
        "status": "complete",
        "experiment": config["experiment"]["name"],
        "subjects": len(selections),
        "maximum_budget": maximum_budget,
        "selectors": list(config["selection"]["selectors"]),
        "quality_gate_fraction_mean": {
            str(fraction): statistics.fmean(values)
            for fraction, values in accepted_fractions.items()
        },
        "soft_quality_weights": soft_weights,
        "selections": selections,
        "privacy": "selected_group_ids_only_no_images_labels_or_features_returned",
    }


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    quality_config = tomllib.loads(args.quality_config.read_text(encoding="utf-8"))
    if sha256_file(args.candidate_manifest) != config["experiment"][
        "candidate_manifest_sha256"
    ]:
        raise ValueError("Candidate manifest hash mismatch")
    candidates = extract_candidates(
        load_manifest(args.candidate_manifest),
        args.dataset_root,
        quality_config["quality"],
    )
    summary = select(candidates, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
