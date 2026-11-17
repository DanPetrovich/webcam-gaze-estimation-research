#!/usr/bin/env python3
"""Audit simple image-validity and head-pose signals on normalized MPIIFaceGaze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SUBJECTS = tuple(f"p{index:02d}" for index in range(15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile without values")
    if not 0 <= probability <= 1:
        raise ValueError("Quantile probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p05": quantile(values, 0.05),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
    }


def binary_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Return tie-aware pairwise AUC without third-party dependencies."""
    if not positive_scores or not negative_scores:
        raise ValueError("AUC requires both positive and negative scores")
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def deterministic_indices(
    population: int,
    count: int,
    *,
    seed: int,
) -> list[int]:
    if population <= 0 or count <= 0 or count > population:
        raise ValueError("Invalid deterministic sample size")
    return sorted(random.Random(seed).sample(range(population), count))


def quality_metrics(image: Any, config: dict[str, Any]) -> dict[str, float]:
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.shape != (224, 224, 3):
        raise ValueError("Unexpected normalized face image shape")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite normalized face image")
    if array.dtype != np.uint8:
        maximum = float(array.max())
        array = np.clip(array * (255.0 if maximum <= 1.0 else 1.0), 0, 255).astype(
            np.uint8
        )
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "log_laplacian_variance": math.log1p(laplacian_variance),
        "dark_fraction": float(
            (gray <= int(config["dark_pixel_threshold"])).mean()
        ),
        "bright_fraction": float(
            (gray >= int(config["bright_pixel_threshold"])).mean()
        ),
    }


def corrupt(image: Any, name: str, config: dict[str, Any]) -> Any:
    import cv2
    import numpy as np

    array = np.asarray(image).copy()
    if array.dtype != np.uint8:
        maximum = float(array.max())
        array = np.clip(array * (255.0 if maximum <= 1.0 else 1.0), 0, 255).astype(
            np.uint8
        )
    if name == "blur":
        kernel = int(config["blur_kernel"])
        return cv2.GaussianBlur(array, (kernel, kernel), float(config["blur_sigma"]))
    if name == "dark":
        return np.clip(
            array.astype(np.float32) * float(config["dark_scale"]), 0, 255
        ).astype(np.uint8)
    if name == "bright":
        return np.clip(
            array.astype(np.float32) * float(config["bright_scale"])
            + float(config["bright_offset"]),
            0,
            255,
        ).astype(np.uint8)
    if name == "occlusion":
        height, width = array.shape[:2]
        y0, y1 = (
            int(height * float(value)) for value in config["occlusion_y_fraction"]
        )
        x0, x1 = (
            int(width * float(value)) for value in config["occlusion_x_fraction"]
        )
        array[y0:y1, x0:x1] = 0
        return array
    raise ValueError(f"Unknown controlled corruption: {name}")


def audit(config: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    import cv2
    import h5py
    import numpy as np

    experiment = config["experiment"]
    quality_config = config["quality"]
    protocol = config["protocol"]
    expected_samples = int(experiment["expected_samples_per_subject"])
    natural_count = int(experiment["natural_samples_per_subject"])
    corruption_count = int(experiment["corruption_samples_per_subject"])
    seed = int(experiment["seed"])
    required_keys = set(protocol["required_hdf5_keys"])

    natural_features: dict[str, list[float]] = defaultdict(list)
    head_pitch: list[float] = []
    head_yaw: list[float] = []
    corruption_scores: dict[str, dict[str, list[float]]] = {
        name: {"original": [], "corrupted": []}
        for name in protocol["controlled_corruptions"]
    }
    per_subject_counts: dict[str, int] = {}

    for subject in EXPECTED_SUBJECTS:
        path = dataset_root / f"{subject}.h5"
        if not path.is_file():
            raise ValueError("Missing normalized subject file")
        with h5py.File(path, "r", swmr=True) as handle:
            if not required_keys.issubset(handle.keys()):
                raise ValueError("Normalized subject file has missing keys")
            images = handle["face_patch"]
            head_pose = handle["face_head_pose"]
            if images.shape != (expected_samples, 224, 224, 3):
                raise ValueError("Unexpected face_patch dataset shape")
            if head_pose.shape != (expected_samples, 2):
                raise ValueError("Unexpected face_head_pose dataset shape")
            pose = np.asarray(head_pose[:], dtype=np.float64)
            if not np.isfinite(pose).all():
                raise ValueError("Non-finite head pose")
            head_pitch.extend(np.degrees(pose[:, 0]).tolist())
            head_yaw.extend(np.degrees(pose[:, 1]).tolist())
            per_subject_counts[subject] = expected_samples

            natural_indices = deterministic_indices(
                expected_samples,
                natural_count,
                seed=stable_seed(seed, subject, "natural"),
            )
            corruption_positions = deterministic_indices(
                natural_count,
                corruption_count,
                seed=stable_seed(seed, subject, "corruption"),
            )
            corruption_indices = {
                natural_indices[position] for position in corruption_positions
            }
            for index in natural_indices:
                image = images[index]
                original = quality_metrics(image, quality_config)
                for key, value in original.items():
                    natural_features[key].append(value)
                if index not in corruption_indices:
                    continue
                for corruption in protocol["controlled_corruptions"]:
                    changed = quality_metrics(
                        corrupt(image, corruption, quality_config), quality_config
                    )
                    if corruption == "blur":
                        metric, sign = "log_laplacian_variance", 1.0
                    elif corruption == "dark":
                        metric, sign = "brightness", 1.0
                    elif corruption == "bright":
                        metric, sign = "bright_fraction", -1.0
                    else:
                        metric, sign = "dark_fraction", -1.0
                    corruption_scores[corruption]["original"].append(
                        sign * original[metric]
                    )
                    corruption_scores[corruption]["corrupted"].append(
                        sign * changed[metric]
                    )

    corruption_auc = {
        name: binary_auc(scores["original"], scores["corrupted"])
        for name, scores in corruption_scores.items()
    }
    return {
        "status": "complete",
        "experiment": experiment["name"],
        "subjects": len(per_subject_counts),
        "samples": sum(per_subject_counts.values()),
        "natural_quality_samples": len(natural_features["brightness"]),
        "controlled_samples_per_corruption": sum(
            len(scores["original"]) for scores in corruption_scores.values()
        )
        // len(corruption_scores),
        "quality_distribution": {
            key: summarize(values) for key, values in sorted(natural_features.items())
        },
        "head_pose_degrees": {
            "pitch": summarize(head_pitch),
            "yaw": summarize(head_yaw),
            "absolute_pitch": summarize([abs(value) for value in head_pitch]),
            "absolute_yaw": summarize([abs(value) for value in head_yaw]),
        },
        "controlled_corruption_auc": corruption_auc,
        "corruption_gate": {
            "all_auc_at_least_0_90": all(
                value >= 0.90 for value in corruption_auc.values()
            ),
            "minimum_auc": min(corruption_auc.values()),
        },
        "dependencies": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
        },
        "privacy": "aggregate_only_no_images_or_per_sample_features_persisted",
    }


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    summary = audit(config, args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
