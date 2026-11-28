from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task1_gazegene_calibration",
    ROOT / "scripts" / "run_task1_gazegene_calibration.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def observation(subject: int, camera: int, frame: int, residual: float):
    target_pitch = residual
    value = MODULE.Observation(
        subject=subject,
        camera=camera,
        frame=frame,
        target_pitch=target_pitch,
        target_yaw=0.0,
        pred_pitch=0.0,
        pred_yaw=0.0,
        uncalibrated_error_deg=math.degrees(abs(residual)),
    )
    return value


def test_load_predictions_requires_complete_unique_sequence(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    fields = (
        "subject",
        "camera",
        "frame",
        "target_pitch_rad",
        "target_yaw_rad",
        "pred_pitch_rad",
        "pred_yaw_rad",
        "angular_error_deg",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in range(4):
            item = observation(47, 0, frame, math.radians(frame + 1))
            writer.writerow(
                {
                    "subject": item.subject,
                    "camera": item.camera,
                    "frame": item.frame,
                    "target_pitch_rad": item.target_pitch,
                    "target_yaw_rad": item.target_yaw,
                    "pred_pitch_rad": item.pred_pitch,
                    "pred_yaw_rad": item.pred_yaw,
                    "angular_error_deg": item.uncalibrated_error_deg,
                }
            )

    sequences, audit = MODULE.load_predictions(
        path,
        expected_subjects=[47],
        cameras=1,
        frames=4,
        angular_error_tolerance_deg=0.01,
        enforce_angular_error_tolerance=True,
    )

    assert tuple(item.frame for item in sequences[(47, 0)]) == (0, 1, 2, 3)
    assert audit.pop("angular_error_max_abs_difference_deg") < 1e-10
    assert audit == {
        "subjects": 1,
        "cameras_per_subject": 1,
        "frames_per_camera": 4,
        "samples": 4,
        "duplicate_subject_camera_frames": 0,
        "finite": True,
        "angular_error_tolerance_deg": 0.01,
        "angular_error_within_tolerance": True,
        "angular_error_mismatch_counts": {
            "over_0_01_deg": 0,
            "over_0_05_deg": 0,
            "over_0_10_deg": 0,
        },
    }


def test_map_prior_is_fit_only_from_supplied_validation_splits() -> None:
    validation = {
        (41, 0): MODULE.SequenceSplit(
            candidate=tuple(observation(41, 0, frame, 0.10) for frame in range(10)),
            guard=(),
            evaluation=(),
        ),
        (42, 0): MODULE.SequenceSplit(
            candidate=tuple(observation(42, 0, frame, 0.20) for frame in range(10)),
            guard=(),
            evaluation=(),
        ),
    }

    prior = MODULE.estimate_map_prior(validation)
    offset = MODULE.map_offset([observation(47, 0, 0, 1.0)], prior)

    assert math.isclose(prior.pitch_mean, 0.15)
    assert prior.pitch_mean < offset[0] < 1.0
    assert offset[1] == 0.0


def test_huber_reduces_single_outlier_in_residuals() -> None:
    selected = [
        observation(47, 0, 0, 0.10),
        observation(47, 0, 1, 0.11),
        observation(47, 0, 2, 1.00),
    ]

    ordinary = MODULE.mean_offset(selected)[0]
    robust = MODULE.huber_offset(selected, delta=0.05, iterations=100, tolerance=1e-12)[
        0
    ]

    assert abs(robust - 0.105) < abs(ordinary - 0.105)


def test_random_sampling_is_deterministic_per_sequence_and_draw() -> None:
    split = MODULE.SequenceSplit(
        candidate=tuple(observation(47, 0, frame, frame / 100) for frame in range(20)),
        guard=(),
        evaluation=(),
    )

    first = MODULE.sample(split, seed=7, key=(47, 0), budget=5, draw=3)
    second = MODULE.sample(split, seed=7, key=(47, 0), budget=5, draw=3)

    assert [item.frame for item in first] == [item.frame for item in second]
    assert len({item.frame for item in first}) == 5


def test_failure_reason_exposes_only_bounded_codes() -> None:
    assert MODULE.bounded_reason(ValueError("angular_error_mismatch")) == (
        "angular_error_mismatch"
    )
    assert MODULE.bounded_reason(OSError("secret/path")) == "unexpected_OSError"
