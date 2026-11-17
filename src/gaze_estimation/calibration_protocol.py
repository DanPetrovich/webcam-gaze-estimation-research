"""Deterministic screen-stimulus plans for webcam gaze calibration."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScreenGeometry:
    """Display geometry with a top-left pixel and millimetre origin."""

    width_px: int
    height_px: int
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if self.width_px <= 1 or self.height_px <= 1:
            raise ValueError("Screen pixel dimensions must exceed one")
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise ValueError("Screen physical dimensions must be positive")

    def normalized_to_pixels(self, x: float, y: float) -> tuple[float, float]:
        _validate_normalized(x, y)
        return x * (self.width_px - 1), y * (self.height_px - 1)

    def normalized_to_mm(self, x: float, y: float) -> tuple[float, float]:
        _validate_normalized(x, y)
        return x * self.width_mm, y * self.height_mm


@dataclass(frozen=True)
class Stimulus:
    index: int
    repetition: int
    point_id: str
    normalized_x: float
    normalized_y: float
    settle_ms: int
    capture_ms: int


def _validate_normalized(x: float, y: float) -> None:
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("Normalized screen coordinates must lie in [0, 1]")


def grid_points(
    rows: int,
    columns: int,
    *,
    margin_fraction: float,
) -> list[tuple[str, float, float]]:
    if rows < 1 or columns < 1:
        raise ValueError("Grid dimensions must be positive")
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError("Margin must lie in [0, 0.5)")

    def positions(count: int) -> list[float]:
        if count == 1:
            return [0.5]
        span = 1.0 - 2.0 * margin_fraction
        return [margin_fraction + span * index / (count - 1) for index in range(count)]

    return [
        (f"r{row_index + 1}c{column_index + 1}", x, y)
        for row_index, y in enumerate(positions(rows))
        for column_index, x in enumerate(positions(columns))
    ]


def build_plan(
    *,
    rows: int,
    columns: int,
    margin_fraction: float,
    repetitions: int,
    settle_ms: int,
    capture_ms: int,
    seed: int,
    shuffle_within_repetition: bool,
) -> list[Stimulus]:
    if repetitions < 1:
        raise ValueError("At least one repetition is required")
    if settle_ms < 0 or capture_ms <= 0:
        raise ValueError("Timing must use non-negative settle and positive capture")
    points = grid_points(rows, columns, margin_fraction=margin_fraction)
    generator = random.Random(seed)
    plan: list[Stimulus] = []
    for repetition in range(repetitions):
        ordered = points[:]
        if shuffle_within_repetition:
            generator.shuffle(ordered)
        for point_id, x, y in ordered:
            plan.append(
                Stimulus(
                    index=len(plan),
                    repetition=repetition,
                    point_id=point_id,
                    normalized_x=x,
                    normalized_y=y,
                    settle_ms=settle_ms,
                    capture_ms=capture_ms,
                )
            )
    return plan


def plan_manifest(
    plan: list[Stimulus],
    *,
    protocol_name: str,
    seed: int,
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Cannot serialize an empty calibration plan")
    total_duration_ms = sum(item.settle_ms + item.capture_ms for item in plan)
    return {
        "schema_version": 1,
        "protocol_name": protocol_name,
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "normalized_range": [0.0, 1.0],
        },
        "seed": seed,
        "stimulus_count": len(plan),
        "planned_duration_ms": total_duration_ms,
        "stimuli": [asdict(item) for item in plan],
        "privacy": "stimulus_plan_only_no_camera_frames_or_biometrics",
    }
