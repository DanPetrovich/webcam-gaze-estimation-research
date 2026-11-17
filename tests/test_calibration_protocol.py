from __future__ import annotations

import unittest

from gaze_estimation.calibration_protocol import (
    ScreenGeometry,
    build_plan,
    grid_points,
    plan_manifest,
)


class CalibrationProtocolTest(unittest.TestCase):
    def test_screen_geometry_uses_top_left_origin(self) -> None:
        screen = ScreenGeometry(1920, 1080, 344.0, 194.0)
        self.assertEqual(screen.normalized_to_pixels(0.0, 0.0), (0.0, 0.0))
        self.assertEqual(screen.normalized_to_pixels(1.0, 1.0), (1919.0, 1079.0))
        self.assertEqual(screen.normalized_to_mm(0.5, 0.5), (172.0, 97.0))

    def test_grid_has_fixed_margin_and_center(self) -> None:
        points = grid_points(3, 3, margin_fraction=0.1)
        self.assertEqual(len(points), 9)
        self.assertIn(("r2c2", 0.5, 0.5), points)
        self.assertEqual(points[0], ("r1c1", 0.1, 0.1))
        self.assertEqual(points[-1], ("r3c3", 0.9, 0.9))

    def test_plan_is_deterministic_and_counts_duration(self) -> None:
        options = {
            "rows": 3,
            "columns": 3,
            "margin_fraction": 0.1,
            "repetitions": 2,
            "settle_ms": 500,
            "capture_ms": 1500,
            "seed": 7,
            "shuffle_within_repetition": True,
        }
        first = build_plan(**options)
        second = build_plan(**options)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        manifest = plan_manifest(first, protocol_name="test", seed=7)
        self.assertEqual(manifest["planned_duration_ms"], 36_000)
        self.assertEqual(
            manifest["privacy"],
            "stimulus_plan_only_no_camera_frames_or_biometrics",
        )

    def test_invalid_normalized_coordinates_are_rejected(self) -> None:
        screen = ScreenGeometry(1920, 1080, 344.0, 194.0)
        with self.assertRaises(ValueError):
            screen.normalized_to_pixels(-0.1, 0.5)


if __name__ == "__main__":
    unittest.main()
