from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_task1_ridge_robustness",
    SCRIPTS / "evaluate_task1_ridge_robustness.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1RidgeRobustnessTest(unittest.TestCase):
    def test_identity_affine_slice_error_matches_uncalibrated(self) -> None:
        class Observation:
            pred_pitch = math.radians(2.0)
            pred_yaw = math.radians(3.0)
            target_pitch = math.radians(4.0)
            target_yaw = math.radians(5.0)

        identity = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        actual = MODULE.affine_slice_error(
            [Observation()],
            identity,
            angle_scale_rad=math.radians(30.0),
        )
        expected = MODULE.baseline.angular_error_deg(
            Observation.pred_pitch,
            Observation.pred_yaw,
            Observation.target_pitch,
            Observation.target_yaw,
        )
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
