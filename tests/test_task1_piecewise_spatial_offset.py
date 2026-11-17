from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_task1_piecewise_spatial_offset",
    SCRIPTS / "run_task1_piecewise_spatial_offset.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1PiecewiseSpatialOffsetTest(unittest.TestCase):
    def test_prediction_eccentricity_uses_prediction_only(self) -> None:
        class Observation:
            pred_pitch = 0.0
            pred_yaw = math.radians(12.0)
            target_pitch = math.radians(40.0)
            target_yaw = math.radians(40.0)

        self.assertAlmostEqual(MODULE.prediction_eccentricity_deg(Observation()), 12.0)

    def test_routed_error_uses_predicted_region(self) -> None:
        class Observation:
            pred_pitch = 0.0
            pred_yaw = 0.0
            target_pitch = 0.0
            target_yaw = 0.0

        offsets = {
            "central_le_10": (0.0, 0.0),
            "mid_10_20": (1.0, 1.0),
            "peripheral_gt_20": (1.0, 1.0),
        }
        self.assertAlmostEqual(
            MODULE.routed_error(
                Observation(), offsets, central_max=10.0, mid_max=20.0
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
