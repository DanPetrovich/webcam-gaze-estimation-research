from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_task1_hybrid_robustness",
    SCRIPTS / "evaluate_task1_hybrid_robustness.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1HybridRobustnessTest(unittest.TestCase):
    def test_slice_boundaries_are_fixed(self) -> None:
        self.assertEqual(MODULE.slice_name(10.0, 10.0, 20.0), "central_le_10")
        self.assertEqual(MODULE.slice_name(10.1, 10.0, 20.0), "mid_10_20")
        self.assertEqual(MODULE.slice_name(20.1, 10.0, 20.0), "peripheral_gt_20")

    def test_target_eccentricity_uses_angular_distance(self) -> None:
        class Observation:
            target_pitch = 0.0
            target_yaw = math.radians(15.0)

        self.assertAlmostEqual(MODULE.target_eccentricity_deg(Observation()), 15.0)


if __name__ == "__main__":
    unittest.main()
