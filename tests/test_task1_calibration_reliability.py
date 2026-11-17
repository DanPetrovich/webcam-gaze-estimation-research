from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_task1_calibration_reliability",
    SCRIPTS / "run_task1_calibration_reliability.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1CalibrationReliabilityTest(unittest.TestCase):
    def test_quantile_interpolates(self) -> None:
        self.assertEqual(MODULE.quantile([0.0, 10.0], 0.25), 2.5)

    def test_average_ranks_handles_ties(self) -> None:
        self.assertEqual(MODULE.average_ranks([3.0, 1.0, 1.0]), [3.0, 1.5, 1.5])

    def test_spearman_detects_monotonic_order(self) -> None:
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [6, 5, 4]), -1.0)

    def test_residual_dispersion_is_angular_median(self) -> None:
        class Observation:
            def __init__(self, pitch: float, yaw: float) -> None:
                self.pitch_residual = pitch
                self.yaw_residual = yaw

        values = [Observation(0.0, 0.0), Observation(math.radians(2.0), 0.0)]
        self.assertAlmostEqual(MODULE.residual_dispersion_deg(values, 0.0, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
