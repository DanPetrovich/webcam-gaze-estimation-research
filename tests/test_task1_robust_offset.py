from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_task1_robust_offset.py"
SPEC = importlib.util.spec_from_file_location("run_task1_robust_offset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1RobustOffsetTest(unittest.TestCase):
    def test_huber_location_downweights_outlier(self) -> None:
        location, weights = MODULE.huber_location(
            [1.0, 1.0, 1.0, 10.0],
            delta=1.0,
            maximum_iterations=50,
            tolerance=1e-12,
        )
        self.assertLess(location, 2.0)
        self.assertLess(weights[-1], 0.2)

    def test_large_delta_matches_mean(self) -> None:
        location, weights = MODULE.huber_location(
            [1.0, 2.0, 3.0],
            delta=100.0,
            maximum_iterations=10,
            tolerance=1e-12,
        )
        self.assertAlmostEqual(location, 2.0)
        self.assertEqual(weights, [1.0, 1.0, 1.0])

    def test_nonpositive_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.huber_location(
                [1.0], delta=0.0, maximum_iterations=10, tolerance=1e-12
            )


if __name__ == "__main__":
    unittest.main()
