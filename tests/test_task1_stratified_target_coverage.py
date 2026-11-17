from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_task1_stratified_target_coverage",
    SCRIPTS / "run_task1_stratified_target_coverage.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1StratifiedTargetCoverageTest(unittest.TestCase):
    def test_region_quotas_are_balanced_and_exact(self) -> None:
        self.assertEqual(MODULE.region_quotas(3), {
            "central_le_10": 1,
            "mid_10_20": 1,
            "peripheral_gt_20": 1,
        })
        self.assertEqual(MODULE.region_quotas(5), {
            "central_le_10": 1,
            "mid_10_20": 2,
            "peripheral_gt_20": 2,
        })
        self.assertEqual(sum(MODULE.region_quotas(16).values()), 16)

    def test_region_quotas_reject_tiny_budget(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.region_quotas(2)


if __name__ == "__main__":
    unittest.main()
