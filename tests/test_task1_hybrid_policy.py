from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_task1_hybrid_policy",
    SCRIPTS / "evaluate_task1_hybrid_policy.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1HybridPolicyTest(unittest.TestCase):
    def test_policy_uses_soft_map_once_and_robust_afterward(self) -> None:
        soft = [
            {
                "subject": "p00",
                "selector": MODULE.ONE_SHOT_SELECTOR,
                "budget": "1",
                "calibrator": "map_offset",
                "calibrated_mean_deg": "2.0",
                "gain_deg": "1.0",
            }
        ]
        robust = [
            {
                "subject": "p00",
                "selector": "random",
                "budget": "3",
                "calibrated_mean_deg": "1.5",
                "mean_gain_deg": "1.5",
            }
        ]
        reference = [
            {
                "subject": "p00",
                "selector": "random",
                "budget": "1",
                "calibrator": "map_offset",
                "calibrated_mean_deg": "2.2",
            },
            {
                "subject": "p00",
                "selector": "random",
                "budget": "3",
                "calibrator": "mean_offset",
                "calibrated_mean_deg": "1.6",
            },
        ]
        selected, _, comparisons = MODULE.assemble_policy(
            soft,
            robust,
            reference,
            budgets=[1, 3],
            replicates=100,
            seed=1,
        )
        self.assertEqual(
            [row["method"] for row in selected],
            ["soft_quality25_map_offset", "random_huber_offset"],
        )
        self.assertAlmostEqual(comparisons[0]["mean_difference_deg"], 0.2)
        self.assertAlmostEqual(comparisons[1]["mean_difference_deg"], 0.1)


if __name__ == "__main__":
    unittest.main()
