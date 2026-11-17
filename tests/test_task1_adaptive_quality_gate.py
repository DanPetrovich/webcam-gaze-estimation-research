from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_task1_adaptive_quality_gate",
    SCRIPTS / "evaluate_task1_adaptive_quality_gate.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1AdaptiveQualityGateTest(unittest.TestCase):
    def test_choose_gate_excludes_held_subject(self) -> None:
        rows = []
        for selector in MODULE.GATE_FRACTIONS:
            rows.extend(
                [
                    {
                        "subject": "train",
                        "selector": selector,
                        "budget": 1,
                        "calibrator": "map_offset",
                        "calibrated_mean_deg": 1.0
                        if selector == "quality90_then_target_diversity"
                        else 2.0,
                    },
                    {
                        "subject": "held",
                        "selector": selector,
                        "budget": 1,
                        "calibrator": "map_offset",
                        "calibrated_mean_deg": 0.0
                        if selector == "quality_then_target_diversity"
                        else 3.0,
                    },
                ]
            )
        self.assertEqual(
            MODULE.choose_gate(
                rows,
                held_subject="held",
                budget=1,
                calibrator="map_offset",
            ),
            "quality90_then_target_diversity",
        )

    def test_choose_gate_prefers_lower_capture_cost_on_exact_tie(self) -> None:
        rows = [
            {
                "subject": "train",
                "selector": selector,
                "budget": 1,
                "calibrator": "mean_offset",
                "calibrated_mean_deg": 1.0,
            }
            for selector in MODULE.GATE_FRACTIONS
        ]
        self.assertEqual(
            MODULE.choose_gate(
                rows,
                held_subject="held",
                budget=1,
                calibrator="mean_offset",
            ),
            "quality95_then_target_diversity",
        )


if __name__ == "__main__":
    unittest.main()
