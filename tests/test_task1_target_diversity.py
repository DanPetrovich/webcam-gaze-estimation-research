from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT_PATH = SCRIPTS_DIR / "run_task1_target_diversity.py"
SPEC = importlib.util.spec_from_file_location("run_task1_target_diversity", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1TargetDiversityTest(unittest.TestCase):
    def observation(self, index: int, pitch: float, yaw: float) -> object:
        return MODULE.baseline.Observation(
            group_id=str(index),
            min_sample_index=index,
            multiplicity=1,
            target_pitch=pitch,
            target_yaw=yaw,
            pred_pitch=pitch,
            pred_yaw=yaw,
            uncalibrated_error_deg=0.0,
        )

    def test_selector_starts_near_center_then_adds_extremes(self) -> None:
        candidates = tuple(
            self.observation(index, 0.0, math.radians(yaw))
            for index, yaw in enumerate((-20, -10, 0, 10, 20))
        )
        selected = MODULE.target_diversity_sample(candidates, 3)
        self.assertEqual(selected[0].group_id, "2")
        self.assertEqual({item.group_id for item in selected}, {"0", "2", "4"})

    def test_selection_is_nested_and_unique(self) -> None:
        candidates = tuple(
            self.observation(index, math.radians(index - 5), math.radians(index))
            for index in range(11)
        )
        selected_three = MODULE.target_diversity_sample(candidates, 3)
        selected_five = MODULE.target_diversity_sample(candidates, 5)
        self.assertEqual(selected_three, selected_five[:3])
        self.assertEqual(len({item.group_id for item in selected_five}), 5)

    def test_covering_radius_does_not_increase_with_budget(self) -> None:
        candidates = tuple(
            self.observation(index, 0.0, math.radians(index * 2))
            for index in range(10)
        )
        radii = [
            MODULE.coverage_metrics(
                candidates,
                MODULE.target_diversity_sample(candidates, budget),
            )[0]
            for budget in (1, 3, 5)
        ]
        self.assertGreaterEqual(radii[0], radii[1])
        self.assertGreaterEqual(radii[1], radii[2])

    def test_invalid_budget_is_rejected(self) -> None:
        candidates = (self.observation(0, 0.0, 0.0),)
        with self.assertRaises(ValueError):
            MODULE.target_diversity_sample(candidates, 2)


if __name__ == "__main__":
    unittest.main()
