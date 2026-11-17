from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_task1_quality_selection", SCRIPTS / "run_task1_quality_selection.py"
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1QualitySelectionTest(unittest.TestCase):
    def candidate(self, index: int, x: float, y: float) -> object:
        return MODULE.Candidate(
            "p00", str(index), index, {}, x, y, x, y
        )

    def test_rank_by_score_obeys_direction(self) -> None:
        candidates = [self.candidate(2, 0.0, 0.0), self.candidate(1, 0.0, 0.0)]
        scores = {"1": 0.2, "2": 0.9}
        self.assertEqual(
            [
                item.group_id
                for item in MODULE.rank_by_score(candidates, scores, reverse=True)
            ],
            ["2", "1"],
        )
        self.assertEqual(
            [
                item.group_id
                for item in MODULE.rank_by_score(candidates, scores, reverse=False)
            ],
            ["1", "2"],
        )

    def test_farthest_first_selects_center_and_extremes(self) -> None:
        candidates = [self.candidate(index, float(index), 0.0) for index in range(5)]
        selected = MODULE.farthest_first(
            candidates,
            3,
            coordinates=lambda item: (item.head_pitch, item.head_yaw),
            center=(2.0, 0.0),
        )
        self.assertEqual(selected[0].group_id, "2")
        self.assertEqual({item.group_id for item in selected}, {"0", "2", "4"})


if __name__ == "__main__":
    unittest.main()
