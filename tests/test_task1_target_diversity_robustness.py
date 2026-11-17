from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_task1_target_diversity_robustness",
    SCRIPTS / "evaluate_task1_target_diversity_robustness.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1TargetDiversityRobustnessTest(unittest.TestCase):
    def test_hybrid_reference_ignores_one_shot(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "subject,budget,slice,calibrated_mean_deg\n"
                "p00,1,central_le_10,3.0\n"
                "p00,3,central_le_10,2.0\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.load_hybrid_reference(path),
                {("p00", 3, "central_le_10"): 2.0},
            )


if __name__ == "__main__":
    unittest.main()
