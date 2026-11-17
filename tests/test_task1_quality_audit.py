from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_task1_quality_audit.py"
SPEC = importlib.util.spec_from_file_location("run_task1_quality_audit", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1QualityAuditTest(unittest.TestCase):
    def test_binary_auc_handles_wins_losses_and_ties(self) -> None:
        self.assertEqual(MODULE.binary_auc([2.0], [1.0]), 1.0)
        self.assertEqual(MODULE.binary_auc([1.0], [2.0]), 0.0)
        self.assertEqual(MODULE.binary_auc([1.0], [1.0]), 0.5)

    def test_quantile_interpolates(self) -> None:
        self.assertAlmostEqual(MODULE.quantile([0.0, 10.0], 0.25), 2.5)

    def test_deterministic_indices_are_stable_unique_and_sorted(self) -> None:
        first = MODULE.deterministic_indices(100, 20, seed=42)
        second = MODULE.deterministic_indices(100, 20, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(set(first)))
        self.assertEqual(len(first), 20)

    def test_invalid_sample_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.deterministic_indices(10, 11, seed=1)

    def test_stable_seed_separates_subjects(self) -> None:
        self.assertNotEqual(
            MODULE.stable_seed(20261112, "p00", "natural"),
            MODULE.stable_seed(20261112, "p01", "natural"),
        )


if __name__ == "__main__":
    unittest.main()
