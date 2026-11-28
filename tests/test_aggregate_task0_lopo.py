from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "aggregate_task0_lopo.py"
SPEC = importlib.util.spec_from_file_location("aggregate_task0_lopo", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateTask0LopoTest(unittest.TestCase):
    def test_quantile_uses_linear_interpolation(self) -> None:
        self.assertEqual(MODULE.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(MODULE.quantile([1.0, 2.0, 3.0, 4.0], 0.9), 3.7)

    def test_bootstrap_is_deterministic(self) -> None:
        first = MODULE.bootstrap_mean_ci([1.0, 2.0, 3.0], replicates=1_000, seed=42)
        second = MODULE.bootstrap_mean_ci([1.0, 2.0, 3.0], replicates=1_000, seed=42)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.0)
        self.assertGreaterEqual(first[1], 2.0)

    def test_distribution_metrics_distinguish_sample_std(self) -> None:
        population = MODULE.distribution_metrics([1.0, 2.0, 3.0], sample_std=False)
        sample = MODULE.distribution_metrics([1.0, 2.0, 3.0], sample_std=True)
        self.assertAlmostEqual(population["mean_deg"], 2.0)
        self.assertGreater(sample["std_deg"], population["std_deg"])


if __name__ == "__main__":
    unittest.main()
