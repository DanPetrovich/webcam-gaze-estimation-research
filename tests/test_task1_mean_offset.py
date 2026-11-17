from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_task1_mean_offset.py"
SPEC = importlib.util.spec_from_file_location("run_task1_mean_offset", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1MeanOffsetTest(unittest.TestCase):
    def observation(self, index: int, residual: float) -> object:
        return MODULE.Observation(
            group_id=str(index),
            min_sample_index=index,
            multiplicity=1,
            target_pitch=residual,
            target_yaw=0.0,
            pred_pitch=0.0,
            pred_yaw=0.0,
            uncalibrated_error_deg=math.degrees(abs(residual)),
        )

    def test_angular_error_matches_simple_pitch_difference(self) -> None:
        error = MODULE.angular_error_deg(0.0, 0.0, math.radians(5), 0.0)
        self.assertAlmostEqual(error, 5.0)

    def test_split_is_disjoint_and_ordered(self) -> None:
        observations = tuple(self.observation(index, 0.01) for index in range(100))
        split = MODULE.split_subject(
            "p00",
            observations,
            candidate_fraction=0.2,
            guard_fraction=0.1,
        )
        self.assertEqual(len(split.candidate), 20)
        self.assertEqual(len(split.guard), 10)
        self.assertEqual(len(split.evaluation), 70)
        self.assertLess(
            split.candidate[-1].min_sample_index,
            split.evaluation[0].min_sample_index,
        )

    def test_fixed_uniform_selector_returns_unique_points(self) -> None:
        candidates = tuple(self.observation(index, 0.01) for index in range(100))
        selected = MODULE.fixed_uniform_sample(candidates, 9)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({item.group_id for item in selected}), 9)

    def test_mean_offset_recovers_constant_bias(self) -> None:
        selected = [self.observation(index, 0.1) for index in range(3)]
        prior = MODULE.MapPrior(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        pitch, yaw, pitch_alpha, yaw_alpha = MODULE.calibration_offsets(
            selected,
            calibrator="mean_offset",
            prior=prior,
        )
        self.assertAlmostEqual(pitch, 0.1)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertEqual((pitch_alpha, yaw_alpha), (1.0, 1.0))

    def test_duplicate_rows_collapse_to_one_observation(self) -> None:
        header = (
            "subject,sample_index,target_pitch_rad,target_yaw_rad,"
            "pred_pitch_rad,pred_yaw_rad,angular_error_deg,"
            "constant_baseline_error_deg\n"
        )
        error = MODULE.angular_error_deg(0.0, 0.0, 0.1, 0.2)
        row = f"p00,0,0.1,0.2,0.0,0.0,{error},10.0\n"
        duplicate = f"p00,10,0.1,0.2,0.0,0.0,{error},10.0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(header + row + duplicate, encoding="utf-8")
            loaded = MODULE.load_unique_observations(path)
        self.assertEqual(len(loaded["p00"]), 1)
        self.assertEqual(loaded["p00"][0].multiplicity, 2)
        self.assertEqual(loaded["p00"][0].min_sample_index, 0)


if __name__ == "__main__":
    unittest.main()
