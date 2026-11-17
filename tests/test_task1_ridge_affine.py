from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT_PATH = SCRIPTS_DIR / "run_task1_ridge_affine.py"
SPEC = importlib.util.spec_from_file_location("run_task1_ridge_affine", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Task1RidgeAffineTest(unittest.TestCase):
    def observation(
        self,
        index: int,
        pred_pitch: float,
        pred_yaw: float,
        target_pitch: float,
        target_yaw: float,
    ) -> object:
        return MODULE.baseline.Observation(
            group_id=str(index),
            min_sample_index=index,
            multiplicity=1,
            target_pitch=target_pitch,
            target_yaw=target_yaw,
            pred_pitch=pred_pitch,
            pred_yaw=pred_yaw,
            uncalibrated_error_deg=MODULE.baseline.angular_error_deg(
                pred_pitch,
                pred_yaw,
                target_pitch,
                target_yaw,
            ),
        )

    def test_identity_data_preserves_identity_map(self) -> None:
        selected = [
            self.observation(0, -0.2, -0.1, -0.2, -0.1),
            self.observation(1, 0.2, -0.1, 0.2, -0.1),
            self.observation(2, -0.1, 0.2, -0.1, 0.2),
            self.observation(3, 0.2, 0.2, 0.2, 0.2),
        ]
        coefficients = MODULE.fit_ridge_affine(
            selected,
            ridge_lambda=0.1,
            angle_scale_rad=math.radians(30),
        )
        prediction = MODULE.predict_affine(
            selected[0],
            coefficients,
            angle_scale_rad=math.radians(30),
        )
        self.assertAlmostEqual(prediction[0], -0.2)
        self.assertAlmostEqual(prediction[1], -0.1)

    def test_affine_fit_learns_offset_and_cross_axis_term(self) -> None:
        selected = []
        for index, (pitch, yaw) in enumerate(
            [(-0.2, -0.1), (0.2, -0.1), (-0.1, 0.2), (0.2, 0.2), (0.0, 0.0)]
        ):
            selected.append(
                self.observation(
                    index,
                    pitch,
                    yaw,
                    1.1 * pitch + 0.2 * yaw + 0.03,
                    -0.1 * pitch + 0.9 * yaw - 0.02,
                )
            )
        coefficients = MODULE.fit_ridge_affine(
            selected,
            ridge_lambda=1e-8,
            angle_scale_rad=math.radians(30),
        )
        prediction = MODULE.predict_affine(
            self.observation(99, 0.1, -0.05, 0.0, 0.0),
            coefficients,
            angle_scale_rad=math.radians(30),
        )
        self.assertAlmostEqual(prediction[0], 0.13, places=6)
        self.assertAlmostEqual(prediction[1], -0.075, places=6)

    def test_large_lambda_shrinks_one_shot_fit_to_identity(self) -> None:
        item = self.observation(0, 0.1, -0.1, 0.4, 0.3)
        coefficients = MODULE.fit_ridge_affine(
            [item],
            ridge_lambda=1e9,
            angle_scale_rad=math.radians(30),
        )
        prediction = MODULE.predict_affine(
            item,
            coefficients,
            angle_scale_rad=math.radians(30),
        )
        self.assertAlmostEqual(prediction[0], item.pred_pitch, places=7)
        self.assertAlmostEqual(prediction[1], item.pred_yaw, places=7)

    def test_nonpositive_lambda_is_rejected(self) -> None:
        item = self.observation(0, 0.0, 0.0, 0.1, 0.1)
        with self.assertRaises(ValueError):
            MODULE.fit_ridge_affine(
                [item],
                ridge_lambda=0.0,
                angle_scale_rad=1.0,
            )


if __name__ == "__main__":
    unittest.main()
