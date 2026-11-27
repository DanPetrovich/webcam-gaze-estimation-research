import math

import pytest

np = pytest.importorskip("numpy")

from scripts import run_task1_gazegene_affine as affine  # noqa: E402
from scripts import run_task1_gazegene_calibration as gg  # noqa: E402


def test_affine_recovers_known_scale_and_cross_axis_correction() -> None:
    pred = np.array([[0.2, 0.3], [0.2, -0.3], [-0.2, 0.3], [-0.2, -0.3]])
    target = pred @ np.array([[1.2, 0.1], [-0.1, 0.9]]) + [0.01, -0.04]
    rows = [
        gg.Observation(1, 0, i, tp, ty, pp, py, 0.0)
        for i, ((pp, py), (tp, ty)) in enumerate(zip(pred, target, strict=True))
    ]
    corrected = affine.affine(rows, pred, math.radians(30), penalty=1e-8)
    np.testing.assert_allclose(corrected, target, atol=1e-7)


def test_single_shot_regularization_is_finite() -> None:
    row = gg.Observation(1, 0, 0, 0.3, 0.2, 0.0, 0.0, 0.0)
    corrected = affine.affine([row], np.zeros((1, 2)), 0.5, penalty=1.0)
    np.testing.assert_allclose(corrected, [[0.15, 0.10]])
    np.testing.assert_allclose(
        affine.angular_error(np.array([[0.0, 0.0]]), np.array([[0.0, math.pi / 6]])),
        [30.0],
    )
