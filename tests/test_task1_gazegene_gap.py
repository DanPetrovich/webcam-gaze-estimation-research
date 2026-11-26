import math

from scripts import diagnose_task1_gazegene_gap as gap
from scripts.run_task1_gazegene_calibration import Observation


def test_profile_is_subject_weighted_with_exact_bin_edges() -> None:
    weights, errors = gap.profile([("a", 0, 1), ("a", 10, 3), ("b", 0, 5)], [10])
    assert weights == [0.75, 0.25]
    assert math.isclose(errors[0], 11 / 3)
    assert errors[1] == 3
    assert gap.weighted_sum(weights, errors) == 3.5


def test_windows_reject_target_change_and_use_median() -> None:
    rows = tuple(
        Observation(47, 0, i, target, 0, prediction, 0, 0)
        for i, (target, prediction) in enumerate(
            [(0, 0.1), (0, 0.5), (0, 0.1), (0, 0), (0, 0), (0.3, 0)]
        )
    )
    assert gap.windows(rows, width=3, target_limit=0.02) == [
        ((-0.5, 0), (-0.1, 0), 0.0)
    ]


def test_drift_diagnostic_uses_later_held_out_rows() -> None:
    rows = tuple(
        Observation(47, 0, i, 0, 0, 0.10 if i < 400 else 0.02, 0, 0)
        for i in range(2000)
    )
    result = gap.diagnose(
        {(47, 0): rows}, seed=7, draws=1, width=9, budget=9, target_limit=0.02
    )
    assert result["offset_drift"]["late_offset"]["mean"] < 1e-5
    assert result["offset_drift"]["early_offset"]["mean"] > 4
    assert result["temporal"]["eligible_sequences"] == 1
    assert result["temporal"]["stable_median"]["gain_vs_random_median"]["mean"] == 0
