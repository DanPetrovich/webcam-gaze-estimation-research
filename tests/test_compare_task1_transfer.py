from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_task1_transfer.py"
SPEC = importlib.util.spec_from_file_location("compare_task1_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_comparison_sign_and_subject_count() -> None:
    baseline = {subject: 5.0 for subject in MODULE.EXPECTED_SUBJECTS}
    gazegene = {subject: 4.5 for subject in MODULE.EXPECTED_SUBJECTS}
    rows, summary = MODULE.compare(baseline, gazegene, replicates=1_000, seed=42)
    assert len(rows) == 15
    assert summary["mean_improvement_deg"] == 0.5
    assert summary["subjects_improved"] == 15
    assert summary["improvement_ci95_low_deg"] == 0.5
    assert summary["improvement_ci95_high_deg"] == 0.5


def test_paired_bootstrap_is_deterministic() -> None:
    values = [-1.0, 0.0, 2.0]
    assert MODULE.paired_bootstrap_ci(values, 1_000, 7) == MODULE.paired_bootstrap_ci(
        values, 1_000, 7
    )
