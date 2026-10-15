from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaze_estimation.experiment_tracking import (
    collect_run_rows,
    finish_run,
    stable_json_hash,
    start_run,
    write_run_index,
)


def test_stable_json_hash_ignores_mapping_order() -> None:
    assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})


def test_run_identity_includes_source_content(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("version = 1\n", encoding="utf-8")
    started_at = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    first = start_run(
        experiment="Task 0",
        config={"seed": 7},
        output_root=tmp_path / "outputs",
        repository_root=tmp_path,
        source_paths=[source],
        started_at=started_at,
    )
    source.write_text("version = 2\n", encoding="utf-8")
    second = start_run(
        experiment="Task 0",
        config={"seed": 7},
        output_root=tmp_path / "outputs",
        repository_root=tmp_path,
        source_paths=[source],
        started_at=started_at,
    )
    assert first.name != second.name
    assert (first / "checkpoints").is_dir()
    assert (first / "resolved_config.json").is_file()


def test_finalize_and_index_completed_run(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("pass\n", encoding="utf-8")
    outputs = tmp_path / "outputs"
    run_dir = start_run(
        experiment="task0",
        config={"seed": 3},
        output_root=outputs,
        repository_root=tmp_path,
        source_paths=[source],
    )
    finish_run(
        run_dir,
        status="completed",
        summary={"test": {"mean_deg": 4.8}, "folds": 15},
    )

    rows = collect_run_rows([outputs])
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["metric.test.mean_deg"] == 4.8

    destination = tmp_path / "results" / "index.csv"
    write_run_index(rows, destination)
    assert "metric.test.mean_deg" in destination.read_text(encoding="utf-8")


def test_run_cannot_be_finalized_twice(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("pass\n", encoding="utf-8")
    run_dir = start_run(
        experiment="task0",
        config={},
        output_root=tmp_path / "outputs",
        repository_root=tmp_path,
        source_paths=[source],
    )
    finish_run(run_dir, status="failed", note="expected test failure")
    with pytest.raises(RuntimeError, match="already finalized"):
        finish_run(run_dir, status="completed")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
