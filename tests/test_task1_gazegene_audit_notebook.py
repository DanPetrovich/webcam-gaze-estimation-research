from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE_PATH = ROOT / "notebooks" / "task1_gazegene_audit.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "task1_gazegene_audit.ipynb"


def test_gazegene_audit_source_has_expected_safety_and_protocol_guards() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    ast.parse(source)

    assert "vector_to_pitchyaw(-gaze_C)" in source
    assert '"train_subjects": list(CONFIG.project_train_subjects)' in source
    assert '"validation_subjects": list(CONFIG.project_validation_subjects)' in source
    assert '"test_subjects": list(CONFIG.project_test_subjects)' in source
    assert "expected_total_samples: int = 1_008_000" in source
    assert "This audit does not train or evaluate a gaze model." in source
    assert "divmod(CONFIG.smoke_samples, len(shuffled_manifest))" in source
    assert 'SOURCE_SHA256 = "__SOURCE_SHA256__"' in source
    assert "urllib" not in source
    assert "requests" not in source


def test_generated_notebook_matches_percent_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    combined = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 14
    assert source_hash in combined
    assert "__SOURCE_SHA256__" not in combined
