from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "notebooks" / "task1_gazegene_b16_finetune.py"
NOTEBOOK = ROOT / "notebooks" / "task1_gazegene_b16_finetune.ipynb"


def test_source_encodes_fixed_protocol_and_safe_outputs() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "tuple(range(1, 41))" in source
    assert "tuple(range(41, 47))" in source
    assert "tuple(range(47, 57))" in source
    assert "vector_to_pitchyaw" in source
    assert "best_backbone_mae_format.pth" in source
    assert "subject_macro_mean_deg" in source
    assert "constant_direction_baseline" in source
    assert "bootstrap_subject_mean_ci" in source
    assert "sample_micro_p95_deg" in source
    assert "TASK1_FULL_CHECKPOINT" in source
    assert "TASK1_RUN_ID" in source
    assert "last_state.pt" in source
    assert "atomic_torch_save" in source
    assert "already exists with a different resolved config" in source
    assert "requests" not in source
    assert "wandb" not in source.lower()


def test_notebook_matches_source() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    combined = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert source_hash in combined
    assert "__SOURCE_SHA256__" not in combined
