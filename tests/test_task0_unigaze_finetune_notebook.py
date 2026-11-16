from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "notebooks" / "task0_unigaze_b16_finetune.py"
NOTEBOOK = ROOT / "notebooks" / "task0_unigaze_b16_finetune.ipynb"


def test_source_exposes_fixed_server_performance_controls() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    for variable in (
        "TASK0_PROJECT_ROOT",
        "TASK0_RUNTIME_ROOT",
        "TASK0_UNIGAZE_ROOT",
        "TASK0_OUTPUT_BASE",
        "TASK0_RUN_ID",
        "TASK0_BATCH_SIZE",
        "TASK0_EVAL_BATCH_SIZE",
        "TASK0_NUM_WORKERS",
        "TASK0_PREFETCH_FACTOR",
        "TASK0_PIN_MEMORY",
        "TASK0_PERSISTENT_WORKERS",
        "TASK0_EVALUATE_TRAIN_EACH_EPOCH",
        "TASK0_PRETRAINED_CHECKPOINT",
    ):
        assert variable in source
    assert '"batch_size": batch_size' in source
    assert '"pin_memory": CONFIG.pin_memory' in source
    assert "CONFIG.num_workers > 0 and CONFIG.persistent_workers" in source
    assert 'kwargs["prefetch_factor"] = CONFIG.prefetch_factor' in source
    assert "except (OSError, subprocess.CalledProcessError):" in source


def test_notebook_matches_source() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    combined = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert source_hash in combined
    assert "__SOURCE_SHA256__" not in combined
