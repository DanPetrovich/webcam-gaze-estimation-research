"""File-based experiment tracking for local, Kaggle, and cluster runs.

The tracker deliberately depends only on the Python standard library.  A run is
identified by its resolved configuration *and* the content of the source files
that produced it, so changing code cannot silently reuse an old checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_STATUSES = {"running", "completed", "failed", "cancelled"}


def stable_json_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for a JSON-serializable value."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    """Describe and fingerprint the exact source files used by a run."""
    root = root.resolve()
    files: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        try:
            portable_path = str(path.relative_to(root))
        except ValueError:
            portable_path = str(path)
        files.append(
            {
                "path": portable_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"sha256": stable_json_hash(files), "files": files}


def git_state(repository: Path) -> dict[str, Any]:
    """Return Git provenance without requiring the checkout to be clean."""

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    try:
        revision = git("rev-parse", "HEAD")
        status_lines = tuple(
            line for line in git("status", "--porcelain=v1").splitlines() if line
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "status_sha256": None}
    return {
        "revision": revision,
        "dirty": bool(status_lines),
        "status_sha256": stable_json_hash(status_lines),
    }


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    if not normalized:
        raise ValueError("Experiment name must contain a letter or digit")
    return normalized


def _atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def start_run(
    *,
    experiment: str,
    config: Mapping[str, Any],
    output_root: Path,
    repository_root: Path,
    source_paths: Sequence[Path],
    dataset_manifest: Mapping[str, Any] | None = None,
    external_ids: Mapping[str, str] | None = None,
    started_at: datetime | None = None,
) -> Path:
    """Create an immutable run identity and its initial manifest.

    ``external_ids`` can hold a Kaggle notebook slug/version, W&B run ID, or a
    cluster job ID. Secrets and credentials must never be placed there.
    """
    started_at = started_at or datetime.now(UTC)
    if started_at.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    resolved_config = dict(config)
    config_hash = stable_json_hash(resolved_config)
    sources = source_manifest(source_paths, repository_root)
    dataset_hash = None
    if dataset_manifest is not None:
        dataset_hash = stable_json_hash(dict(dataset_manifest))
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = (
        f"{timestamp}_{_slug(experiment)}_"
        f"c{config_hash[:8]}_s{sources['sha256'][:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "metrics", "plots"):
        (run_dir / name).mkdir()

    _atomic_json_write(resolved_config, run_dir / "resolved_config.json")
    _atomic_json_write(sources, run_dir / "source_manifest.json")
    if dataset_manifest is not None:
        _atomic_json_write(dict(dataset_manifest), run_dir / "dataset_manifest.json")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": experiment,
        "status": "running",
        "started_at": started_at.astimezone(UTC).isoformat(),
        "finished_at": None,
        "config_sha256": config_hash,
        "source_sha256": sources["sha256"],
        "dataset_sha256": dataset_hash,
        "git": git_state(repository_root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "external_ids": dict(external_ids or {}),
    }
    _atomic_json_write(manifest, run_dir / "run_manifest.json")
    return run_dir


def finish_run(
    run_dir: Path,
    *,
    status: str,
    summary: Mapping[str, Any] | None = None,
    note: str | None = None,
    finished_at: datetime | None = None,
) -> None:
    """Finalize a run while preserving its original identity and provenance."""
    if status not in RUN_STATUSES - {"running"}:
        raise ValueError(
            f"Final status must be completed, failed, or cancelled: {status}"
        )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "running":
        raise RuntimeError(f"Run is already finalized as {manifest['status']}")
    if summary is not None:
        _atomic_json_write(dict(summary), run_dir / "summary.json")
    finished_at = finished_at or datetime.now(UTC)
    if finished_at.tzinfo is None:
        raise ValueError("finished_at must be timezone-aware")
    manifest["status"] = status
    manifest["finished_at"] = finished_at.astimezone(UTC).isoformat()
    manifest["note"] = note
    _atomic_json_write(manifest, manifest_path)


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_scalars(child, child_prefix))
    elif value is None or isinstance(value, (str, int, float, bool)):
        result[prefix] = value
    return result


def collect_run_rows(output_roots: Sequence[Path]) -> list[dict[str, Any]]:
    """Collect comparable scalar metadata and metrics from multiple run roots."""
    rows: list[dict[str, Any]] = []
    for output_root in output_roots:
        for manifest_path in sorted(output_root.rglob("run_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = {
                "run_dir": str(manifest_path.parent),
                **_flatten_scalars(manifest),
            }
            summary_path = manifest_path.parent / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        f"metric.{key}": value
                        for key, value in _flatten_scalars(summary).items()
                    }
                )
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("started_at", "")))


def write_run_index(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    """Write a deterministic CSV index suitable for Git and spreadsheet tools."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
