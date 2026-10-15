#!/usr/bin/env python3
"""Convert a lightweight ``# %%`` Python notebook source into ``.ipynb``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def markdown_source(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if line.startswith("# "):
            result.append(line[2:])
        elif line.startswith("#"):
            result.append(line[1:])
        else:
            result.append(line)
    return result


def convert(source_path: Path, destination: Path) -> None:
    raw_source = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()
    lines = raw_source.splitlines(keepends=True)
    cells: list[dict[str, object]] = []
    kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if kind is None:
            if any(line.strip() for line in buffer):
                raise ValueError("Content exists before the first # %% marker")
            buffer = []
            return
        cell_source = markdown_source(buffer) if kind == "markdown" else buffer
        text = "".join(cell_source).replace("__SOURCE_SHA256__", source_hash)
        cell: dict[str, object] = {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
        buffer = []

    for line in lines:
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            buffer.append(line)
    flush()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": str(source_path),
                "destination": str(destination),
                "cells": len(cells),
                "source_sha256": source_hash,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
