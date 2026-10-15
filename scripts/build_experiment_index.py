#!/usr/bin/env python3
"""Build one comparison table from file-based experiment records."""

from __future__ import annotations

import argparse
from pathlib import Path

from gaze_estimation.experiment_tracking import collect_run_rows, write_run_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs",
        type=Path,
        action="append",
        required=True,
        help="Run root to scan; repeat for local and downloaded Kaggle outputs.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("results/experiment_index.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_run_rows(args.outputs)
    write_run_index(rows, args.destination)
    print(f"Indexed {len(rows)} runs into {args.destination}")


if __name__ == "__main__":
    main()
