#!/usr/bin/env python3
"""Build a deterministic normalized webcam calibration stimulus manifest."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from gaze_estimation.calibration_protocol import build_plan, plan_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    seed = int(protocol["seed"])
    plan = build_plan(
        rows=int(protocol["rows"]),
        columns=int(protocol["columns"]),
        margin_fraction=float(protocol["margin_fraction"]),
        repetitions=int(protocol["repetitions"]),
        settle_ms=int(protocol["settle_ms"]),
        capture_ms=int(protocol["capture_ms"]),
        seed=seed,
        shuffle_within_repetition=bool(protocol["shuffle_within_repetition"]),
    )
    manifest = plan_manifest(plan, protocol_name=str(protocol["name"]), seed=seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "stimulus_count": manifest["stimulus_count"],
                "planned_duration_ms": manifest["planned_duration_ms"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
