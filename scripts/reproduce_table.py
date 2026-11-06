#!/usr/bin/env python3
"""Rebuild the combined60 table from saved scored summaries, without inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {("plain", 2026), ("adaptive", 42), ("adaptive", 2026), ("dense", 42), ("dense", 2026)}
FIELDS = [
    "model", "seed", "tiny_fire_AP_pct", "small_fire_AP_pct", "small_smoke_AP_pct",
    "overall_AP_pct", "overall_AP50_pct", "tiny_recall_at_30FP_per100_pct",
    "small_recall_at_30FP_per100_pct", "false_fire_boxes_at_ceiling",
    "false_fire_boxes_per100", "negative_alarm_images_at_ceiling", "negative_images",
    "threshold_at_ceiling", "false_fire_boxes_at90", "false_fire_boxes_per100_at90",
    "training_hours", "parameters",
]


def rows_from_summaries(root: Path) -> list[dict]:
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "runs").glob("*/local-summary.json")]
    keys = [(s["model"], s["seed"]) for s in summaries]
    if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
        raise ValueError("Expected the five completed runs; plain seed42 must not be imputed")
    summaries.sort(key=lambda s: (["plain", "adaptive", "dense"].index(s["model"]), s["seed"]))
    rows = []
    for s in summaries:
        op = s["operating_points"]["combined"]
        at_ceiling = op["at_30_false_boxes_per100_ceiling"]
        at90 = op["at_90percent_tiny_recall"]
        normal = s["normal"]["overall"]
        rows.append(dict(zip(FIELDS, [
            s["model"], s["seed"], 100*s["tiny_fire_AP"], 100*s["small_fire_AP"],
            100*s["small_smoke_AP"], 100*normal["metrics/mAP50-95(B)"],
            100*normal["metrics/mAP50(B)"], 100*at_ceiling["tiny_fire_recall"],
            100*at_ceiling["small_fire_recall"], at_ceiling["false_fire_boxes"],
            at_ceiling["false_fire_boxes_per100"], at_ceiling["fire_free_alarm_images"],
            at_ceiling["fire_free_images"], at_ceiling["confidence"],
            at90["false_fire_boxes"] if at90 else "",
            at90["false_fire_boxes_per100"] if at90 else "", s["training_hours"], s["parameters"],
        ])))
    return rows


def check_table(rows: list[dict], reference: Path) -> None:
    with reference.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected reference columns")
        saved = list(reader)
    if len(saved) != len(rows):
        raise ValueError("Reference row count differs")
    for actual, expected in zip(rows, saved):
        for field in FIELDS:
            value, old = actual[field], expected[field]
            equal = str(value) == old
            if isinstance(value, (int, float)) and old:
                equal = math.isfinite(float(value)) and math.isclose(float(value), float(old), rel_tol=1e-12, abs_tol=1e-10)
            if not equal:
                raise ValueError(f"Mismatch: {actual['model']} seed{actual['seed']} / {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "combined60")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare rebuilt values with the saved CSV")
    mode.add_argument("--output", type=Path, help="Write a new CSV; an existing file is never overwritten")
    args = parser.parse_args()
    try:
        rows = rows_from_summaries(args.input)
        if args.check:
            check_table(rows, args.input / "comparison.csv")
            print("PASS: all 5 rows and 19 columns reproduce from saved scored summaries.")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {args.output}")
        print("This rebuilds a table; it does not rescore predictions or repeat training/inference.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
