"""Check the saved-table reproduction boundary and failure cases."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce_table", ROOT / "scripts/reproduce_table.py")
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


def test_saved_table_rebuilds_every_field():
    source = ROOT / "results/combined60"
    table.check_table(table.rows_from_summaries(source), source / "comparison.csv")


def test_missing_run_is_not_silently_averaged(tmp_path):
    shutil.copytree(ROOT / "results/combined60/runs", tmp_path / "runs")
    (tmp_path / "runs/plain_seed2026/local-summary.json").unlink()
    with pytest.raises(ValueError, match="five completed runs"):
        table.rows_from_summaries(tmp_path)


def test_changed_metric_is_rejected(tmp_path):
    source = ROOT / "results/combined60"
    rows = table.rows_from_summaries(source)
    rows[0]["tiny_fire_AP_pct"] += 1.0
    with pytest.raises(ValueError, match="Mismatch"):
        table.check_table(rows, source / "comparison.csv")
