#!/usr/bin/env python3
"""
Root pipeline runner script for Fire and Smoke Dataset Audit, Segmentation, Partitioning & Verification.

Usage:
    python scripts/run_pipeline.py all [--data-dir PATH] [--output-dir PATH] [--train-ratio 0.8] [--seed 42] [--check-hashes]
    python scripts/run_pipeline.py audit [--data-dir PATH] [--output-dir PATH]
    python scripts/run_pipeline.py segment [--data-dir PATH] [--output-dir PATH] [--dhash-threshold 18] [--min-length 5]
    python scripts/run_pipeline.py prepare [--data-dir PATH] [--output-dir PATH] [--train-ratio 0.8] [--seed 42] [--check-hashes]
    python scripts/run_pipeline.py verify [--data-yaml data.yaml] [--manifest-json PATH] [--manifest-csv PATH] [--tolerance 1e-4] [--check-hashes]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository and src/ are on the import path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from fire_audit.cli import main
except ImportError:
    from fire_audit.cli import main

if __name__ == "__main__":
    sys.exit(main())
