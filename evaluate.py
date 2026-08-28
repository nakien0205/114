#!/usr/bin/env python3
"""
Evaluation and Quantitative Metrics Reporting CLI for Fire and Smoke Detection.
Computes Precision, Recall, mAP50, mAP50-95 on validation/test sets, and exports JSON and CSV reports.

Usage:
    python evaluate.py --weights PATH [--data PATH] [--split test] [--save-json metrics.json] [--save-csv metrics.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace root and src to python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from src.fire_detection.evaluate import main
except ImportError:
    from fire_detection.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
