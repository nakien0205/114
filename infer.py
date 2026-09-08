#!/usr/bin/env python3
"""
Inference and Visual Demonstration CLI for Fire and Smoke Detection.
Runs object detection model on images and outputs visual bounding boxes with labels and confidences.

Usage:
    python infer.py --weights PATH [--source PATH] [--output-dir runs/infer] [--max-images 20]

The default source is loaded from the config.yaml file.
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
    from src.fire_detection.infer import main
except ImportError:
    from fire_detection.infer import main

if __name__ == "__main__":
    sys.exit(main())
