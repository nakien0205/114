#!/usr/bin/env python3
"""
Transfer Learning Training Pipeline CLI for Fire and Smoke Detection.
Loads the configured pretrained checkpoint and fine-tunes on the configured dataset.

Usage:
    python train.py [--weights PATH] [--data PATH] [--epochs N] [--batch N] [--imgsz N] [--device 0]

Values omitted from the command line are loaded from the config.yaml file.
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
    from src.fire_detection.train import main
except ImportError:
    from fire_detection.train import main

if __name__ == "__main__":
    sys.exit(main())
