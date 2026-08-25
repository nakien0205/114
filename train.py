#!/usr/bin/env python3
"""
Transfer Learning Training Pipeline CLI for Fire and Smoke Detection.
Loads pretrained YOLOv8-P2 checkpoint and fine-tunes on Home Fire Dataset.

Usage:
    python train.py [--weights D:/Python/Projects/Maritime-SAR/best.pt] [--data data.yaml] [--epochs 50] [--batch 16] [--imgsz 640] [--device 0]
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
