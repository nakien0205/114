# src/fire_detection/models/__init__.py
"""
Modular Object Detection Architectures for Fire and Smoke Detection.
Supports custom backbones (MobileNetV4), necks (PAN, PAN-weight, PAN-Bag, BiFPN),
and attention modules (None, Coordinate Attention, SAM/CAM, EMA).
"""

from .detector import MobileNetV4_PAN, build_mobilenetv4_pan

__all__ = ["MobileNetV4_PAN", "build_mobilenetv4_pan"]
