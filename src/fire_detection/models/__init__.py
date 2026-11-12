# src/fire_detection/models/__init__.py
"""
MobileNetV4 backbone with the implemented standard PAN neck and detection head.
"""

from .detector import MobileNetV4_PAN, build_mobilenetv4_pan

__all__ = ["MobileNetV4_PAN", "build_mobilenetv4_pan"]
