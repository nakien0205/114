"""
Fire and Smoke Detection Pipeline based on YOLOv8-P2 Transfer Learning.
"""

from .dataset import DatasetVerifier, verify_dataset
from .evaluate import evaluate_yolo
from .infer import infer_yolo
from .train import train_yolo

__all__ = [
    "DatasetVerifier",
    "verify_dataset",
    "train_yolo",
    "evaluate_yolo",
    "infer_yolo",
]
