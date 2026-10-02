"""DDP-safe trainers for YOLO26 models with DWT backbones."""

from ultralytics.models.yolo.detect import DetectionTrainer

from src.modules.nwd.trainer import NWDDetectionTrainer
from src.modules.wiou.trainer import WIoUDetectionTrainer

from . import register_dwt_modules


class _DWTRegistrationMixin:
    def get_model(self, cfg=None, weights=None, verbose=True):
        # Ultralytics rebuilds YAML models inside fresh DDP worker processes.
        register_dwt_modules()
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)


class DWTDetectionTrainer(_DWTRegistrationMixin, DetectionTrainer):
    """Standard detection loss with DWT modules registered in each worker."""


class DWTNWDDetectionTrainer(_DWTRegistrationMixin, NWDDetectionTrainer):
    """NWD loss with DWT modules registered in each worker."""


class DWTWIoUDetectionTrainer(_DWTRegistrationMixin, WIoUDetectionTrainer):
    """WIoU loss with DWT modules registered in each worker."""
