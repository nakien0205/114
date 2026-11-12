"""Trainer integration for one proposed model; never launches a baseline run."""
from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .model import SmokeFireDetectionModel, proposal_yaml


class SmokeFireTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Logging is handled by the launcher with one explicit online W&B run.
        for event, callbacks in self.callbacks.items():
            self.callbacks[event] = [f for f in callbacks if not f.__module__.endswith(".wb")]

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 2 or self.data["names"] != {0: "fire", 1: "smoke"}:
            raise ValueError("The approved Home Fire mapping is 0=fire, 1=smoke")
        # Build and initialize the ordinary architecture only to transfer its
        # tensors. There is no baseline optimizer, dataset loop or baseline run.
        base = DetectionModel(cfg, nc=2, ch=self.data["channels"], verbose=False)
        if weights is not None:
            base.load(weights)
        model = SmokeFireDetectionModel(proposal_yaml(base), nc=2, ch=self.data["channels"], verbose=verbose)
        model.load_base(base)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "guidance_loss"
        return validator
