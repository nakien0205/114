"""Guidance-supervision control with an exact identity P2 feature path."""
from __future__ import annotations

import ultralytics.nn.tasks as tasks
from ultralytics.nn.tasks import DetectionModel

from .neck import SmokeToFireNeck
from .model import SmokeFireDetectionModel, proposal_yaml
from .trainer import SmokeFireTrainer


class GuidanceOnlyNeck(SmokeToFireNeck):
    """Keep parameter construction/order, but learn only occupancy guidance.

    The unused routing parameters preserve the proposal's initialization RNG
    consumption and optimizer groups. They have no loss path in this control.
    """

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Unexpected P2 feature shape")
        self.semantic_logits = self.guidance(x)
        return x


tasks.GuidanceOnlyNeck = GuidanceOnlyNeck


def guidance_yaml(base_model):
    config = proposal_yaml(base_model)
    config["head"][-2][2] = "GuidanceOnlyNeck"
    config["yaml_file"] = "yolo26-p2-guidance-only.yaml"
    return config


class GuidanceOnlyDetectionModel(SmokeFireDetectionModel):
    """The same 0.1-weight guidance loss with no routed feature update."""

    def __init__(self, cfg, ch=3, nc=2, verbose=False):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        if not isinstance(self.smoke_neck, GuidanceOnlyNeck):
            raise TypeError("Guidance control requires its identity module")


class GuidanceOnlyTrainer(SmokeFireTrainer):
    expected_base_tensor_sha256 = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 2 or self.data["names"] != {0: "fire", 1: "smoke"}:
            raise ValueError("The approved Home Fire mapping is 0=fire, 1=smoke")
        base = DetectionModel(cfg, nc=2, ch=self.data["channels"], verbose=False)
        if weights is not None:
            base.load(weights)
        model = GuidanceOnlyDetectionModel(
            guidance_yaml(base), nc=2, ch=self.data["channels"], verbose=verbose
        )
        model.load_base(base)
        if self.expected_base_tensor_sha256 is not None and (
            model.base_transfer["canonical_base_tensor_sha256"] != self.expected_base_tensor_sha256
        ):
            raise RuntimeError("Control initialization differs from the full proposal's recorded base tensors")
        return model
