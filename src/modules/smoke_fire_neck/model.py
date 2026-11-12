"""Explicit P2 neck node; standard YOLO26 end-to-end Detect head is preserved."""
from __future__ import annotations

from copy import deepcopy
import hashlib

import torch
import ultralytics.nn.tasks as tasks
from ultralytics.nn.tasks import DetectionModel

from .neck import SmokeToFireNeck, guidance_loss

# Runtime parser registration. No installed library file is changed.
tasks.SmokeToFireNeck = SmokeToFireNeck


def proposal_yaml(base_model):
    config = deepcopy(base_model.yaml)
    detect = config["head"].pop()
    sources = list(detect[0])
    if sources != [19, 22, 25, 28] or base_model.stride.tolist() != [4, 8, 16, 32]:
        raise ValueError("This first candidate is restricted to the verified YOLO26n-P2 graph")
    channels = base_model.model[-1].cv2[0][0].conv.in_channels
    new_index = len(config["backbone"]) + len(config["head"])
    config["head"].append([sources[0], 1, "SmokeToFireNeck", [channels, 16, 8, 7, 2, 0.2]])
    config["head"].append([[new_index, *sources[1:]], detect[1], detect[2], detect[3]])
    config["yaml_file"] = "yolo26-p2-smoke-fire-neck.yaml"
    return config


class SmokeFireDetectionModel(DetectionModel):
    auxiliary_weight = 0.1

    @property
    def smoke_neck(self):
        neck = self.model[-2]
        if not isinstance(neck, SmokeToFireNeck):
            raise TypeError("Expected the explicit smoke-to-fire neck node")
        return neck

    def __init__(self, cfg, ch=3, nc=2, verbose=False):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if nc != 2 or not self.end2end or self.model[-1].f != [29, 22, 25, 28]:
            raise ValueError("Unexpected class count, head mode or neck wiring")
        self.smoke_neck.semantic_logits = None

    def load_base(self, base_model):
        """Preserve every base tensor, including the shifted Detect node."""
        original = base_model.state_dict()
        source_head, target_head = base_model.model[-1].i, self.model[-1].i
        prefix = f"model.{source_head}."
        renamed = {f"model.{target_head}." + k[len(prefix):] if k.startswith(prefix) else k: v
                   for k, v in original.items()}
        result = self.load_state_dict(renamed, strict=False)
        if result.unexpected_keys or any(not k.startswith("model.29.") for k in result.missing_keys):
            raise ValueError("Pretrained base transfer left unexpected mismatches")
        current = self.state_dict()
        if any(not torch.equal(current[k], v) for k, v in renamed.items()):
            raise ValueError("A base tensor changed during proposal construction")
        digest = hashlib.sha256()
        for k in sorted(original):
            value = original[k].detach().cpu().contiguous()
            digest.update(k.encode() + b"\0" + str(value.shape).encode() + b"\0")
            digest.update(value.numpy().tobytes())
        self.base_transfer = {"base_tensor_count": len(original), "verified_equal": True,
                              "canonical_base_tensor_sha256": digest.hexdigest(),
                              "new_tensor_keys": result.missing_keys}

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.forward(batch["img"])
        detection, items = super().loss(batch, preds)
        logits = self.smoke_neck.semantic_logits
        if logits is None:
            raise RuntimeError("Guidance logits missing from the current forward pass")
        auxiliary = self.auxiliary_weight * guidance_loss(logits, batch)
        self.smoke_neck.semantic_logits = None
        scaled = auxiliary * batch["img"].shape[0]
        loss = torch.cat((detection.reshape(-1), scaled.reshape(1)))
        return loss, torch.cat((items.reshape(-1), auxiliary.detach().reshape(1)))
