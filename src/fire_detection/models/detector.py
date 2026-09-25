# src/fire_detection/models/detector.py
"""
Unified Detector Architecture: MobileNetV4 + Standard PAN + Decoupled Detection Head.
Baseline fire and smoke detector for edge deployment without attention mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect

from .backbones.mobilenetv4 import MobileNetV4Backbone
from .necks.pan import StandardPAN


class MobileNetV4_PAN(nn.Module):
    """
    MobileNetV4 Backbone + Standard PAN Neck + YOLO Decoupled Detection Head.
    """

    def __init__(
        self,
        nc: int = 2,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        neck_channels: int = 128,
        freeze_stages: int = 0,
    ) -> None:
        """
        Args:
            nc: Number of target classes (default: 2 for fire, smoke).
            backbone_name: MobileNetV4 variant from timm (default: mobilenetv4_conv_small).
            pretrained: Whether to initialize backbone with ImageNet-1k pretrained weights.
            neck_channels: Channel width of the PAN neck layers (default: 128).
            freeze_stages: Number of initial stages in backbone to freeze.
        """
        super().__init__()
        self.nc = nc
        self.backbone_name = backbone_name
        self.neck_channels = neck_channels

        # 1. Pretrained MobileNetV4 Backbone
        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(2, 3, 4),
            freeze_stages=freeze_stages,
        )

        # 2. Standard PAN Neck (No Attention)
        self.neck = StandardPAN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
            attention_factory=None,  # Baseline: Attention is None
        )

        # 3. YOLO Decoupled Anchor-Free Detection Head
        self.head = Detect(
            nc=nc,
            ch=[neck_channels, neck_channels, neck_channels],
        )
        # Register strides (P3: 8, P4: 16, P5: 32)
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x: torch.Tensor) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through Backbone -> Neck -> Detection Head.
        In training: returns dict with 'boxes', 'scores', 'feats'.
        In eval: returns tuple (preds, feats).
        """
        feats = self.backbone(x)
        pan_feats = self.neck(feats)
        return self.head(pan_feats)

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        neck_total = sum(p.numel() for p in self.neck.parameters())
        head_total = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_total,
            "neck": neck_total,
            "head": head_total,
        }


def build_mobilenetv4_pan(
    nc: int = 2,
    backbone_name: str = "mobilenetv4_conv_small",
    pretrained: bool = True,
    neck_channels: int = 128,
    freeze_stages: int = 0,
) -> MobileNetV4_PAN:
    """Factory function to build and configure the baseline detector."""
    model = MobileNetV4_PAN(
        nc=nc,
        backbone_name=backbone_name,
        pretrained=pretrained,
        neck_channels=neck_channels,
        freeze_stages=freeze_stages,
    )
    return model
