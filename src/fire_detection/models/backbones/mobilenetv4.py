# src/fire_detection/models/backbones/mobilenetv4.py
"""
MobileNetV4 Backbone using official pretrained weights from timm.
Extracts multi-scale intermediate feature maps for edge object detection.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import timm


class MobileNetV4Backbone(nn.Module):
    """
    MobileNetV4 feature extractor initialized with ImageNet-1k pretrained weights.
    
    Default extracts multi-scale feature maps at:
      - P3: Stride 8  (e.g., 80x80 for 640x640 input)
      - P4: Stride 16 (e.g., 40x40 for 640x640 input)
      - P5: Stride 32 (e.g., 20x20 for 640x640 input)
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        out_indices: Sequence[int] = (2, 3, 4),
        freeze_stages: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)

        # Create model extracting intermediate features
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        # Extract output channels and strides for the selected feature maps
        self._out_channels = list(self.backbone.feature_info.channels())
        self._strides = list(self.backbone.feature_info.reduction())

        if freeze_stages > 0:
            self._freeze(freeze_stages)

    @property
    def out_channels(self) -> List[int]:
        """Channels for [P3, P4, P5]."""
        return self._out_channels

    @property
    def strides(self) -> List[int]:
        """Strides for [P3, P4, P5]."""
        return self._strides

    def _freeze(self, freeze_stages: int) -> None:
        """Optionally freeze early feature extraction stages for transfer learning."""
        params = list(self.backbone.parameters())
        freeze_count = min(len(params), freeze_stages * 10)
        for p in params[:freeze_count]:
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        Returns:
            List of feature tensors [P3, P4, P5].
        """
        return self.backbone(x)
