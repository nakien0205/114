# src/fire_detection/models/necks/pan.py
"""
Standard Path Aggregation Network (PAN) Neck.
Fuses multi-scale features (P3, P4, P5) through bidirectional (top-down + bottom-up) pathways.
Designed with pluggable attention hook (None for baseline).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ConvBNAct


class StandardPAN(nn.Module):
    """
    Standard Path Aggregation Network (PANet).
    
    Feature flow:
      1. Lateral 1x1 convs project [P3, P4, P5] to uniform channel depth `out_channels`.
      2. Top-down FPN:
         - P5 lateral upsampled -> fused with P4 lateral -> smooth conv -> P4_td
         - P4_td upsampled -> fused with P3 lateral -> smooth conv -> Out_P3
      3. Bottom-up PAN:
         - Out_P3 downsampled -> fused with P4_td -> smooth conv -> Out_P4
         - Out_P4 downsampled -> fused with P5 lateral -> smooth conv -> Out_P5
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 128,
        attention_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        """
        Args:
            in_channels: List of input channel counts for [P3, P4, P5], e.g. [64, 96, 960].
            out_channels: Uniform channel width across all neck stages (default: 128).
            attention_factory: Optional factory returning an attention nn.Module(channels).
                               None for baseline.
        """
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"StandardPAN expects exactly 3 feature stages (P3, P4, P5), got {len(in_channels)}")

        self.in_channels = list(in_channels)
        self.out_channels = out_channels

        c3, c4, c5 = in_channels
        c_out = out_channels

        # 1. Lateral Projections (1x1 Conv)
        self.lateral_c5 = ConvBNAct(c5, c_out, kernel_size=1)
        self.lateral_c4 = ConvBNAct(c4, c_out, kernel_size=1)
        self.lateral_c3 = ConvBNAct(c3, c_out, kernel_size=1)

        # 2. Top-Down Pathway (FPN)
        # Fusing P5 upsample with P4 lateral
        self.fpn_conv4 = ConvBNAct(c_out * 2, c_out, kernel_size=3)
        # Fusing P4 upsample with P3 lateral
        self.fpn_conv3 = ConvBNAct(c_out * 2, c_out, kernel_size=3)

        # 3. Bottom-Up Pathway (PAN)
        # Downsampling P3 to P4
        self.downsample_p3 = ConvBNAct(c_out, c_out, kernel_size=3, stride=2)
        self.pan_conv4 = ConvBNAct(c_out * 2, c_out, kernel_size=3)

        # Downsampling P4 to P5
        self.downsample_p4 = ConvBNAct(c_out, c_out, kernel_size=3, stride=2)
        self.pan_conv5 = ConvBNAct(c_out * 2, c_out, kernel_size=3)

        # 4. Optional Attention Hooks (None for baseline)
        self.att3 = attention_factory(c_out) if attention_factory else nn.Identity()
        self.att4 = attention_factory(c_out) if attention_factory else nn.Identity()
        self.att5 = attention_factory(c_out) if attention_factory else nn.Identity()

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] from backbone.
        Returns:
            [Out_P3, Out_P4, Out_P5] each with channel depth `out_channels`.
        """
        p3, p4, p5 = features

        # Lateral projections
        lat_5 = self.lateral_c5(p5)
        lat_4 = self.lateral_c4(p4)
        lat_3 = self.lateral_c3(p3)

        # Top-Down FPN
        up_5 = F.interpolate(lat_5, size=lat_4.shape[-2:], mode="nearest")
        td_4 = self.fpn_conv4(torch.cat([lat_4, up_5], dim=1))

        up_4 = F.interpolate(td_4, size=lat_3.shape[-2:], mode="nearest")
        out_p3 = self.fpn_conv3(torch.cat([lat_3, up_4], dim=1))
        out_p3 = self.att3(out_p3)

        # Bottom-Up PAN
        down_3 = self.downsample_p3(out_p3)
        out_p4 = self.pan_conv4(torch.cat([td_4, down_3], dim=1))
        out_p4 = self.att4(out_p4)

        down_4 = self.downsample_p4(out_p4)
        out_p5 = self.pan_conv5(torch.cat([lat_5, down_4], dim=1))
        out_p5 = self.att5(out_p5)

        return [out_p3, out_p4, out_p5]
