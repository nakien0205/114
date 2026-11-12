# src/fire_detection/models/common.py
"""
Common neural network layers and utility blocks for lightweight detectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def autopad(k: int | tuple[int, ...], p: int | tuple[int, ...] | None = None, d: int = 1) -> int | tuple[int, ...]:
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (x - 1) + 1 for x in k)
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        if isinstance(act, nn.Module):
            self.act = act
        elif act is True:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
