"""Approach 1: add wavelet refinement after existing Conv, before CSP.

ConvThenHaar retains the original Conv/BN/activation modules and their state
keys. Only the added residual path changes features. Both normal and fused
inference execute that path.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv

from .haar import haar_dwt2, haar_idwt2


class HaarResidual(nn.Module):
    """One-level all-band processing, inverse reconstruction and channel fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.band_conv = nn.Conv2d(4 * channels, 4 * channels, 3, padding=1,
                                   groups=4 * channels, bias=False)
        self.act = nn.SiLU(inplace=False)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected NCHW features with {self.channels} channels")
        bands = haar_dwt2(x)
        processed = self.act(self.band_conv(bands.flatten(1, 2))).reshape_as(bands)
        reconstructed = haar_idwt2(processed, x.shape[-2:])
        return x + self.project(reconstructed)


class ConvThenHaar(Conv):
    """Keep an existing Ultralytics Conv and append a same-size Haar residual.

    Inheriting Conv also lets Ultralytics fuse the retained Conv/BatchNorm in
    its usual way. forward_fuse must retain the added wavelet path.
    """

    def __init__(self, original: Conv) -> None:
        nn.Module.__init__(self)
        if type(original) is not Conv or not hasattr(original, "bn"):
            raise ValueError("Insertion requires an unfused standard Ultralytics Conv")
        self.conv = original.conv
        self.bn = original.bn
        self.act = original.act
        self.dwt = HaarResidual(self.conv.out_channels).to(
            device=self.conv.weight.device, dtype=self.conv.weight.dtype
        )
        for name in ("i", "f"):
            setattr(self, name, getattr(original, name))
        self.type = "ConvThenHaar"
        self.np = sum(parameter.numel() for parameter in self.parameters())
        self.train(original.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwt(super().forward(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwt(super().forward_fuse(x))
