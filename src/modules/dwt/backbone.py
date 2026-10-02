"""Haar wavelet layers and DWT backbones for YOLO26 detection models."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DWTBlock(nn.Module):
    """Single-level, two-dimensional Haar discrete wavelet transform.

    The transform is fixed and has no learnable parameters. Each input channel
    produces four output subbands, preserving the low-low and three high-frequency
    components at half the input spatial resolution.
    """

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("filters", filters.unsqueeze(1) * 0.5, persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(LL, LH, HL, HH)`` subbands for a BCHW tensor."""
        if x.ndim != 4:
            raise ValueError(f"DWTBlock expects a 4D BCHW tensor, got shape {tuple(x.shape)}")

        _, channels, height, width = x.shape
        if height % 2 or width % 2:
            # Replication keeps the boundary value and makes the transform usable
            # for arbitrary image sizes while leaving even-sized inputs unchanged.
            x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")

        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        transformed = F.conv2d(x, weight, stride=2, groups=channels)
        batch, _, out_height, out_width = transformed.shape
        transformed = transformed.view(batch, channels, 4, out_height, out_width)
        return tuple(transformed[:, :, index] for index in range(4))


class _ConvAct(nn.Sequential):
    """Small convolutional processing block used by the wavelet layers."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int = 3, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.SiLU(inplace=True),
        )


class WaveletFusion(nn.Module):
    """Process each Haar subband and fuse all four into one feature map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("WaveletFusion channel counts must be positive")

        self.band_processors = nn.ModuleList(
            _ConvAct(in_channels, out_channels) for _ in range(4)
        )
        self.fuse = _ConvAct(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, subbands: Sequence[Tensor] | Tensor, *additional: Tensor) -> Tensor:
        """Fuse ``(LL, LH, HL, HH)`` passed as a sequence or four tensors."""
        if additional:
            subbands = (subbands, *additional)
        elif isinstance(subbands, Tensor):
            subbands = (subbands,)
        if len(subbands) != 4:
            raise ValueError(f"WaveletFusion expects four subbands, got {len(subbands)}")
        reference_shape = subbands[0].shape
        if any(band.ndim != 4 or band.shape != reference_shape for band in subbands):
            raise ValueError("All wavelet subbands must be BCHW tensors with equal shapes")
        processed = [processor(band) for processor, band in zip(self.band_processors, subbands)]
        return self.fuse(torch.cat(processed, dim=1))


class _DWTLevel(nn.Module):
    """One DWT level followed by trainable subband processing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.transform = DWTBlock()
        self.fusion = WaveletFusion(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fusion(self.transform(x))


class _DWTBackboneBase(nn.Module):
    """Shared P3 extraction and P5 projection for both DWT experiments."""

    levels: int

    def __init__(
        self,
        in_channels: int = 3,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"DWT YOLO backbones support RGB input (3 channels), got {in_channels}")
        if min(p3_channels, p4_channels, p5_channels) < 1:
            raise ValueError("Backbone channel counts must be positive")

        # Three DWT levels always produce P3 at stride 8.  L4 uses a fourth
        # DWT level for P4; L3 uses a stride-2 convolution for P4 instead.
        stem_channels = max(16, p3_channels // 4)
        intermediate_channels = max(32, p3_channels // 2)
        level_channels = [stem_channels, intermediate_channels, intermediate_channels, p3_channels]
        if self.levels == 4:
            level_channels.append(p4_channels)
        self.stem = _ConvAct(in_channels, stem_channels, kernel_size=3)
        self.dwt_levels = nn.ModuleList(
            _DWTLevel(level_channels[index], level_channels[index + 1])
            for index in range(self.levels)
        )
        if self.levels == 3:
            self.p4_projection = _ConvAct(p3_channels, p4_channels, kernel_size=3, stride=2)
        self.p5_projection = _ConvAct(p4_channels, p5_channels, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return P3, P4 and P5 as separate tensors in neck input order."""
        x = self.stem(x)
        for level in range(3):
            x = self.dwt_levels[level](x)
        p3 = x
        p4 = self.dwt_levels[3](p3) if self.levels == 4 else self.p4_projection(p3)
        p5 = self.p5_projection(p4)
        return p3, p4, p5


class DWTBackboneL3(_DWTBackboneBase):
    """Three Haar levels to P3, then learned stride-2 P4 and P5 projections."""

    levels = 3


class DWTBackboneL4(_DWTBackboneBase):
    """Three Haar levels to P3, fourth Haar level to P4, then learned P5."""

    levels = 4


__all__ = ["DWTBlock", "WaveletFusion", "DWTBackboneL3", "DWTBackboneL4"]
