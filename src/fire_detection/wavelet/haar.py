"""Differentiable orthonormal Haar analysis/synthesis for NCHW features.

Band order is LL, LH, HL, HH; letters refer to height then width filtering.
LH responds to variation across width, HL to variation across height.
No learned weights, external wavelet package, or dataset access is involved.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def haar_dwt2(x: torch.Tensor) -> torch.Tensor:
    """Return [N,C,4,ceil(H/2),ceil(W/2)], replicating odd boundaries.

    Each 2x2 analysis filter has entries +/- 1/2. This normalization makes
    the even-size transform orthonormal, rather than an average-pooling map.
    """
    if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("Haar DWT requires a nonempty floating-point NCHW tensor")
    height, width = x.shape[-2:]
    if height % 2 or width % 2:
        x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.stack(
        ((a + b + c + d) * 0.5,
         (a - b + c - d) * 0.5,
         (a + b - c - d) * 0.5,
         (a - b - c + d) * 0.5),
        dim=2,
    )


def haar_idwt2(bands: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Reconstruct NCHW features; optionally crop the replicated boundary."""
    if bands.ndim != 5 or bands.shape[2] != 4 or min(bands.shape) < 1 or not bands.is_floating_point():
        raise ValueError("Inverse Haar DWT requires a nonempty floating-point [N,C,4,H,W] tensor")
    ll, lh, hl, hh = bands.unbind(dim=2)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    # Interleave columns first, then rows. Stack/reshape retain autograd.
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    reconstructed = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    if size is not None:
        height, width = size
        full_height, full_width = reconstructed.shape[-2:]
        if height not in (full_height, full_height - 1) or width not in (full_width, full_width - 1) or min(size) < 1:
            raise ValueError("Crop size must match the original even or odd DWT input")
        reconstructed = reconstructed[..., :height, :width]
    return reconstructed
