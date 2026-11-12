"""Local smoke context with predicted class gates; no labels at inference."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SmokeToFireNeck(nn.Module):
    """Refine P2 at predicted fire locations using nearby predicted smoke evidence.

    The two guidance logits are supervised with box-derived occupancy, not
    segmentation masks. Routing gates are detached from detection gradients.
    A mistaken guidance prediction can still produce a mistaken update.
    """

    def __init__(self, channels=32, context_channels=16, key_channels=8,
                 kernel_size=7, pool_stride=2, max_residual=0.2):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1 or pool_stride < 1:
            raise ValueError("Use an odd neighborhood of at least 3 and positive pooling stride")
        if not 0 < max_residual <= 1:
            raise ValueError("max_residual must be in (0, 1]")
        self.channels = channels
        self.context_channels = context_channels
        self.key_channels = key_channels
        self.kernel_size = kernel_size
        self.pool_stride = pool_stride
        self.max_residual = float(max_residual)
        self.guidance = nn.Conv2d(channels, 2, 1)
        self.query = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, context_channels, 1, bias=False)
        self.project = nn.Conv2d(context_channels, channels, 1, bias=False)
        self.offset_bias = nn.Parameter(torch.zeros(kernel_size * kernel_size))
        # Initial alpha=.02; it stays below the declared .2 bound.
        self.gain_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        nn.init.constant_(self.guidance.bias, -2.0)
        self.semantic_logits = None

    @staticmethod
    def evidence(logits):
        return (2 * logits.float().sigmoid() - 1).clamp(0, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Unexpected P2 feature shape")
        logits = self.guidance(x)
        self.semantic_logits = logits
        gates = self.evidence(logits).detach()
        receiving_fire = gates[:, 0:1]
        # Suppress source locations that themselves have strong fire evidence.
        source_smoke = gates[:, 1:2] * (1 - gates[:, 0:1])
        small = F.avg_pool2d(x, self.pool_stride, self.pool_stride, ceil_mode=True)
        source = F.avg_pool2d(source_smoke, self.pool_stride, self.pool_stride, ceil_mode=True)
        batch, _, height, width = small.shape
        positions = height * width
        neighbors = self.kernel_size ** 2
        padding = self.kernel_size // 2
        q = self.query(small).float().reshape(batch, self.key_channels, 1, positions)
        k = F.unfold(self.key(small), self.kernel_size, padding=padding).float()
        k = k.reshape(batch, self.key_channels, neighbors, positions)
        v = F.unfold(self.value(small), self.kernel_size, padding=padding).float()
        v = v.reshape(batch, self.context_channels, neighbors, positions)
        availability = F.unfold(source, self.kernel_size, padding=padding).float()
        valid = F.unfold(torch.ones_like(source), self.kernel_size, padding=padding).bool()
        # Exclude the receiving cell itself; safely handle a grid with no neighbors.
        valid[:, neighbors // 2, :] = False
        scores = (q * k).sum(1) / math.sqrt(self.key_channels)
        scores = scores + self.offset_bias.float()[None, :, None]
        scores = scores + availability.clamp_min(1e-6).log()
        weights = scores.masked_fill(~valid, -1e4).softmax(1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        context = (weights[:, None] * v).sum(2).reshape(batch, self.context_channels, height, width)
        support = (weights * availability).sum(1).reshape(batch, 1, height, width)
        context = F.interpolate(context.to(x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
        support = F.interpolate(support, size=x.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.project(context).tanh()
        alpha = self.max_residual * self.gain_logit.sigmoid()
        return x + (alpha * receiving_fire * support).to(x.dtype) * delta


def occupancy_targets(logits, batch):
    """Rasterize post-augmentation boxes, retaining at least one cell per box."""
    size, _, height, width = logits.shape
    target = torch.zeros_like(logits, dtype=torch.float32)
    rows = torch.cat((batch["batch_idx"].reshape(-1, 1), batch["cls"].reshape(-1, 1),
                      batch["bboxes"].reshape(-1, 4)), 1).detach().cpu().tolist()
    for image_id, class_id, cx, cy, bw, bh in rows:
        image_id, class_id = int(image_id), int(class_id)
        if not 0 <= image_id < size or class_id not in (0, 1):
            raise ValueError("Guidance expects batch identities and 0=fire, 1=smoke")
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0, min(width - 1, math.floor((cx - bw / 2) * width)))
        y1 = max(0, min(height - 1, math.floor((cy - bh / 2) * height)))
        x2 = max(x1 + 1, min(width, math.ceil((cx + bw / 2) * width)))
        y2 = max(y1 + 1, min(height, math.ceil((cy + bh / 2) * height)))
        target[image_id, class_id, y1:y2, x1:x2] = 1
    # Fire-box occupancy wins in overlap; do not teach flame cells as smoke sources.
    target[:, 1] *= 1 - target[:, 0]
    return target


def guidance_loss(logits, batch):
    target = occupancy_targets(logits, batch)
    element = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    positive, negative = target, 1 - target
    axes = (0, 2, 3)
    pos_n, neg_n = positive.sum(axes), negative.sum(axes)
    pos_mean = (element * positive).sum(axes) / pos_n.clamp_min(1)
    neg_mean = (element * negative).sum(axes) / neg_n.clamp_min(1)
    present = (pos_n > 0).float() + (neg_n > 0).float()
    return ((pos_mean + neg_mean) / present.clamp_min(1)).mean()
