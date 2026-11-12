"""Class-conditioned P2 gate using already supervised P3 class logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK


VARIANTS = ("local", "symmetric", "directional", "reversed")


class P2CrossCueGate(nn.Module):
    """Residual fire/smoke gates on P2, conditioned on P3 class evidence.

    The P3 class logits are produced by the existing detection classifier, whose
    ordinary detection loss already trains it.  They are detached here so the
    gate cannot change that classifier through its own shortcut.  Class IDs 0
    and 1 are fire and smoke for the Home Fire experiment.
    """

    def __init__(self, channels: int, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown cross-cue variant: {variant}")
        self.variant = variant
        hidden = max(channels // 4, 8)
        self.fire_gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1)
        )
        self.smoke_gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1)
        )
        self.fire_delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.smoke_delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        # A zero residual gives the new model exactly the baseline prediction
        # before any optimizer step, even with fire-only or smoke-only images.
        for gate in (self.fire_gate, self.smoke_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

        above = torch.zeros((1, 1, 5, 5))
        above[:, :, :2, :] = 1 / 10
        below = torch.zeros((1, 1, 5, 5))
        below[:, :, 3:, :] = 1 / 10
        symmetric = torch.full((1, 1, 5, 5), 1 / 25)
        self.register_buffer("above_kernel", above, persistent=False)
        self.register_buffer("below_kernel", below, persistent=False)
        self.register_buffer("symmetric_kernel", symmetric, persistent=False)

    def _context(self, fire: torch.Tensor, smoke: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "local":
            return smoke, fire
        if self.variant == "symmetric":
            return (
                F.conv2d(smoke, self.symmetric_kernel, padding=2),
                F.conv2d(fire, self.symmetric_kernel, padding=2),
            )
        if self.variant == "directional":
            return (
                F.conv2d(smoke, self.above_kernel, padding=2),
                F.conv2d(fire, self.below_kernel, padding=2),
            )
        return (
            F.conv2d(smoke, self.below_kernel, padding=2),
            F.conv2d(fire, self.above_kernel, padding=2),
        )

    def forward(self, p2: torch.Tensor, p3_logits: torch.Tensor) -> torch.Tensor:
        if p3_logits.shape[1] != 2:
            raise ValueError("P2 cross-cue gate requires exactly two classes: fire and smoke")
        cues = F.interpolate(
            p3_logits.detach().sigmoid(), size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        fire, smoke = cues[:, :1], cues[:, 1:2]
        smoke_for_fire, fire_for_smoke = self._context(fire, smoke)
        fire_weight = 2 * torch.sigmoid(self.fire_gate(torch.cat((p2, fire, smoke_for_fire), dim=1))) - 1
        smoke_weight = 2 * torch.sigmoid(self.smoke_gate(torch.cat((p2, smoke, fire_for_smoke), dim=1))) - 1
        return p2 + fire_weight * self.fire_delta(p2) + smoke_weight * self.smoke_delta(p2)


class CrossCueDetect(Detect):
    """Unchanged YOLO Detect head with a P2-only pre-head attention gate."""

    def __init__(self, nc: int, reg_max: int, end2end: bool, ch: tuple[int, ...], variant: str) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.p2_gate = P2CrossCueGate(ch[0], variant)

    def forward(self, x: list[torch.Tensor]):
        # P3 class scores are already part of the supervised head. Only the
        # P2 input changes; the P3/P4/P5 feature tensors pass through intact.
        # The one-to-one classifier is retained when Ultralytics fuses the
        # checkpoint for inference; cv3 (one-to-many) is removed there.
        p3_logits = self.one2one_cv3[1](x[1])
        x = [self.p2_gate(x[0], p3_logits), *x[1:]]
        return super().forward(x)


def attach_cross_cue(model: DetectionModel, variant: str) -> DetectionModel:
    """Replace only the terminal Detect module after YOLO has built the graph."""
    base = model.model[-1]
    if not isinstance(base, Detect) or isinstance(base, CrossCueDetect) or base.nl != 4:
        raise ValueError("Expected an unmodified four-scale YOLO26n-P2 Detect head")
    channels = tuple(branch[0].conv.in_channels for branch in base.cv2)
    head = CrossCueDetect(base.nc, base.reg_max, base.end2end, channels, variant)
    head.load_state_dict(base.state_dict(), strict=False)
    head.stride = base.stride
    for name in ("i", "f", "type", "np"):
        setattr(head, name, getattr(base, name))
    model.model[-1] = head
    model.stride = head.stride
    return model


class CrossCueTrainer(DetectionTrainer):
    """Build a gated model while preserving the standard detector and loss."""

    variant = "directional"

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        attach_cross_cue(model, self.variant)
        if weights is not None:
            model.load(weights)
        return model
