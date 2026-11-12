"""Same-parameter P2 residual control: all class-cue inputs are always zero."""
from __future__ import annotations

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from .head import CrossCueDetect, P2CrossCueGate


class ResidualOnlyP2Gate(P2CrossCueGate):
    def __init__(self, channels: int) -> None:
        # Exactly the full gate's parameter/buffer construction and RNG draws.
        super().__init__(channels, "directional")

    def forward(self, p2: torch.Tensor, p3_logits: torch.Tensor) -> torch.Tensor:
        if p3_logits.shape[1] != 2:
            raise ValueError("The paired control requires two-class P3 logits")
        zeros = p2.new_zeros((p2.shape[0], 1, *p2.shape[-2:]))
        inputs = torch.cat((p2, zeros, zeros), dim=1)
        fire_weight = 2 * torch.sigmoid(self.fire_gate(inputs)) - 1
        smoke_weight = 2 * torch.sigmoid(self.smoke_gate(inputs)) - 1
        return p2 + fire_weight * self.fire_delta(p2) + smoke_weight * self.smoke_delta(p2)


class ResidualOnlyDetect(CrossCueDetect):
    def __init__(self, nc: int, reg_max: int, end2end: bool, ch: tuple[int, ...]) -> None:
        Detect.__init__(self, nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.p2_gate = ResidualOnlyP2Gate(ch[0])
    # Inherit CrossCueDetect.forward, including its extra P3 classifier call.
    # This keeps that classifier's training BatchNorm call pattern matched.


def attach_residual_control(model: DetectionModel) -> DetectionModel:
    base = model.model[-1]
    if not isinstance(base, Detect) or isinstance(base, CrossCueDetect) or base.nl != 4:
        raise ValueError("Expected the original four-scale Detect head")
    channels = tuple(branch[0].conv.in_channels for branch in base.cv2)
    head = ResidualOnlyDetect(base.nc, base.reg_max, base.end2end, channels)
    head.load_state_dict(base.state_dict(), strict=False)
    head.stride = base.stride
    for name in ("i", "f", "type", "np"):
        setattr(head, name, getattr(base, name))
    model.model[-1] = head
    model.stride = head.stride
    return model


class ResidualOnlyTrainer(DetectionTrainer):
    expected_initial_state_sha256 = None

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        attach_residual_control(model)
        if weights is not None:
            model.load(weights)
        if self.expected_initial_state_sha256 is not None:
            import hashlib
            digest = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            if digest.hexdigest() != self.expected_initial_state_sha256:
                raise RuntimeError("Actual trainer initialization differs from paired preflight")
            model.control_initial_state_sha256 = digest.hexdigest()
        return model
