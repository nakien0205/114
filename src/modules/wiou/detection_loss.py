from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss

from .bbox_loss import WIoUBboxLoss


class WIoUDetectionLoss(v8DetectionLoss):
    """
    Custom detection criterion.

    Parent:
        ultralytics.utils.loss.v8DetectionLoss

    Modification:
        BboxLoss -> WIoUBboxLoss

    No modification to:
        - TaskAlignedAssigner
        - classification BCE
        - target preprocessing
        - bbox decoding
        - DFL implementation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        wiou_monotonous: bool = False,
        wiou_alpha: float = 1.9,
        wiou_delta: float = 3.0,
        wiou_momentum: float = 0.01,
    ) -> None:
        super().__init__(
            model=model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
        )

        self.bbox_loss = WIoUBboxLoss(
            reg_max=self.reg_max,
            wiou_monotonous=wiou_monotonous,
            wiou_alpha=wiou_alpha,
            wiou_delta=wiou_delta,
            wiou_momentum=wiou_momentum,
        ).to(self.device)