from ultralytics.utils.loss import v8DetectionLoss

from .bbox_loss import NWDBboxLoss


class NWDDetectionLoss(v8DetectionLoss):
    """
    YOLOv8 / YOLO11 detection loss with NWD.

    Original:
        v8DetectionLoss
            -> BboxLoss

    Custom:
        v8DetectionLoss
            -> NWDBboxLoss
    """

    def __init__(
        self,
        model,
        tal_topk=10,
        tal_topk2=None,
        nwd_weight=0.25,
        nwd_constant=12.8,
    ):
        # Initialize everything from Ultralytics.
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
        )

        # Replace only bbox loss.
        self.bbox_loss = NWDBboxLoss(
            reg_max=self.reg_max,
            nwd_weight=nwd_weight,
            nwd_constant=nwd_constant,
        ).to(self.device)
        