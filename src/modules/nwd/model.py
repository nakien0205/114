from ultralytics.utils.loss import E2ELoss
from ultralytics.nn.tasks import DetectionModel

from .detection_loss import NWDDetectionLoss


class NWDDetectionModel(DetectionModel):
    """
    DetectionModel using NWDDetectionLoss.
    """
    def __init__(
        self,
        cfg=None,
        ch=3,
        nc=None,
        verbose=True,
        nwd_weight=0.25,
        nwd_constant=12.8,
    ):
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
        )

        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)

    def init_criterion(self):
        """
        Replace default loss with NWDDetectionLoss.
        Supports both end-to-end (e.g. YOLO26) and standard YOLO models.
        """
        if not hasattr(self, "args") or self.args is None:
            self.args = {}

        def build_loss(model, tal_topk=10, tal_topk2=None):
            return NWDDetectionLoss(
                model,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                nwd_weight=self.nwd_weight,
                nwd_constant=self.nwd_constant,
            )

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=build_loss)

        tal_topk = self.args.get("tal_topk", 10) if isinstance(self.args, dict) else getattr(self.args, "tal_topk", 10)
        return build_loss(self, tal_topk=tal_topk)