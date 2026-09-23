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
        Replace default v8DetectionLoss
        with NWDDetectionLoss.
        """

        return NWDDetectionLoss(
            self,
            tal_topk=self.args.get(
                "tal_topk",
                10,
            ),
            nwd_weight=self.nwd_weight,
            nwd_constant=self.nwd_constant,
        )