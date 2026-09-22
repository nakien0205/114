from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel

from .detection_loss import WIoUDetectionLoss


class WIoUDetectionModel(DetectionModel):
    """
    DetectionModel with WIoU box regression.

    The network architecture itself is unchanged.

    Only init_criterion() is overridden.
    """

    def __init__(
        self,
        cfg=None,
        ch=3,
        nc=None,
        verbose=True,
        wiou_monotonous=False,
        wiou_alpha=1.9,
        wiou_delta=3.0,
        wiou_momentum=0.01,
    ):
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
        )

        self.wiou_monotonous = bool(
            wiou_monotonous
        )

        self.wiou_alpha = float(
            wiou_alpha
        )

        self.wiou_delta = float(
            wiou_delta
        )

        self.wiou_momentum = float(
            wiou_momentum
        )

    def init_criterion(self):
        """
        Return the custom WIoU detection criterion.
        """

        return WIoUDetectionLoss(
            self,
            wiou_monotonous=self.wiou_monotonous,
            wiou_alpha=self.wiou_alpha,
            wiou_delta=self.wiou_delta,
            wiou_momentum=self.wiou_momentum,
        )