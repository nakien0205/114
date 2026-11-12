from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, DEFAULT_CFG

from .model import NWDDetectionModel

class NWDDetectionTrainer(DetectionTrainer):
    """
    Custom Ultralytics trainer.

    It creates NWDDetectionModel instead of the default
    DetectionModel.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})

        self.nwd_weight = float(
            overrides.pop("nwd_weight", 0.25)
        )

        self.nwd_constant = float(
            overrides.pop("nwd_constant", 12.8)
        )

        super().__init__(
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks,
        )

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):
        """
        Build the custom model and load pretrained weights.
        """

        model = NWDDetectionModel(
            cfg=cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=(
                verbose and RANK == -1
            ),
            nwd_weight=self.nwd_weight,
            nwd_constant=self.nwd_constant,
        )

        if weights:
            model.load(weights)

        return model