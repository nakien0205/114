from __future__ import annotations

from typing import Any


from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

from .model import WIoUDetectionModel


class WIoUDetectionTrainer(DetectionTrainer):
    """
    DetectionTrainer that builds WIoUDetectionModel.

    The normal Ultralytics training pipeline remains unchanged.

    Only:
        get_model()
    is overridden.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
    ):
        overrides = dict(overrides or {})

        # ----------------------------------------------------------
        # WIoU hyperparameters
        #
        # Pop them out before passing overrides to Ultralytics so
        # unknown custom keys do not leak into standard configuration.
        # ----------------------------------------------------------
        self.wiou_monotonous = self._pop_bool(
            overrides,
            "wiou_monotonous",
            False,
        )

        self.wiou_alpha = self._pop_float(
            overrides,
            "wiou_alpha",
            1.9,
        )

        self.wiou_delta = self._pop_float(
            overrides,
            "wiou_delta",
            3.0,
        )

        self.wiou_momentum = self._pop_float(
            overrides,
            "wiou_momentum",
            0.01,
        )

        self._validate_wiou_config()

        super().__init__(
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks,
        )

    @staticmethod
    def _pop_float(
        overrides: dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        value = overrides.pop(key, default)

        if value is None:
            return float(default)

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must be a float, got {value!r}"
            ) from exc

    @staticmethod
    def _pop_bool(
        overrides: dict[str, Any],
        key: str,
        default: bool,
    ) -> bool:
        value = overrides.pop(key, default)

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {
                "true",
                "1",
                "yes",
                "y",
            }:
                return True

            if normalized in {
                "false",
                "0",
                "no",
                "n",
            }:
                return False

        raise ValueError(
            f"{key} must be a boolean, got {value!r}"
        )

    def _validate_wiou_config(self) -> None:
        """Validate WIoU hyperparameters."""

        if self.wiou_alpha <= 0:
            raise ValueError(
                "wiou_alpha must be > 0, "
                f"got {self.wiou_alpha}"
            )

        if self.wiou_delta <= 0:
            raise ValueError(
                "wiou_delta must be > 0, "
                f"got {self.wiou_delta}"
            )

        if not 0.0 <= self.wiou_momentum <= 1.0:
            raise ValueError(
                "wiou_momentum must be in [0, 1], "
                f"got {self.wiou_momentum}"
            )

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):
        """
        Build WIoUDetectionModel and optionally load pretrained weights.
        """

        model = WIoUDetectionModel(
            cfg=cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=(
                verbose
                and RANK == -1
            ),
            wiou_monotonous=self.wiou_monotonous,
            wiou_alpha=self.wiou_alpha,
            wiou_delta=self.wiou_delta,
            wiou_momentum=self.wiou_momentum,
        )

        if weights:
            model.load(weights)

        return model

    def get_model_name(self):
        """
        Return a descriptive custom model name when available.
        """

        return "WIoUDetectionModel"