"""Repository-local YOLO26n-P2 DWT construction and trainer integration.

Use DWTYOLO, not plain YOLO, for the custom YAML. Custom layer tags deliberately
fail in the stock parser so they cannot silently construct a plain backbone.
Shared Ultralytics source and its module registry are never modified.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import RANK, ROOT, YAML

from .backbone import ConvThenHaar

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs" / "models" / "yolo26n-p2-dwt-insert.yaml"
INSERTIONS = (3, 5)


def _prepare_config(cfg: str | Path | dict) -> tuple[dict, dict]:
    """Validate the approved graph, resolving composite tags for base parsing."""
    custom = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
    resolved = deepcopy(custom)
    positions = tuple(index for index, layer in enumerate(custom.get("backbone", []))
                      if layer[2] == "ConvThenHaar")
    if positions != INSERTIONS:
        raise ValueError(f"Approved DWT insertions must be exactly after Conv layers {INSERTIONS}")
    for index in positions:
        resolved["backbone"][index][2] = "Conv"
    reference = YAML.load(ROOT / "cfg" / "models" / "26" / "yolo26-p2.yaml")
    if resolved["backbone"] != reference["backbone"] or resolved.get("head") != reference["head"]:
        raise ValueError("Expected the otherwise unchanged official YOLO26-P2 backbone and neck/head")
    if (resolved.get("scale") != "n" or resolved.get("scales", {}).get("n") != reference["scales"]["n"]
            or resolved.get("end2end") is not True or resolved.get("reg_max") != reference["reg_max"]):
        raise ValueError("Expected YOLO26 nano scaling and the original end-to-end head settings")
    return custom, resolved


class DWTDetectionModel(DetectionModel):
    """Build the original detector first, then append the approved residuals."""

    def __init__(self, cfg=DEFAULT_CONFIG, ch=3, nc=None, verbose=True):
        custom, resolved = _prepare_config(cfg)
        super().__init__(resolved, ch=ch, nc=nc, verbose=False)
        for index in INSERTIONS:
            self.model[index] = ConvThenHaar(self.model[index])
        # Preserve custom tags in the model's config so the trainer rebuilds
        # the same architecture, including after saving/resuming a checkpoint.
        for index in INSERTIONS:
            self.yaml["backbone"][index][2] = custom["backbone"][index][2]
        if self.yaml["nc"] == 2:
            self.names = {0: "fire", 1: "smoke"}
        if verbose:
            self.info()


class DWTTrainer(DetectionTrainer):
    """Rebuild DWT additions when Ultralytics prepares a later training run."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DWTDetectionModel(cfg if cfg is not None else DEFAULT_CONFIG,
                                  nc=self.data["nc"], ch=self.data["channels"],
                                  verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights, verbose=verbose)
        return model


class DWTYOLO(YOLO):
    """YOLO API using the custom model and trainer, with the standard predictor."""

    def __init__(self, model=DEFAULT_CONFIG, task="detect", verbose=False):
        if task != "detect":
            raise ValueError("This DWT implementation supports detection only")
        super().__init__(model=model, task=task, verbose=verbose)
        if not isinstance(self.model, DWTDetectionModel):
            raise ValueError("Build from the DWT YAML, then .load(base_weights); or load a DWT checkpoint")

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["detect"]["model"] = DWTDetectionModel
        mapping["detect"]["trainer"] = DWTTrainer
        return mapping
