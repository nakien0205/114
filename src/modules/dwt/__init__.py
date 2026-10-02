"""DWT backbones and Ultralytics parser registration."""

from __future__ import annotations

from .backbone import DWTBackboneL3, DWTBackboneL4, DWTBlock, WaveletFusion


def register_dwt_modules() -> None:
    """Make custom layers available to Ultralytics YAML model parsing.

    Ultralytics resolves YAML module names from ``ultralytics.nn.tasks`` globals.
    Registration is kept local to this package, so importing the package is all
    that is required before loading either DWT YAML configuration.
    """
    import ultralytics.nn.tasks as tasks

    tasks.DWTBlock = DWTBlock
    tasks.WaveletFusion = WaveletFusion
    tasks.DWTBackboneL3 = DWTBackboneL3
    tasks.DWTBackboneL4 = DWTBackboneL4


register_dwt_modules()

__all__ = [
    "DWTBlock",
    "WaveletFusion",
    "DWTBackboneL3",
    "DWTBackboneL4",
    "register_dwt_modules",
]
