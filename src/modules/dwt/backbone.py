"""Legacy import path; implementation lives in fire_detection.wavelet.backbone."""
from importlib import import_module as _import_module
_impl = _import_module("fire_detection.wavelet.backbone")
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
