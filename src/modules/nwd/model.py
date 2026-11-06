"""Legacy import path; implementation lives in fire_detection.losses.nwd.model."""
from importlib import import_module as _import_module
_impl = _import_module("fire_detection.losses.nwd.model")
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
