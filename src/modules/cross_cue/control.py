"""Legacy import path; implementation lives in fire_detection.attention.cross_cue.control."""
from importlib import import_module as _import_module
_impl = _import_module("fire_detection.attention.cross_cue.control")
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
