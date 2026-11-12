"""Compatibility namespace for historical src.modules checkpoint imports."""
import sys
from pathlib import Path
_source = str(Path(__file__).resolve().parents[1])
if _source not in sys.path:
    sys.path.insert(0, _source)
