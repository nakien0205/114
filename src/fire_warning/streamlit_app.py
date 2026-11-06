"""Streamlit launch shim; run via python -m fire_warning.demo_app --mode streamlit."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fire_warning.demo_app import parse_args, streamlit_page

streamlit_page(parse_args())

