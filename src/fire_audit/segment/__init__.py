"""
Automated Video Sequence Segmentation package for fire_audit.
"""

from .classifier import (
    compute_aspect_ratio_str,
    compute_dhash,
    compute_dhashes_parallel,
    get_image_dimensions,
    get_jpeg_dimensions,
    hamming_distance,
    is_transition_smooth,
    split_into_coherent_segments,
    verify_temporal_coherence,
)
from .engine import VideoSegmentationEngine
from .manifest_writer import (
    ManifestWriter,
    export_manifest_csv,
    export_manifest_json,
    export_manifests,
)


__all__ = [
    "VideoSegmentationEngine",
    "ManifestWriter",
    "export_manifest_json",
    "export_manifest_csv",
    "export_manifests",
    "get_jpeg_dimensions",
    "get_image_dimensions",
    "compute_aspect_ratio_str",
    "compute_dhash",
    "compute_dhashes_parallel",
    "hamming_distance",
    "is_transition_smooth",
    "verify_temporal_coherence",
    "split_into_coherent_segments",
]

