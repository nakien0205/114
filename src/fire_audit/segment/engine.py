"""
Engine module for automated video sequence segmentation.

Provides:
- VideoSegmentationEngine: orchestrates two-stage segmentation on image collections
  or raw dataset directories.
- Stage 1: Prefix & index grouping, fast header dimension scan, pre-filtering short runs (< 5 frames).
- Stage 2: Parallel dHash perceptual hashing, temporal lookahead bridge (k=2),
  and coherence verification to detect true video sequences.
- Structured output via SegmentationResult containing SegmentedFrame and VideoSequence dataclasses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    from src.fire_audit.config import (
        DEFAULT_EPSILON,
        VALID_IMAGE_EXTENSIONS,
        SegmentedFrame,
        SegmentationConfig,
        SegmentationResult,
        VideoSequence,
    )
    from src.fire_audit.audit.validator import validate_label_file
    from src.fire_audit.segment.classifier import (
        compute_aspect_ratio_str,
        compute_dhashes_parallel,
        get_image_dimensions,
        split_into_coherent_segments,
    )
except ImportError:
    from fire_audit.config import (
        DEFAULT_EPSILON,
        VALID_IMAGE_EXTENSIONS,
        SegmentedFrame,
        SegmentationConfig,
        SegmentationResult,
        VideoSequence,
    )
    from fire_audit.audit.validator import validate_label_file
    from fire_audit.segment.classifier import (
        compute_aspect_ratio_str,
        compute_dhashes_parallel,
        get_image_dimensions,
        split_into_coherent_segments,
    )


@dataclass
class _ImageEntry:
    path: Path
    prefix: str
    index: int
    sort_key: Tuple[str, int, str]


@dataclass
class _ContiguousRun:
    prefix: str
    dimensions: Tuple[int, int]
    entries: List[_ImageEntry]


class VideoSegmentationEngine:
    """
    Automated Video Sequence Segmentation Engine.

    Segments collections of images (such as FASDD_CV) into contiguous video clips
    and isolated static images using a high-throughput two-stage pipeline:
    1. Structural grouping (filename prefix, contiguous integer indexing, identical dimensions).
    2. Perceptual validation (multithreaded 64-bit dHash, k=2 lookahead bridge, temporal coherence).
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()

    @staticmethod
    def parse_filename(stem: str) -> Tuple[str, int]:
        """
        Parse filename stem into category prefix and numerical index.

        Examples:
        - 'bothFireAndSmoke_CV000000' -> ('bothFireAndSmoke_CV', 0)
        - 'fire_CV000123' -> ('fire_CV', 123)
        - 'clip1_0005' -> ('clip1_', 5)
        - 'sample' -> ('sample', -1)
        """
        m = re.match(r"^(.*?)(\d+)$", stem)
        if m:
            prefix = m.group(1)
            index = int(m.group(2))
            return prefix, index
        return stem, -1

    @staticmethod
    def clean_prefix(prefix: str) -> str:
        """
        Normalize category prefix for clean video_id naming.

        Examples:
        - 'bothFireAndSmoke_CV' -> 'bothFireAndSmoke'
        - 'fire_CV' -> 'fire'
        - 'smoke_CV' -> 'smoke'
        - 'clip1_' -> 'clip1'
        """
        cleaned = prefix
        if cleaned.endswith("_CV"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.rstrip("_-")
        return cleaned if cleaned else "video"

    def run(
        self,
        dataset_path: Union[Path, str],
        labels_dir: Optional[Union[Path, str]] = None,
    ) -> SegmentationResult:
        """Alias for segment_dataset to adhere to interface contract."""
        return self.segment_dataset(dataset_path=dataset_path, labels_dir=labels_dir)

    def segment_dataset(
        self,
        dataset_path: Union[Path, str],
        labels_dir: Optional[Union[Path, str]] = None,
    ) -> SegmentationResult:
        """
        Discover images in dataset directory and perform video sequence segmentation.

        Automatically resolves 'images/' and 'annotations/YOLO_CV/labels/' subfolders
        if present in dataset_path.
        """
        root = Path(dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")

        # Resolve images directory
        if (root / "images").is_dir():
            images_dir = root / "images"
        else:
            images_dir = root

        # Resolve labels directory if not explicitly provided
        resolved_labels_dir: Optional[Path] = None
        if labels_dir is not None:
            resolved_labels_dir = Path(labels_dir)
        elif (root / "annotations" / "YOLO_CV" / "labels").is_dir():
            resolved_labels_dir = root / "annotations" / "YOLO_CV" / "labels"
        elif (root / "labels").is_dir():
            resolved_labels_dir = root / "labels"
        elif (root.parent / "annotations" / "YOLO_CV" / "labels").is_dir():
            resolved_labels_dir = root.parent / "annotations" / "YOLO_CV" / "labels"
        elif (root.parent / "labels").is_dir():
            resolved_labels_dir = root.parent / "labels"

        # Discover all valid image files
        image_paths: List[Path] = []
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                image_paths.append(p)

        return self.segment_images(image_paths=image_paths, labels_dir=resolved_labels_dir)

    def segment_images(
        self,
        image_paths: Sequence[Union[Path, str]],
        labels_dir: Optional[Union[Path, str]] = None,
    ) -> SegmentationResult:
        """
        Execute two-stage video segmentation on an explicit sequence of image paths.

        Returns a complete SegmentationResult containing classified SegmentedFrame
        and VideoSequence objects.
        """
        paths = [Path(p) for p in image_paths if Path(p).is_file()]
        if not paths:
            return SegmentationResult()

        resolved_labels_dir = Path(labels_dir) if labels_dir else None

        # 1. Parse image entries and sort deterministically
        entries: List[_ImageEntry] = []
        for p in paths:
            prefix, idx = self.parse_filename(p.stem)
            sort_key = (prefix, idx if idx >= 0 else 0, p.name)
            entries.append(_ImageEntry(path=p, prefix=prefix, index=idx, sort_key=sort_key))

        entries.sort(key=lambda e: e.sort_key)

        # 2. Extract dimensions in parallel (using fast binary header parser)
        dimensions_map = self._scan_dimensions([e.path for e in entries])

        # 3. Parse label metadata in parallel (or single pass)
        labels_map = self._parse_labels_quick([e.path for e in entries], resolved_labels_dir)

        # 4. Stage 1: Group into contiguous (prefix, index, dimension) runs
        runs = self._group_into_contiguous_runs(entries, dimensions_map)

        # 5. Partition runs into candidate video runs (>= min_len) vs pre-filtered static runs (< min_len)
        min_len = self.config.min_sequence_length
        candidate_runs: List[_ContiguousRun] = []
        short_runs: List[_ContiguousRun] = []

        for r in runs:
            if len(r.entries) >= min_len and r.prefix != "neitherFireNorSmoke":
                candidate_runs.append(r)
            else:
                short_runs.append(r)

        # 6. Stage 2: Parallel dHash on candidate runs
        candidate_paths: List[Path] = [e.path for r in candidate_runs for e in r.entries]
        hashes_map = compute_dhashes_parallel(candidate_paths, max_workers=self.config.max_workers)

        # 7. Evaluate temporal coherence and lookahead bridge on candidate runs
        video_sequences: Dict[str, VideoSequence] = {}
        video_counters: Dict[str, int] = defaultdict(int)
        frame_records_map: Dict[Path, SegmentedFrame] = {}

        for run in candidate_runs:
            run_hashes = [hashes_map.get(str(e.path), 0) for e in run.entries]
            coherent_segments = split_into_coherent_segments(
                run_hashes,
                threshold=self.config.dhash_threshold,
                lookahead_k=self.config.lookahead_k,
                min_length=min_len,
                smooth_ratio_threshold=self.config.smooth_ratio_threshold,
            )

            # Track which indices in this run are accepted as video frames
            assigned_indices: set[int] = set()

            for start_idx, end_idx in coherent_segments:
                seg_entries = run.entries[start_idx:end_idx]
                seg_len = len(seg_entries)
                clean_pref = self.clean_prefix(run.prefix)
                video_counters[clean_pref] += 1
                vid_id = f"{clean_pref}_video_{video_counters[clean_pref]:04d}"

                seg_frames: List[SegmentedFrame] = []
                for f_idx, entry in enumerate(seg_entries):
                    w, h = dimensions_map.get(str(entry.path), (0, 0))
                    ar = compute_aspect_ratio_str(w, h)
                    fb, sb, is_neg, is_corr, err_msg = labels_map.get(str(entry.path), (0, 0, True, False, None))

                    frame = SegmentedFrame(
                        filename=entry.path.name,
                        path=entry.path,
                        category="video_frame",
                        video_id=vid_id,
                        frame_index=f_idx,
                        sequence_length=seg_len,
                        width=w,
                        height=h,
                        aspect_ratio=ar,
                        fire_boxes=fb,
                        smoke_boxes=sb,
                        is_negative=is_neg,
                        is_corrupt=is_corr or (w <= 0 or h <= 0),
                        error_message=err_msg,
                    )
                    seg_frames.append(frame)
                    frame_records_map[entry.path] = frame

                video_seq = VideoSequence(
                    video_id=vid_id,
                    category_prefix=run.prefix,
                    start_index=seg_entries[0].index,
                    end_index=seg_entries[-1].index,
                    total_frames=seg_len,
                    resolution=run.dimensions,
                    frames=seg_frames,
                )
                video_sequences[vid_id] = video_seq
                assigned_indices.update(range(start_idx, end_idx))

            # Unassigned frames in candidate run (e.g. rejected sub-segments) become static_image
            for idx, entry in enumerate(run.entries):
                if idx not in assigned_indices:
                    w, h = dimensions_map.get(str(entry.path), (0, 0))
                    ar = compute_aspect_ratio_str(w, h)
                    fb, sb, is_neg, is_corr, err_msg = labels_map.get(str(entry.path), (0, 0, True, False, None))

                    frame = SegmentedFrame(
                        filename=entry.path.name,
                        path=entry.path,
                        category="static_image",
                        video_id=None,
                        frame_index=None,
                        sequence_length=None,
                        width=w,
                        height=h,
                        aspect_ratio=ar,
                        fire_boxes=fb,
                        smoke_boxes=sb,
                        is_negative=is_neg,
                        is_corrupt=is_corr or (w <= 0 or h <= 0),
                        error_message=err_msg,
                    )
                    frame_records_map[entry.path] = frame

        # 8. Categorize all short runs (< min_len) as static_image
        for run in short_runs:
            for entry in run.entries:
                w, h = dimensions_map.get(str(entry.path), (0, 0))
                ar = compute_aspect_ratio_str(w, h)
                fb, sb, is_neg, is_corr, err_msg = labels_map.get(str(entry.path), (0, 0, True, False, None))

                frame = SegmentedFrame(
                    filename=entry.path.name,
                    path=entry.path,
                    category="static_image",
                    video_id=None,
                    frame_index=None,
                    sequence_length=None,
                    width=w,
                    height=h,
                    aspect_ratio=ar,
                    fire_boxes=fb,
                    smoke_boxes=sb,
                    is_negative=is_neg,
                    is_corrupt=is_corr or (w <= 0 or h <= 0),
                    error_message=err_msg,
                )
                frame_records_map[entry.path] = frame

        # 9. Assemble final records in sorted entry order
        all_records = [frame_records_map[e.path] for e in entries]

        video_count = sum(1 for r in all_records if r.category == "video_frame")
        static_count = sum(1 for r in all_records if r.category == "static_image")

        return SegmentationResult(
            total_images=len(all_records),
            video_frames_count=video_count,
            static_images_count=static_count,
            total_video_sequences=len(video_sequences),
            sequences=video_sequences,
            records=all_records,
        )

    def _scan_dimensions(
        self,
        paths: Sequence[Path],
    ) -> Dict[str, Tuple[int, int]]:
        """Extract dimensions for all image paths using thread pool."""
        if not paths:
            return {}

        num_workers = min(self.config.max_workers, len(paths)) if len(paths) > 0 else 1
        if num_workers > 1 and len(paths) > 1:
            chunksize = max(1, len(paths) // (num_workers * 4))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                dims = list(executor.map(get_image_dimensions, paths, chunksize=chunksize))
            return {str(p): d for p, d in zip(paths, dims)}
        else:
            return {str(p): get_image_dimensions(p) for p in paths}

    def _parse_labels_quick(
        self,
        image_paths: Sequence[Path],
        labels_dir: Optional[Path],
    ) -> Dict[str, Tuple[int, int, bool, bool, Optional[str]]]:
        """
        Fast label parser returning (fire_boxes, smoke_boxes, is_negative, is_corrupt, error_message) for each image.
        """
        if not labels_dir or not labels_dir.exists():
            return {str(p): (0, 0, True, False, None) for p in image_paths}

        def parse_single_label(p: Path) -> Tuple[int, int, bool, bool, Optional[str]]:
            lbl_file = labels_dir / f"{p.stem}.txt"
            if not lbl_file.exists():
                return (0, 0, True, False, None)
            try:
                val_res = validate_label_file(lbl_file, epsilon=DEFAULT_EPSILON)
                if not val_res.valid:
                    return (0, 0, False, True, "; ".join(val_res.errors))
                fire_count = sum(1 for b in val_res.boxes if b.class_id == 0)
                smoke_count = sum(1 for b in val_res.boxes if b.class_id == 1)
                return (fire_count, smoke_count, val_res.is_negative, False, None)
            except Exception as e:
                return (0, 0, True, True, str(e))

        num_workers = min(self.config.max_workers, len(image_paths)) if len(image_paths) > 0 else 1
        if num_workers > 1 and len(image_paths) > 1:
            chunksize = max(1, len(image_paths) // (num_workers * 4))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                label_results = list(executor.map(parse_single_label, image_paths, chunksize=chunksize))
            return {str(p): r for p, r in zip(image_paths, label_results)}
        else:
            return {str(p): parse_single_label(p) for p in image_paths}

    def _group_into_contiguous_runs(
        self,
        entries: Sequence[_ImageEntry],
        dimensions_map: Dict[str, Tuple[int, int]],
    ) -> List[_ContiguousRun]:
        """
        Group sorted image entries into contiguous runs where:
        1. Prefix matches.
        2. Numerical index is consecutive (index[k] == index[k-1] + 1).
        3. Dimensions are identical and valid (width > 0, height > 0).
        """
        if not entries:
            return []

        runs: List[_ContiguousRun] = []
        curr_entries: List[_ImageEntry] = [entries[0]]
        curr_dim = dimensions_map.get(str(entries[0].path), (0, 0))

        for entry in entries[1:]:
            prev_entry = curr_entries[-1]
            dim = dimensions_map.get(str(entry.path), (0, 0))

            same_prefix = entry.prefix == prev_entry.prefix
            contiguous_index = (
                entry.index >= 0
                and prev_entry.index >= 0
                and entry.index == prev_entry.index + 1
            )
            same_dimension = (dim == curr_dim) and (dim[0] > 0 and dim[1] > 0)

            if same_prefix and contiguous_index and same_dimension:
                curr_entries.append(entry)
            else:
                runs.append(_ContiguousRun(
                    prefix=curr_entries[0].prefix,
                    dimensions=curr_dim,
                    entries=curr_entries,
                ))
                curr_entries = [entry]
                curr_dim = dim

        if curr_entries:
            runs.append(_ContiguousRun(
                prefix=curr_entries[0].prefix,
                dimensions=curr_dim,
                entries=curr_entries,
            ))

        return runs
