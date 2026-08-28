"""Dataset discovery and validation engine for arbitrary image datasets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from src.fire_audit.config import (
    DEFAULT_EPSILON,
    VALID_IMAGE_EXTENSIONS,
    DatasetScanResult,
    ImageRecord,
)
from src.fire_audit.audit.validator import (
    validate_image_file,
    validate_label_file,
)


class DatasetScanner:
    """Discovers, pairs, and validates image-label pairs across datasets."""

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        image_extensions: Optional[Set[str]] = None,
        max_workers: int = 32,
    ) -> None:
        self.epsilon = epsilon
        self.image_extensions = image_extensions or VALID_IMAGE_EXTENSIONS
        self.max_workers = max_workers

    def is_image_file(self, path: Path) -> bool:
        """Check whether a file matches accepted image extensions."""
        return path.is_file() and path.suffix.lower() in self.image_extensions

    def extract_sequence_id(self, filename: str) -> str:
        """
        Extract a stable sequence identifier from a frame filename.
        Numeric suffixes are grouped into blocks to preserve local continuity.
        """
        stem = Path(filename).stem
        if "_CV" in stem:
            prefix = stem.split("_CV")[0]
            # Group into sequential blocks if index is available
            idx_part = stem.split("_CV")[-1]
            try:
                # Group by blocks of 50 contiguous frames to simulate scene/video continuity
                num_idx = int(idx_part)
                block_id = num_idx // 50
                return f"{prefix}_block_{block_id:04d}"
            except ValueError:
                return prefix
        return stem

    def scan_split_dataset(
        self,
        root_path: Path,
        validate_images: bool = True,
        dataset_name: str = "dataset",
    ) -> DatasetScanResult:
        """
        Scan dataset structure:
        root/
          train/images, train/labels
          val/images, val/labels
          test/images, test/labels
        or flat layout.
        """
        result = DatasetScanResult(dataset_name=dataset_name, root_path=root_path)

        if not root_path.exists():
            return result

        # Check for standard split folders (train, val, test)
        splits = ["train", "val", "test"]
        split_dirs = [root_path / s for s in splits if (root_path / s).is_dir()]

        raw_pairs: List[Tuple[Path, Optional[Path], str, str]] = []

        if split_dirs:
            for s_dir in split_dirs:
                split_name = s_dir.name
                img_dir = s_dir / "images" if (s_dir / "images").is_dir() else s_dir
                lbl_dir = s_dir / "labels" if (s_dir / "labels").is_dir() else s_dir

                for entry in sorted(img_dir.iterdir()):
                    if self.is_image_file(entry):
                        lbl_file = lbl_dir / f"{entry.stem}.txt"
                        label_path = lbl_file if lbl_file.exists() else None
                        seq_id = f"dataset_{split_name}_{entry.stem.split('_')[0]}"
                        raw_pairs.append((entry, label_path, split_name, seq_id))
        else:
            # Fallback: scan any images under root
            for root_dir, _, files in os.walk(root_path):
                rpath = Path(root_dir)
                for f in sorted(files):
                    p = rpath / f
                    if self.is_image_file(p):
                        # Look for matching label in parallel 'labels' or same directory
                        candidate_lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
                        if not candidate_lbl.exists():
                            candidate_lbl = p.parent / f"{p.stem}.txt"
                        label_path = candidate_lbl if candidate_lbl.exists() else None
                        raw_pairs.append((p, label_path, "raw", f"dataset_{p.stem}"))

        self._process_records(raw_pairs, dataset_name, result, validate_images)
        return result

    def scan_flat_dataset(
        self,
        root_path: Path,
        validate_images: bool = True,
        dataset_name: str = "dataset",
    ) -> DatasetScanResult:
        """
        Scan a flat dataset structure:
        root/
          images/
          annotations/YOLO_CV/labels/ (or annotations/YOLO_CV/ or labels/)
        """
        result = DatasetScanResult(dataset_name=dataset_name, root_path=root_path)

        if not root_path.exists():
            return result

        img_dir = root_path / "images" if (root_path / "images").is_dir() else root_path

        # Locate YOLO labels directory
        yolo_lbl_candidates = [
            root_path / "annotations" / "YOLO_CV" / "labels",
            root_path / "annotations" / "YOLO_CV",
            root_path / "labels",
            img_dir.parent / "labels",
        ]
        lbl_dir = None
        for cand in yolo_lbl_candidates:
            if cand.is_dir():
                lbl_dir = cand
                break

        raw_pairs: List[Tuple[Path, Optional[Path], str, str]] = []

        if img_dir.is_dir():
            for entry in sorted(img_dir.iterdir()):
                if self.is_image_file(entry):
                    label_path = None
                    if lbl_dir:
                        candidate_lbl = lbl_dir / f"{entry.stem}.txt"
                        if candidate_lbl.exists():
                            label_path = candidate_lbl

                    seq_id = self.extract_sequence_id(entry.name)
                    raw_pairs.append((entry, label_path, "raw", seq_id))

        self._process_records(raw_pairs, dataset_name, result, validate_images)
        return result

    def scan_dataset(
        self,
        root_path: Path,
        dataset_name: Optional[str] = None,
        validate_images: bool = True,
    ) -> DatasetScanResult:
        """
        Auto-detect dataset topology and scan accordingly.
        """
        name = dataset_name or root_path.name

        # Check directory structure
        if (root_path / "annotations" / "YOLO_CV").exists():
            return self.scan_flat_dataset(root_path, validate_images=validate_images, dataset_name=name)
        if any((root_path / split / "images").is_dir() for split in ("train", "valid", "val", "test")):
            return self.scan_split_dataset(root_path, validate_images=validate_images, dataset_name=name)

        if (root_path / "images").is_dir():
            return self.scan_flat_dataset(root_path, validate_images=validate_images, dataset_name=name)

        # Generic scan
        result = DatasetScanResult(dataset_name=name, root_path=root_path)
        raw_pairs: List[Tuple[Path, Optional[Path], str, str]] = []

        for root_dir, _, files in os.walk(root_path):
            rpath = Path(root_dir)
            for f in sorted(files):
                p = rpath / f
                if self.is_image_file(p):
                    lbl_candidate = p.parent.parent / "labels" / f"{p.stem}.txt"
                    if not lbl_candidate.exists():
                        lbl_candidate = p.parent / f"{p.stem}.txt"
                    label_path = lbl_candidate if lbl_candidate.exists() else None
                    raw_pairs.append((p, label_path, "raw", p.stem))

        self._process_records(raw_pairs, name, result, validate_images)
        return result

    def _validate_single_entry(
        self,
        entry: Tuple[Path, Optional[Path], str, str],
        dataset_name: str,
        validate_images: bool,
    ) -> Tuple[ImageRecord, Optional[Path], Optional[Path], Optional[Path]]:
        """Validate a single image-label entry and return (record, missing_label, corrupt_label, corrupt_image)."""
        img_path, lbl_path, split_source, seq_id = entry
        rec = ImageRecord(
            image_path=img_path,
            label_path=lbl_path,
            dataset=dataset_name,
            split_source=split_source,
            sequence_id=seq_id,
        )
        missing_lbl: Optional[Path] = None
        corrupt_lbl: Optional[Path] = None
        corrupt_img: Optional[Path] = None

        # Check label
        if lbl_path is None:
            missing_lbl = img_path
            rec.is_negative = False
        else:
            lbl_val = validate_label_file(lbl_path, epsilon=self.epsilon)
            if not lbl_val.valid:
                corrupt_lbl = lbl_path
                rec.is_corrupt = True
                rec.error_message = "; ".join(lbl_val.errors)
            else:
                rec.boxes = lbl_val.boxes
                rec.is_negative = lbl_val.is_negative

        # Check image
        if validate_images:
            img_valid, img_err, w, h = validate_image_file(img_path)
            if not img_valid:
                corrupt_img = img_path
                rec.is_corrupt = True
                rec.error_message = (
                    f"{rec.error_message}; {img_err}" if rec.error_message else img_err
                )
            else:
                rec.width = w
                rec.height = h

        return rec, missing_lbl, corrupt_lbl, corrupt_img

    def _process_records(
        self,
        raw_pairs: List[Tuple[Path, Optional[Path], str, str]],
        dataset_name: str,
        result: DatasetScanResult,
        validate_images: bool,
    ) -> None:
        """Process and validate paired image-label entries in parallel using ThreadPoolExecutor."""
        if not raw_pairs:
            return

        workers = self.max_workers if self.max_workers > 0 else 32
        if len(raw_pairs) > 1 and workers > 1:
            chunksize = max(1, len(raw_pairs) // (workers * 4))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                processed_results = list(
                    executor.map(
                        lambda p: self._validate_single_entry(p, dataset_name, validate_images),
                        raw_pairs,
                        chunksize=chunksize,
                    )
                )
        else:
            processed_results = [
                self._validate_single_entry(p, dataset_name, validate_images)
                for p in raw_pairs
            ]

        for rec, missing_lbl, corrupt_lbl, corrupt_img in processed_results:
            if missing_lbl:
                result.missing_labels.append(missing_lbl)
            if corrupt_lbl:
                result.corrupt_labels.append(corrupt_lbl)
            if corrupt_img:
                result.corrupt_images.append(corrupt_img)
            result.records.append(rec)


def scan_dataset(
    root_path: Path,
    dataset_name: Optional[str] = None,
    validate_images: bool = True,
    epsilon: float = DEFAULT_EPSILON,
    max_workers: int = 32,
) -> DatasetScanResult:
    """Convenience function to scan any supported dataset."""
    scanner = DatasetScanner(epsilon=epsilon, max_workers=max_workers)
    return scanner.scan_dataset(root_path, dataset_name=dataset_name, validate_images=validate_images)
