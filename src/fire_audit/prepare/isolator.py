"""
Dataset isolator module for strict test set isolation.
Consolidates Home Fire Dataset frames exclusively into the held-out test split,
ensuring zero test leakage into training or validation partitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.fire_audit.config import (
    DEFAULT_EPSILON,
    HOME_FIRE_DIR_NAME,
    VALID_IMAGE_EXTENSIONS,
    DatasetScanResult,
    ImageRecord,
)
from src.fire_audit.audit.scanner import DatasetScanner, scan_home_fire_dataset
from src.fire_audit.audit.validator import validate_image_file


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class IsolationResult:
    """Result of isolating a dataset partition."""
    dataset_name: str
    root_path: Path
    test_images: List[Path] = field(default_factory=list)
    quarantined_images: List[Path] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=dict)
    sha256_hashes: Set[str] = field(default_factory=set)

    @property
    def total_valid_images(self) -> int:
        return len(self.test_images)

    @property
    def total_quarantined(self) -> int:
        return len(self.quarantined_images)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "root_path": str(self.root_path),
            "total_test_images": len(self.test_images),
            "total_quarantined": len(self.quarantined_images),
            "split_counts": self.split_counts,
        }


class DatasetIsolator:
    """
    Consolidates all images/labels from Home Fire Dataset into the test partition.
    Guarantees that 100% of valid Home Fire frames are assigned exclusively to test
    and 0% leak into training or validation sets.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        compute_hashes: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.compute_hashes = compute_hashes
        self.scanner = DatasetScanner(epsilon=epsilon)

    def isolate_home_fire(
        self,
        home_fire_root: Path,
        validate_images: bool = True,
    ) -> IsolationResult:
        """
        Isolate all Home Fire Dataset frames into the test partition.
        Consolidates subdirectories (train, val, test) into a single test collection.
        """
        result = IsolationResult(
            dataset_name="Home Fire Dataset",
            root_path=home_fire_root,
        )

        if not home_fire_root.exists():
            return result

        # Scan Home Fire Dataset
        scan_res = self.scanner.scan_home_fire(home_fire_root, validate_images=validate_images)

        # Categorize by original split source
        split_counts: Dict[str, int] = {}
        for rec in scan_res.records:
            split_src = rec.split_source or "unknown"
            split_counts[split_src] = split_counts.get(split_src, 0) + 1

            if rec.is_corrupt:
                result.quarantined_images.append(rec.image_path)
            else:
                result.test_images.append(rec.image_path)
                if self.compute_hashes:
                    try:
                        h = compute_file_sha256(rec.image_path)
                        result.sha256_hashes.add(h)
                    except Exception:
                        pass

        result.split_counts = split_counts
        return result

    def isolate_from_records(
        self,
        records: List[ImageRecord],
        dataset_name: str = "Home Fire Dataset",
    ) -> IsolationResult:
        """Isolate test images directly from pre-scanned ImageRecord objects."""
        result = IsolationResult(
            dataset_name=dataset_name,
            root_path=records[0].image_path.parent if records else Path("."),
        )

        split_counts: Dict[str, int] = {}
        for rec in records:
            split_src = rec.split_source or "unknown"
            split_counts[split_src] = split_counts.get(split_src, 0) + 1

            if rec.is_corrupt:
                result.quarantined_images.append(rec.image_path)
            else:
                result.test_images.append(rec.image_path)
                if self.compute_hashes:
                    try:
                        h = compute_file_sha256(rec.image_path)
                        result.sha256_hashes.add(h)
                    except Exception:
                        pass

        result.split_counts = split_counts
        return result


def isolate_home_fire_dataset(
    home_fire_root: Path,
    validate_images: bool = True,
    compute_hashes: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> IsolationResult:
    """Convenience function to isolate Home Fire Dataset into test partition."""
    isolator = DatasetIsolator(epsilon=epsilon, compute_hashes=compute_hashes)
    return isolator.isolate_home_fire(home_fire_root, validate_images=validate_images)
