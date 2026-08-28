"""Dataset isolator for strict test-set isolation across arbitrary datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.fire_audit.config import (
    DEFAULT_EPSILON,
    ImageRecord,
)
from src.fire_audit.audit.scanner import DatasetScanner


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
    Collects all valid records from a dataset into a held-out test collection.
    Corrupt records are quarantined and optional SHA-256 hashes can be retained
    for later cross-split leakage checks.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        compute_hashes: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.compute_hashes = compute_hashes
        self.scanner = DatasetScanner(epsilon=epsilon)

    def isolate_dataset(
        self,
        dataset_root: Path,
        validate_images: bool = True,
        dataset_name: Optional[str] = None,
    ) -> IsolationResult:
        """
        Isolate all valid images from a dataset into the test partition.
        The scanner auto-detects flat and split-based dataset topologies.
        """
        dataset_root = Path(dataset_root).resolve()
        result = IsolationResult(
            dataset_name=dataset_name or dataset_root.name,
            root_path=dataset_root,
        )

        if not dataset_root.exists():
            return result

        scan_res = self.scanner.scan_dataset(
            dataset_root,
            dataset_name=dataset_name or dataset_root.name,
            validate_images=validate_images,
        )

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
        dataset_name: Optional[str] = None,
    ) -> IsolationResult:
        """Isolate test images directly from pre-scanned ImageRecord objects."""
        result = IsolationResult(
            dataset_name=dataset_name or (records[0].dataset if records and records[0].dataset else "dataset"),
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


def isolate_dataset(
    dataset_root: Path,
    validate_images: bool = True,
    compute_hashes: bool = False,
    epsilon: float = DEFAULT_EPSILON,
    dataset_name: Optional[str] = None,
) -> IsolationResult:
    """Convenience function to isolate any dataset into a test collection."""
    isolator = DatasetIsolator(epsilon=epsilon, compute_hashes=compute_hashes)
    return isolator.isolate_dataset(
        dataset_root,
        validate_images=validate_images,
        dataset_name=dataset_name,
    )
