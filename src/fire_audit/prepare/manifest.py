"""
YOLO dataset manifest and split file generation module.
Generates data.yaml configuration and POSIX-compliant train.txt, val.txt, and test.txt manifests,
and integrates VideoSegmentationEngine and ManifestWriter for dual metadata exports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

try:
    from src.fire_audit.config import (
        CLASS_MAP,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        DEFAULT_EPSILON,
        FASDD_CV_DIR_NAME,
        HOME_FIRE_DIR_NAME,
        SegmentationResult,
    )
    from src.fire_audit.prepare.isolator import DatasetIsolator, IsolationResult
    from src.fire_audit.prepare.partitioner import SequencePartitioner, PartitionResult
    from src.fire_audit.segment.engine import VideoSegmentationEngine
    from src.fire_audit.segment.manifest_writer import ManifestWriter
except ImportError:
    from fire_audit.config import (
        CLASS_MAP,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        DEFAULT_EPSILON,
        FASDD_CV_DIR_NAME,
        HOME_FIRE_DIR_NAME,
        SegmentationResult,
    )
    from fire_audit.prepare.isolator import DatasetIsolator, IsolationResult
    from fire_audit.prepare.partitioner import SequencePartitioner, PartitionResult
    from fire_audit.segment.engine import VideoSegmentationEngine
    from fire_audit.segment.manifest_writer import ManifestWriter


@dataclass
class ManifestSummary:
    """Summary of generated manifests and split statistics."""
    data_yaml_path: Path
    splits_dir: Path
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_count: int = 0
    nc: int = 2
    names: Dict[int, str] = field(default_factory=lambda: {0: "fire", 1: "smoke"})
    relative_paths: bool = False
    manifest_json_path: Optional[Path] = None
    manifest_csv_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_yaml_path": str(self.data_yaml_path).replace("\\", "/"),
            "splits_dir": str(self.splits_dir).replace("\\", "/"),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "total_count": self.total_count,
            "nc": self.nc,
            "names": self.names,
            "relative_paths": self.relative_paths,
            "manifest_json_path": str(self.manifest_json_path).replace("\\", "/") if self.manifest_json_path else None,
            "manifest_csv_path": str(self.manifest_csv_path).replace("\\", "/") if self.manifest_csv_path else None,
        }


class ManifestGenerator:
    """
    Generates standardized YOLO format manifests (data.yaml and split text files).
    All written file paths conform to POSIX forward slash convention ('/').
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = Path("."),
        splits_dirname: str = "splits",
        nc: int = 2,
        names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.splits_dirname = splits_dirname
        self.splits_dir = self.output_dir / splits_dirname
        self.nc = nc
        self.names = names or {0: "fire", 1: "smoke"}

    def format_path(self, path: Path, relative_to: Optional[Path] = None) -> str:
        """Format a file path with POSIX forward slashes."""
        if relative_to is not None:
            try:
                rel = path.resolve().relative_to(relative_to.resolve())
                return rel.as_posix()
            except ValueError:
                pass
        return path.resolve().as_posix()

    def write_split_file(
        self,
        file_path: Path,
        image_paths: List[Path],
        relative_to: Optional[Path] = None,
    ) -> int:
        """Write a list of image paths to a split text file with POSIX slashes."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        formatted_lines = [self.format_path(p, relative_to=relative_to) for p in sorted(image_paths)]
        content = "\n".join(formatted_lines) + ("\n" if formatted_lines else "")
        file_path.write_text(content, encoding="utf-8")
        return len(formatted_lines)

    def write_data_yaml(
        self,
        yaml_path: Path,
        train_rel: str = "splits/train.txt",
        val_rel: str = "splits/val.txt",
        test_rel: str = "splits/test.txt",
        dataset_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Write data.yaml manifest conforming to YOLOv8/YOLOv5 standard."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        root_str = (dataset_root or self.output_dir).resolve().as_posix()

        manifest_data = {
            "path": root_str,
            "train": train_rel.replace("\\", "/"),
            "val": val_rel.replace("\\", "/"),
            "test": test_rel.replace("\\", "/"),
            "nc": self.nc,
            "names": self.names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest_data, f, sort_keys=False, default_flow_style=False)

        return manifest_data

    def generate(
        self,
        train_images: List[Path],
        val_images: List[Path],
        test_images: List[Path],
        relative_to: Optional[Path] = None,
    ) -> ManifestSummary:
        """
        Generate complete YOLO dataset structure:
        - output_dir/data.yaml
        - output_dir/splits/train.txt
        - output_dir/splits/val.txt
        - output_dir/splits/test.txt
        """
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        train_txt = self.splits_dir / "train.txt"
        val_txt = self.splits_dir / "val.txt"
        test_txt = self.splits_dir / "test.txt"

        n_train = self.write_split_file(train_txt, train_images, relative_to=relative_to)
        n_val = self.write_split_file(val_txt, val_images, relative_to=relative_to)
        n_test = self.write_split_file(test_txt, test_images, relative_to=relative_to)

        yaml_path = self.output_dir / "data.yaml"
        train_rel = f"{self.splits_dirname}/train.txt"
        val_rel = f"{self.splits_dirname}/val.txt"
        test_rel = f"{self.splits_dirname}/test.txt"

        self.write_data_yaml(
            yaml_path=yaml_path,
            train_rel=train_rel,
            val_rel=val_rel,
            test_rel=test_rel,
            dataset_root=self.output_dir,
        )

        return ManifestSummary(
            data_yaml_path=yaml_path,
            splits_dir=self.splits_dir,
            train_count=n_train,
            val_count=n_val,
            test_count=n_test,
            total_count=n_train + n_val + n_test,
            nc=self.nc,
            names=self.names,
            relative_paths=relative_to is not None,
        )


class DataPreparer:
    """
    End-to-end dataset preparation pipeline orchestrator:
    1. Isolates Home Fire Dataset into held-out test split.
    2. Runs VideoSegmentationEngine to classify FASDD_CV into contiguous video sequences and static images.
    3. Serializes structured metadata manifests (JSON + CSV) via ManifestWriter.
    4. Partitions FASDD_CV into leak-free sequence-aware train and validation splits.
    5. Emits data.yaml and splits/ text manifests.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
        output_dir: Union[str, Path] = Path("."),
        train_ratio: float = 0.8,
        seed: int = 42,
        epsilon: float = DEFAULT_EPSILON,
        validate_images: bool = True,
        compute_hashes: bool = False,
        use_segmentation: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.train_ratio = train_ratio
        self.seed = seed
        self.epsilon = epsilon
        self.validate_images = validate_images
        self.compute_hashes = compute_hashes
        self.use_segmentation = use_segmentation

        self.isolator = DatasetIsolator(epsilon=epsilon, compute_hashes=compute_hashes)
        self.partitioner = SequencePartitioner(
            train_ratio=train_ratio,
            random_seed=seed,
            epsilon=epsilon,
        )
        self.manifest_gen = ManifestGenerator(output_dir=self.output_dir)
        self.manifest_writer = ManifestWriter(dataset_name="FASDD_CV")
        self.segment_engine = VideoSegmentationEngine()
        self.last_segmentation_result: Optional[SegmentationResult] = None

    def prepare(
        self,
        home_fire_dir: Optional[Path] = None,
        fasdd_dir: Optional[Path] = None,
        export_manifests: bool = True,
    ) -> Tuple[IsolationResult, PartitionResult, ManifestSummary]:
        """
        Execute full preparation workflow.
        """
        home_root = home_fire_dir or (self.data_dir / HOME_FIRE_DIR_NAME)
        fasdd_root = fasdd_dir or (self.data_dir / FASDD_CV_DIR_NAME)

        if not home_root.exists() and (self.data_dir / "test").exists():
            home_root = self.data_dir
        if not fasdd_root.exists() and (self.data_dir / "images").exists():
            fasdd_root = self.data_dir

        # 1. Isolate Home Fire into Test
        iso_res = self.isolator.isolate_home_fire(
            home_root, validate_images=self.validate_images
        )

        # 2. Segment & Partition FASDD into Train and Val
        json_path: Optional[Path] = None
        csv_path: Optional[Path] = None

        if self.use_segmentation and fasdd_root.exists():
            try:
                seg_res = self.segment_engine.segment_dataset(fasdd_root)
                self.last_segmentation_result = seg_res

                if export_manifests and seg_res.records:
                    manifests_dir = self.output_dir / "manifests"
                    json_path, csv_path = self.manifest_writer.write_manifests(
                        seg_res, manifests_dir
                    )

                part_res = self.partitioner.partition_segmented_frames(
                    seg_res.records,
                    dataset_name="FASDD_CV",
                    root_path=fasdd_root,
                )
                part_res.segmentation_result = seg_res
            except Exception:
                # Fallback to standard scan partitioner
                part_res = self.partitioner.partition_fasdd_cv(
                    fasdd_root, validate_images=self.validate_images, use_segmentation=False
                )
        else:
            part_res = self.partitioner.partition_fasdd_cv(
                fasdd_root, validate_images=self.validate_images, use_segmentation=False
            )

        # 3. Generate YOLO Manifests
        manifest_summary = self.manifest_gen.generate(
            train_images=part_res.train_images,
            val_images=part_res.val_images,
            test_images=iso_res.test_images,
        )

        manifest_summary.manifest_json_path = json_path
        manifest_summary.manifest_csv_path = csv_path

        return iso_res, part_res, manifest_summary


def generate_yolo_manifests(
    train_images: List[Path],
    val_images: List[Path],
    test_images: List[Path],
    output_dir: Path = Path("."),
    splits_dirname: str = "splits",
    nc: int = 2,
    names: Optional[Dict[int, str]] = None,
) -> ManifestSummary:
    """Convenience function to generate YOLO manifests and split text lists."""
    gen = ManifestGenerator(
        output_dir=output_dir,
        splits_dirname=splits_dirname,
        nc=nc,
        names=names,
    )
    return gen.generate(train_images, val_images, test_images)


def prepare_pipeline_datasets(
    data_dir: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = Path("."),
    train_ratio: float = 0.8,
    seed: int = 42,
    validate_images: bool = True,
    compute_hashes: bool = False,
    use_segmentation: bool = True,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[IsolationResult, PartitionResult, ManifestSummary]:
    """Convenience function to run end-to-end dataset preparation."""
    preparer = DataPreparer(
        data_dir=data_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
        epsilon=epsilon,
        validate_images=validate_images,
        compute_hashes=compute_hashes,
        use_segmentation=use_segmentation,
    )
    return preparer.prepare()
