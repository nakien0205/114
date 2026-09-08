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
        DEFAULT_EPSILON,
        get_configured_data_root,
        MANIFEST_DIR_NAME,
        SPLITS_DIR_NAME,
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
        DEFAULT_EPSILON,
        get_configured_data_root,
        MANIFEST_DIR_NAME,
        SPLITS_DIR_NAME,
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
    dataset_summaries: Dict[str, "ManifestSummary"] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        def serialize_summary(summary: "ManifestSummary") -> Dict[str, Any]:
            # The primary summary can be present in dataset_summaries for
            # convenient CLI iteration; avoid recursively serializing itself.
            if summary is self:
                return {
                    "data_yaml_path": str(summary.data_yaml_path).replace("\\", "/"),
                    "splits_dir": str(summary.splits_dir).replace("\\", "/"),
                    "train_count": summary.train_count,
                    "val_count": summary.val_count,
                    "test_count": summary.test_count,
                    "total_count": summary.total_count,
                    "nc": summary.nc,
                    "names": summary.names,
                    "relative_paths": summary.relative_paths,
                    "manifest_json_path": str(summary.manifest_json_path).replace("\\", "/") if summary.manifest_json_path else None,
                    "manifest_csv_path": str(summary.manifest_csv_path).replace("\\", "/") if summary.manifest_csv_path else None,
                    "dataset_summaries": {},
                }
            return summary.to_dict()

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
            "dataset_summaries": {
                name: serialize_summary(summary)
                for name, summary in self.dataset_summaries.items()
            },
        }


class ManifestGenerator:
    """
    Generates standardized YOLO format manifests (data.yaml and split text files).
    All written file paths conform to POSIX forward slash convention ('/').

    ``output_dir`` is one dataset root; generated files are placed in
    ``output_dir/manifest`` and ``output_dir/splits``.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = Path("."),
        splits_dirname: str = SPLITS_DIR_NAME,
        nc: int = 2,
        names: Optional[Dict[int, str]] = None,
        manifest_dirname: str = MANIFEST_DIR_NAME,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.splits_dirname = splits_dirname
        self.splits_dir = self.output_dir / splits_dirname
        self.manifest_dirname = manifest_dirname
        self.manifest_dir = self.output_dir / manifest_dirname
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
        - output_dir/manifest/data.yaml
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

        yaml_path = self.manifest_dir / "data.yaml"
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


def dataset_artifact_root(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path] = Path("."),
    data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve an output root dedicated to one dataset.

    With the default output location, artifacts stay next to the source
    dataset. A custom output directory is treated as a parent and receives a
    child named after the dataset, preserving isolation between datasets.
    """
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    current_dir = Path.cwd().resolve()
    data_root = Path(data_dir).resolve() if data_dir is not None else None

    if output_dir in {
        current_dir,
        dataset_root,
        dataset_root.parent,
    } or (data_root is not None and output_dir == data_root):
        return dataset_root
    return output_dir / dataset_root.name


def resolve_dataset_roots(
    data_dir: Union[str, Path],
    holdout_dir: Optional[Union[str, Path]] = None,
    partition_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Resolve two dataset roots from a parent or direct dataset path."""
    data_root = Path(data_dir).resolve()

    if holdout_dir is not None and partition_dir is not None:
        return Path(holdout_dir).resolve(), Path(partition_dir).resolve()

    if data_root.exists() and (
        (data_root / "images").is_dir()
        or any((data_root / name).is_dir() for name in ("train", "valid", "val", "test"))
    ):
        return (
            Path(holdout_dir).resolve() if holdout_dir is not None else data_root,
            Path(partition_dir).resolve() if partition_dir is not None else data_root,
        )

    child_dirs = [p for p in data_root.iterdir() if p.is_dir()] if data_root.exists() else []
    unassigned = [p for p in child_dirs if p.name.lower() not in {MANIFEST_DIR_NAME, SPLITS_DIR_NAME}]

    split_candidates = [
        p for p in unassigned
        if any((p / name / "images").is_dir() for name in ("train", "valid", "val", "test"))
    ]
    flat_candidates = [p for p in unassigned if (p / "images").is_dir()]
    ordered = []
    for candidate in split_candidates + flat_candidates + sorted(unassigned):
        if candidate not in ordered:
            ordered.append(candidate)

    if len(ordered) >= 2:
        return (
            Path(holdout_dir).resolve() if holdout_dir is not None else ordered[0],
            Path(partition_dir).resolve() if partition_dir is not None else ordered[1],
        )
    if len(ordered) == 1:
        only = ordered[0]
        return (
            Path(holdout_dir).resolve() if holdout_dir is not None else only,
            Path(partition_dir).resolve() if partition_dir is not None else only,
        )
    return (
        Path(holdout_dir).resolve() if holdout_dir is not None else data_root,
        Path(partition_dir).resolve() if partition_dir is not None else data_root,
    )

class DataPreparer:
    """
    End-to-end dataset preparation pipeline orchestrator:
    1. Isolates one dataset into a held-out test split.
    2. Runs VideoSegmentationEngine to classify a sequence dataset into contiguous video sequences and static images.
    3. Serializes structured metadata manifests (JSON + CSV) via ManifestWriter.
    4. Partitions the sequence dataset into leak-free train and validation splits.
    5. Emits per-dataset ``manifest/`` and ``splits/`` artifacts.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = Path("."),
        train_ratio: float = 0.8,
        seed: int = 42,
        epsilon: float = DEFAULT_EPSILON,
        validate_images: bool = True,
        compute_hashes: bool = False,
        use_segmentation: bool = True,
    ) -> None:
        configured_data_dir = get_configured_data_root()
        self.data_dir = Path(data_dir or configured_data_dir or Path.cwd()).resolve()
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
        self.manifest_writer = ManifestWriter(dataset_name="dataset")
        self.segment_engine = VideoSegmentationEngine()
        self.last_segmentation_result: Optional[SegmentationResult] = None

    def _dataset_output_dir(self, dataset_root: Path) -> Path:
        """Return the artifact root for one dataset without mixing datasets."""
        return dataset_artifact_root(
            dataset_root,
            output_dir=self.output_dir,
            data_dir=self.data_dir,
        )

    def _resolve_dataset_roots(
        self,
        holdout_dir: Optional[Path],
        partition_dir: Optional[Path],
    ) -> Tuple[Path, Path]:
        return resolve_dataset_roots(
            self.data_dir,
            holdout_dir=holdout_dir,
            partition_dir=partition_dir,
        )

    def prepare(
        self,
        holdout_dir: Optional[Path] = None,
        partition_dir: Optional[Path] = None,
        export_manifests: bool = True,
    ) -> Tuple[IsolationResult, PartitionResult, ManifestSummary]:
        """
        Execute full preparation workflow.
        """
        holdout_root, partition_root = self._resolve_dataset_roots(holdout_dir, partition_dir)

        # 1. Isolate the holdout dataset into Test
        iso_res = self.isolator.isolate_dataset(
            holdout_root,
            validate_images=self.validate_images,
            dataset_name=holdout_root.name,
        )

        # 2. Segment & partition the sequence dataset into Train and Val
        json_path: Optional[Path] = None
        csv_path: Optional[Path] = None

        if self.use_segmentation and partition_root.exists():
            try:
                seg_res = self.segment_engine.segment_dataset(partition_root)
                self.last_segmentation_result = seg_res

                # Split-based datasets (such as DFire) do not expose a flat
                # images/ directory, so segmentation may legitimately find no
                # records. Fall back to the standard scanner in that case.
                if not seg_res.records:
                    raise ValueError("segmentation produced no records")

                if export_manifests and seg_res.records:
                    manifest_dir = self._dataset_output_dir(partition_root) / MANIFEST_DIR_NAME
                    json_path, csv_path = self.manifest_writer.write_manifests(
                        seg_res, manifest_dir
                    )

                part_res = self.partitioner.partition_segmented_frames(
                    seg_res.records,
                    dataset_name=partition_root.name,
                    root_path=partition_root,
                )
                part_res.segmentation_result = seg_res
            except Exception:
                # Fallback to standard scan partitioner
                part_res = self.partitioner.partition_dataset(
                    partition_root, validate_images=self.validate_images, use_segmentation=False
                )
        else:
            part_res = self.partitioner.partition_dataset(
                partition_root, validate_images=self.validate_images, use_segmentation=False
            )

        # 3. Generate independent YOLO manifests for each source dataset.
        # The holdout dataset contributes the held-out test split; the
        # partition dataset contributes sequence-aware train/validation splits.
        # manifest/ and splits/ directories.
        holdout_exists = holdout_root.exists()
        partition_exists = partition_root.exists()
        holdout_output_dir = self._dataset_output_dir(holdout_root)
        partition_output_dir = self._dataset_output_dir(partition_root)

        # A caller may provide one dataset root that contains both layouts.
        # In that case there is only one artifact owner and its three splits
        # should be written together instead of overwriting one another.
        same_dataset = holdout_root.resolve() == partition_root.resolve()
        if same_dataset and holdout_exists:
            manifest_summary = ManifestGenerator(output_dir=holdout_output_dir).generate(
                train_images=part_res.train_images,
                val_images=part_res.val_images,
                test_images=iso_res.test_images,
            )
            manifest_summary.manifest_json_path = json_path
            manifest_summary.manifest_csv_path = csv_path
            manifest_summary.dataset_summaries = {"dataset": manifest_summary}
        else:
            summaries: Dict[str, ManifestSummary] = {}
            holdout_summary: Optional[ManifestSummary] = None
            partition_summary: Optional[ManifestSummary] = None
            if holdout_exists:
                holdout_summary = ManifestGenerator(output_dir=holdout_output_dir).generate(
                    train_images=[],
                    val_images=[],
                    test_images=iso_res.test_images,
                )
                summaries[iso_res.dataset_name] = holdout_summary
            if partition_exists:
                partition_summary = ManifestGenerator(output_dir=partition_output_dir).generate(
                    train_images=part_res.train_images,
                    val_images=part_res.val_images,
                    test_images=[],
                )
                partition_summary.manifest_json_path = json_path
                partition_summary.manifest_csv_path = csv_path
                summaries[part_res.dataset_name] = partition_summary

            if partition_summary is not None:
                manifest_summary = partition_summary
            elif holdout_summary is not None:
                manifest_summary = holdout_summary
            else:
                # Preserve a useful return value for an empty/missing input.
                manifest_summary = ManifestGenerator(output_dir=self.output_dir).generate([], [], [])
            manifest_summary.dataset_summaries = summaries or {"dataset": manifest_summary}

        return iso_res, part_res, manifest_summary


def generate_yolo_manifests(
    train_images: List[Path],
    val_images: List[Path],
    test_images: List[Path],
    output_dir: Path = Path("."),
    splits_dirname: str = SPLITS_DIR_NAME,
    nc: int = 2,
    names: Optional[Dict[int, str]] = None,
    manifest_dirname: str = MANIFEST_DIR_NAME,
) -> ManifestSummary:
    """Convenience function to generate YOLO manifests and split text lists."""
    gen = ManifestGenerator(
        output_dir=output_dir,
        splits_dirname=splits_dirname,
        manifest_dirname=manifest_dirname,
        nc=nc,
        names=names,
    )
    return gen.generate(train_images, val_images, test_images)


def prepare_pipeline_datasets(
    data_dir: Optional[Path] = None,
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
