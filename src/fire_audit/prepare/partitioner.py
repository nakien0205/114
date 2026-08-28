"""Sequence-aware dataset partitioner for arbitrary image datasets.

Partitions datasets into leak-free train and validation splits
using deterministic sequence clustering and stratified class balancing.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from src.fire_audit.config import (
        DEFAULT_EPSILON,
        DatasetScanResult,
        ImageRecord,
        SegmentedFrame,
        SegmentationResult,
    )
    from src.fire_audit.audit.scanner import DatasetScanner
except ImportError:
    from fire_audit.config import (
        DEFAULT_EPSILON,
        DatasetScanResult,
        ImageRecord,
        SegmentedFrame,
        SegmentationResult,
    )
    from fire_audit.audit.scanner import DatasetScanner



@dataclass
class PartitionResult:
    """Result of partitioning a dataset into train and validation splits."""
    dataset_name: str = "dataset"
    root_path: Path = field(default_factory=lambda: Path("."))
    train_images: List[Path] = field(default_factory=list)
    val_images: List[Path] = field(default_factory=list)
    quarantined_images: List[Path] = field(default_factory=list)
    video_train_count: int = 0
    video_val_count: int = 0
    static_train_count: int = 0
    static_val_count: int = 0
    sequence_map: Dict[str, List[Path]] = field(default_factory=dict)
    category_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    train_ratio: float = 0.8
    random_seed: int = 42
    segmentation_result: Optional[SegmentationResult] = None

    @property
    def total_train(self) -> int:
        return len(self.train_images)

    @property
    def total_val(self) -> int:
        return len(self.val_images)

    @property
    def total_images(self) -> int:
        return len(self.train_images) + len(self.val_images)

    @property
    def total_video_frames(self) -> int:
        return self.video_train_count + self.video_val_count

    @property
    def total_static_images(self) -> int:
        return self.static_train_count + self.static_val_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "root_path": str(self.root_path),
            "total_train": self.total_train,
            "total_val": self.total_val,
            "total_images": self.total_images,
            "video_train_count": self.video_train_count,
            "video_val_count": self.video_val_count,
            "static_train_count": self.static_train_count,
            "static_val_count": self.static_val_count,
            "total_quarantined": len(self.quarantined_images),
            "train_ratio": self.train_ratio,
            "random_seed": self.random_seed,
            "category_counts": self.category_counts,
        }


class SequencePartitioner:
    """
    Partitions datasets into train and validation splits using sequence-level grouping.
    Prevents cross-split temporal/scene leakage by ensuring that all frames from
    the same sequence or scene cluster reside strictly within a single partition.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.scanner = DatasetScanner(epsilon=epsilon)

    def extract_sequence_and_category(self, path: Path) -> Tuple[str, str]:
        """
        Extract category and sequence identifier from an image path.
        Examples:
        - bothFireAndSmoke_CV000123.jpg -> category='bothFireAndSmoke', seq_id='bothFireAndSmoke_block_0002'
        - fire_CV000456.jpg -> category='fire', seq_id='fire_block_0009'
        - seq1_001.jpg -> category='seq1', seq_id='seq1'
        """
        stem = path.stem
        if "_CV" in stem:
            category = stem.split("_CV")[0]
            idx_str = stem.split("_CV")[-1]
            try:
                idx = int(idx_str)
                block_id = idx // 20
                seq_id = f"{category}_block_{block_id:04d}"
            except ValueError:
                seq_id = category
            return category, seq_id

        # Generic prefix
        parts = stem.split("_")
        category = parts[0] if parts else "default"
        seq_id = stem if len(parts) <= 1 else "_".join(parts[:-1])
        return category, seq_id

    @staticmethod
    def _extract_category(stem: str) -> str:
        """Extract dataset category prefix from filename stem."""
        if "_CV" in stem:
            return stem.split("_CV")[0]
        parts = stem.split("_")
        return parts[0] if parts else "default"

    def partition_segmented_frames(
        self,
        records: Sequence[SegmentedFrame],
        dataset_name: str = "dataset",
        root_path: Optional[Path] = None,
        train_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> PartitionResult:
        """
        Partition a sequence of SegmentedFrame records into train and val splits.
        Enforces atomic video_id assignment (0% temporal leakage) and deterministic
        stratified sampling for static images.
        """
        ratio = self.train_ratio if train_ratio is None else train_ratio
        s = self.random_seed if seed is None else seed

        if not records:
            return PartitionResult(
                dataset_name=dataset_name,
                root_path=root_path or Path("."),
                train_ratio=ratio,
                random_seed=s,
            )

        resolved_root = root_path or records[0].path.parent

        # 1. Separate into video sequences, static images, and quarantined corrupt images
        video_sequences: Dict[str, List[SegmentedFrame]] = defaultdict(list)
        video_prefix_map: Dict[str, set[str]] = defaultdict(set)
        static_by_category: Dict[str, List[SegmentedFrame]] = defaultdict(list)
        sequence_map: Dict[str, List[Path]] = defaultdict(list)
        quarantined_frames: List[Path] = []

        for rec in records:
            p = Path(rec.path)
            cat = self._extract_category(rec.filename)

            # Check if frame is corrupt (invalid dimensions <= 0 or marked corrupt)
            if rec.width <= 0 or rec.height <= 0 or getattr(rec, "is_corrupt", False):
                quarantined_frames.append(p)
                continue

            if rec.category == "video_frame" and rec.video_id:
                video_sequences[rec.video_id].append(rec)
                sequence_map[rec.video_id].append(p)
                # Map video_id to category prefix
                pref = rec.video_id.split("_video_")[0] if "_video_" in rec.video_id else cat
                video_prefix_map[pref].add(rec.video_id)
            else:
                static_by_category[cat].append(rec)
                static_key = f"{cat}_static_{rec.filename}"
                sequence_map[static_key].append(p)


        # 2. Partition video sequences atomically across whole dataset
        train_video_frames: List[Path] = []
        val_video_frames: List[Path] = []
        rng = random.Random(s)

        all_video_ids = sorted(video_sequences.keys())
        if all_video_ids:
            shuffled_vids = all_video_ids[:]
            rng.shuffle(shuffled_vids)

            if len(shuffled_vids) == 1:
                cutoff = 1 if ratio >= 0.5 else 0
            else:
                cutoff = int(len(shuffled_vids) * ratio)
                if cutoff == 0 and ratio > 0:
                    cutoff = 1
                elif cutoff == len(shuffled_vids) and ratio < 1.0:
                    cutoff = len(shuffled_vids) - 1

            train_vids = set(shuffled_vids[:cutoff])
            val_vids = set(shuffled_vids[cutoff:])

            for vid_id in sorted(train_vids):
                for f in sorted(video_sequences[vid_id], key=lambda x: x.frame_index if x.frame_index is not None else 0):
                    train_video_frames.append(Path(f.path))

            for vid_id in sorted(val_vids):
                for f in sorted(video_sequences[vid_id], key=lambda x: x.frame_index if x.frame_index is not None else 0):
                    val_video_frames.append(Path(f.path))

        # 3. Partition static images with seeded stratified sampling
        train_static_frames: List[Path] = []
        val_static_frames: List[Path] = []

        for cat in sorted(static_by_category.keys()):
            cat_frames = sorted(static_by_category[cat], key=lambda x: x.filename)
            if not cat_frames:
                continue

            shuffled_cat = cat_frames[:]
            rng.shuffle(shuffled_cat)

            if len(shuffled_cat) == 1:
                cutoff = 1 if ratio >= 0.5 else 0
            else:
                cutoff = int(len(shuffled_cat) * ratio)
                if cutoff == 0 and ratio > 0:
                    cutoff = 1
                elif cutoff == len(shuffled_cat) and ratio < 1.0:
                    cutoff = len(shuffled_cat) - 1

            train_cat = shuffled_cat[:cutoff]
            val_cat = shuffled_cat[cutoff:]

            for f in train_cat:
                train_static_frames.append(Path(f.path))
            for f in val_cat:
                val_static_frames.append(Path(f.path))


        # 4. Build category counts
        category_counts: Dict[str, Dict[str, int]] = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "quarantined": defaultdict(int),
        }

        for p in train_video_frames:
            cat = self._extract_category(p.name)
            category_counts["train"][cat] += 1
        for p in train_static_frames:
            cat = self._extract_category(p.name)
            category_counts["train"][cat] += 1

        for p in val_video_frames:
            cat = self._extract_category(p.name)
            category_counts["val"][cat] += 1
        for p in val_static_frames:
            cat = self._extract_category(p.name)
            category_counts["val"][cat] += 1

        for p in quarantined_frames:
            cat = self._extract_category(p.name)
            category_counts["quarantined"][cat] += 1

        # Combine train and val image lists (sorted deterministically)
        all_train = sorted(train_video_frames + train_static_frames)
        all_val = sorted(val_video_frames + val_static_frames)

        return PartitionResult(
            dataset_name=dataset_name,
            root_path=resolved_root,
            train_images=all_train,
            val_images=all_val,
            quarantined_images=sorted(quarantined_frames),
            video_train_count=len(train_video_frames),
            video_val_count=len(val_video_frames),
            static_train_count=len(train_static_frames),
            static_val_count=len(val_static_frames),
            sequence_map=dict(sequence_map),
            category_counts={k: dict(v) for k, v in category_counts.items()},

            train_ratio=ratio,
            random_seed=s,
        )

    def partition_sequences(
        self,
        sequence_map: Dict[str, List[Path]],
        train_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Path], List[Path]]:
        """
        Deterministically partition a mapping of sequence_id -> [image_paths] into train and val.
        Ensures all frames of each sequence remain strictly in one split.
        """
        ratio = self.train_ratio if train_ratio is None else train_ratio
        s = self.random_seed if seed is None else seed

        seq_ids = sorted(sequence_map.keys())
        if not seq_ids:
            return [], []

        if len(seq_ids) == 1:
            if ratio >= 0.5:
                return list(sequence_map[seq_ids[0]]), []
            else:
                return [], list(sequence_map[seq_ids[0]])

        rng = random.Random(s)
        shuffled = seq_ids[:]
        rng.shuffle(shuffled)

        cutoff = int(len(shuffled) * ratio)
        if cutoff == 0 and ratio > 0:
            cutoff = 1
        elif cutoff == len(shuffled) and ratio < 1.0:
            cutoff = len(shuffled) - 1

        train_seqs = set(shuffled[:cutoff])
        val_seqs = set(shuffled[cutoff:])

        train_images: List[Path] = []
        val_images: List[Path] = []

        for seq_id in sorted(train_seqs):
            train_images.extend(sorted(sequence_map[seq_id]))

        for seq_id in sorted(val_seqs):
            val_images.extend(sorted(sequence_map[seq_id]))

        return train_images, val_images

    def partition(
        self,
        data: Union[SegmentationResult, Sequence[Union[SegmentedFrame, ImageRecord, Dict[str, Any], Path]], DatasetScanResult],
        dataset_name: str = "dataset",
        root_path: Optional[Path] = None,
        train_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> PartitionResult:
        """
        Unified partition entrypoint supporting SegmentationResult, SegmentedFrame list,
        ImageRecord list, DatasetScanResult, or Path sequence.
        """
        if type(data).__name__ == "SegmentationResult" or isinstance(data, SegmentationResult):
            res = self.partition_segmented_frames(
                getattr(data, "records", []),
                dataset_name=dataset_name,
                root_path=root_path,
                train_ratio=train_ratio,
                seed=seed,
            )
            res.segmentation_result = data
            return res

        if type(data).__name__ == "DatasetScanResult" or isinstance(data, DatasetScanResult):
            return self.partition_from_scan_result(data)

        if isinstance(data, (list, tuple, Sequence)):
            if not data:
                return PartitionResult(
                    dataset_name=dataset_name,
                    root_path=root_path or Path("."),
                    train_ratio=self.train_ratio if train_ratio is None else train_ratio,
                    random_seed=self.random_seed if seed is None else seed,
                )
            first = data[0]
            if type(first).__name__ == "SegmentedFrame" or isinstance(first, SegmentedFrame) or hasattr(first, "video_id"):
                return self.partition_segmented_frames(
                    data,  # type: ignore
                    dataset_name=dataset_name,
                    root_path=root_path,
                    train_ratio=train_ratio,
                    seed=seed,
                )
            if type(first).__name__ == "ImageRecord" or isinstance(first, ImageRecord) or hasattr(first, "image_path"):
                return self.partition_from_records(
                    data,  # type: ignore
                    dataset_name=dataset_name,
                    root_path=root_path,
                )
            if isinstance(first, Path):
                records = [ImageRecord(image_path=p) for p in data]  # type: ignore
                return self.partition_from_records(
                    records,
                    dataset_name=dataset_name,
                    root_path=root_path,
                )
            if isinstance(first, dict):
                # Convert dicts to SegmentedFrame if category is present
                frames: List[SegmentedFrame] = []
                for d in data:  # type: ignore
                    f_p = Path(d.get("path", d.get("filename", "")))
                    frames.append(SegmentedFrame(
                        filename=d.get("filename", f_p.name),
                        path=f_p,
                        category=d.get("category", "static_image"),
                        video_id=d.get("video_id"),
                        frame_index=d.get("frame_index"),
                        sequence_length=d.get("sequence_length"),
                        width=d.get("width", 0),
                        height=d.get("height", 0),
                        fire_boxes=d.get("fire_boxes", 0),
                        smoke_boxes=d.get("smoke_boxes", 0),
                        is_negative=d.get("is_negative", False),
                    ))
                return self.partition_segmented_frames(
                    frames,
                    dataset_name=dataset_name,
                    root_path=root_path,
                    train_ratio=train_ratio,
                    seed=seed,
                )


        raise TypeError(f"Unsupported data type for SequencePartitioner: {type(data)}")

    def partition_dataset(
        self,
        dataset_root: Path,
        validate_images: bool = True,
        use_segmentation: bool = True,
    ) -> PartitionResult:
        """
        Scan and partition a dataset into train and validation splits.
        Uses VideoSegmentationEngine when available, with a scanner fallback.
        """
        result = PartitionResult(
            dataset_name=dataset_root.name,
            root_path=dataset_root,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
        )

        if not dataset_root.exists():
            return result

        if use_segmentation:
            try:
                try:
                    from src.fire_audit.segment.engine import VideoSegmentationEngine
                except ImportError:
                    from fire_audit.segment.engine import VideoSegmentationEngine
                engine = VideoSegmentationEngine()


                seg_res = engine.segment_dataset(dataset_root)
                part_res = self.partition_segmented_frames(
                    seg_res.records,
                    dataset_name=dataset_root.name,
                    root_path=dataset_root,
                )
                part_res.segmentation_result = seg_res
                return part_res
            except Exception:
                pass

        # Split-based exports use train/valid/test/images directories rather
        # than a flat images/ directory. Reuse the split-aware scanner
        # scanner for that topology so those datasets are still partitioned.
        if not (dataset_root / "images").is_dir() and any(
            (dataset_root / name / "images").is_dir()
            for name in ("train", "valid", "val", "test")
        ):
            scan_res = self.scanner.scan_split_dataset(
                dataset_root,
                validate_images=validate_images,
                dataset_name=dataset_root.name,
            )
        else:
            scan_res = self.scanner.scan_dataset(
                dataset_root,
                dataset_name=dataset_root.name,
                validate_images=validate_images,
            )
        return self.partition_from_scan_result(scan_res)

    def partition_from_scan_result(
        self,
        scan_result: DatasetScanResult,
    ) -> PartitionResult:
        """Partition pre-scanned records with sequence grouping and stratification."""
        result = PartitionResult(
            dataset_name=scan_result.dataset_name,
            root_path=scan_result.root_path,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
        )

        sequence_map: Dict[str, List[Path]] = {}
        category_counts: Dict[str, Dict[str, int]] = {
            "train": {},
            "val": {},
            "quarantined": {},
        }

        for rec in scan_result.records:
            cat, seq_id = self.extract_sequence_and_category(rec.image_path)
            # Use record's sequence_id if already computed
            if rec.sequence_id:
                seq_id = rec.sequence_id

            if rec.is_corrupt:
                result.quarantined_images.append(rec.image_path)
                category_counts["quarantined"][cat] = category_counts["quarantined"].get(cat, 0) + 1
            else:
                sequence_map.setdefault(seq_id, []).append(rec.image_path)

        result.sequence_map = sequence_map

        # Partition sequences
        train_imgs, val_imgs = self.partition_sequences(
            sequence_map,
            train_ratio=self.train_ratio,
            seed=self.random_seed,
        )

        result.train_images = train_imgs
        result.val_images = val_imgs

        # Count categories in train and val
        for img in train_imgs:
            cat, _ = self.extract_sequence_and_category(img)
            category_counts["train"][cat] = category_counts["train"].get(cat, 0) + 1

        for img in val_imgs:
            cat, _ = self.extract_sequence_and_category(img)
            category_counts["val"][cat] = category_counts["val"].get(cat, 0) + 1

        result.category_counts = category_counts
        return result

    def partition_from_records(
        self,
        records: List[ImageRecord],
        dataset_name: str = "dataset",
        root_path: Optional[Path] = None,
    ) -> PartitionResult:
        """Partition directly from a list of ImageRecord objects."""
        scan_res = DatasetScanResult(
            dataset_name=dataset_name,
            root_path=root_path or (records[0].image_path.parent if records else Path(".")),
            records=records,
        )
        return self.partition_from_scan_result(scan_res)


def partition_dataset_by_sequence(
    dataset_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    validate_images: bool = True,
    use_segmentation: bool = True,
    epsilon: float = DEFAULT_EPSILON,
) -> PartitionResult:
    """Convenience function to partition one dataset by sequence."""
    partitioner = SequencePartitioner(
        train_ratio=train_ratio,
        random_seed=seed,
        epsilon=epsilon,
    )
    return partitioner.partition_dataset(
        dataset_root,
        validate_images=validate_images,
        use_segmentation=use_segmentation,
    )
