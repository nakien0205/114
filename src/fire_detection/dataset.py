"""
Dataset verification module for Fire and Smoke Detection.
Validates YOLO dataset configuration, split image files, and label annotations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES: Dict[int, str] = {0: "fire", 1: "smoke"}


@dataclass
class BBoxAnnotation:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitStats:
    split_name: str
    images_count: int = 0
    labels_count: int = 0
    boxes_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=lambda: {"fire": 0, "smoke": 0})
    empty_images_count: int = 0
    resolutions: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class DatasetVerificationResult:
    status: str  # "PASS" or "FAIL"
    dataset_path: str
    num_classes: int
    class_names: Dict[int, str]
    splits: Dict[str, SplitStats]
    total_images: int = 0
    total_boxes: int = 0
    total_fire_boxes: int = 0
    total_smoke_boxes: int = 0
    total_empty_images: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "status": self.status,
            "dataset_path": self.dataset_path,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "total_images": self.total_images,
            "total_boxes": self.total_boxes,
            "total_fire_boxes": self.total_fire_boxes,
            "total_smoke_boxes": self.total_smoke_boxes,
            "total_empty_images": self.total_empty_images,
            "splits": {k: asdict(v) for k, v in self.splits.items()},
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DatasetVerifier:
    """Programmatic verification engine for YOLO fire and smoke datasets."""

    def __init__(self, tolerance: float = 1e-4) -> None:
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        self.tolerance = tolerance

    def load_config(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse dataset yaml configuration."""
        p = Path(yaml_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset config YAML not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {p}: expected dictionary")
        return data

    def resolve_split_images(
        self, split_val: Union[str, Path, List[str]], base_dir: Path, dataset_root: Optional[Path] = None
    ) -> List[Path]:
        """Resolve list of image paths from split value (directory, txt file, or list)."""
        image_paths: List[Path] = []
        root = dataset_root if dataset_root is not None else base_dir

        if isinstance(split_val, list):
            for item in split_val:
                item_path = Path(item)
                if not item_path.is_absolute():
                    item_path = root / item_path
                image_paths.append(item_path)
            return image_paths

        target_path = Path(split_val)
        if not target_path.is_absolute():
            cand = root / target_path
            if cand.exists():
                target_path = cand
            else:
                cand2 = base_dir / target_path
                if cand2.exists():
                    target_path = cand2
                else:
                    target_path = cand

        if not target_path.exists():
            raise FileNotFoundError(f"Split path not found on disk: {target_path}")

        if target_path.is_dir():
            # If target_path has an 'images' subfolder and is not already named 'images', prioritize it
            if (target_path / "images").is_dir() and target_path.name.lower() != "images":
                search_dirs = [target_path / "images", target_path]
            else:
                search_dirs = [target_path]
                if (target_path / "images").is_dir():
                    search_dirs.append(target_path / "images")

            found_any = False
            for s_dir in search_dirs:
                for f in sorted(s_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                        image_paths.append(f)
                        found_any = True
                if found_any:
                    break
        elif target_path.is_file():
            # If it's a text split manifest
            lines = target_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                l_str = line.strip()
                if not l_str or l_str.startswith("#"):
                    continue
                img_p = Path(l_str)
                if not img_p.is_absolute():
                    cand1 = root / img_p
                    if cand1.exists():
                        img_p = cand1
                    else:
                        img_p = base_dir / img_p
                image_paths.append(img_p)
        return image_paths

    def resolve_label_path(self, img_path: Path) -> Path:
        """Derive label file path from image path supporting multiple standard YOLO structures."""
        # 1. Structure: .../train/images/img.jpg -> .../train/labels/img.txt
        if img_path.parent.name == "images":
            candidate = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if candidate.exists():
                return candidate

        # 2. Structure: .../images/train/img.jpg -> .../labels/train/img.txt
        parts = list(img_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels"
            lbl_parts[-1] = f"{img_path.stem}.txt"
            candidate = Path(*lbl_parts)
            if candidate.exists():
                return candidate

        # 3. Structure: .../train/labels/img.txt
        candidate = img_path.parent / "labels" / f"{img_path.stem}.txt"
        if candidate.exists():
            return candidate

        # 4. Structure: .../train/img.txt (same directory)
        candidate = img_path.parent / f"{img_path.stem}.txt"
        if candidate.exists():
            return candidate

        # Default fallback for the standard YOLO layout
        if img_path.parent.name == "images":
            return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels"
            lbl_parts[-1] = f"{img_path.stem}.txt"
            return Path(*lbl_parts)
        return img_path.parent / f"{img_path.stem}.txt"

    def validate_label_file(
        self, label_path: Path, num_classes: int = 2
    ) -> Tuple[List[BBoxAnnotation], List[str]]:
        """Parse and validate a YOLO bounding box label file."""
        boxes: List[BBoxAnnotation] = []
        errors: List[str] = []

        if not label_path.exists():
            errors.append(f"Label file missing: {label_path}")
            return boxes, errors

        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            errors.append(f"Cannot read label file {label_path}: {e}")
            return boxes, errors

        if not content:
            # Background / empty image (valid in YOLO)
            return boxes, errors

        for line_num, line in enumerate(content.splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_num}: Invalid number of values (expected 5, got {len(parts)}): '{line}'"
                )
                continue

            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                errors.append(f"{label_path.name}:{line_num}: Non-numeric bounding box values: {e}")
                continue

            line_has_error = False

            if cls_id < 0 or cls_id >= num_classes:
                errors.append(
                    f"{label_path.name}:{line_num}: Invalid class_id {cls_id} (allowed: 0..{num_classes-1})"
                )
                line_has_error = True

            # Check normalized coordinate ranges [0.0 - tol, 1.0 + tol] and finite numbers
            for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
                if math.isnan(val) or math.isinf(val):
                    errors.append(
                        f"{label_path.name}:{line_num}: Coordinate {name} is non-finite: {val}"
                    )
                    line_has_error = True
                elif val < -self.tolerance or val > (1.0 + self.tolerance):
                    errors.append(
                        f"{label_path.name}:{line_num}: Coordinate {name}={val} out of bounds [0.0, 1.0]"
                    )
                    line_has_error = True

            if math.isnan(w) or math.isnan(h) or w <= 0.0 or h <= 0.0:
                errors.append(
                    f"{label_path.name}:{line_num}: Non-positive dimension w={w}, h={h}"
                )
                line_has_error = True

            if not line_has_error:
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                boxes.append(
                    BBoxAnnotation(
                        class_id=cls_id,
                        class_name=cls_name,
                        x_center=xc,
                        y_center=yc,
                        width=w,
                        height=h,
                    )
                )

        return boxes, errors

    def verify(
        self,
        data_yaml_path: Union[str, Path],
        check_images_readable: bool = True,
        max_image_checks: Optional[int] = None,
    ) -> DatasetVerificationResult:
        """Run comprehensive verification on the dataset."""
        if max_image_checks is not None and max_image_checks <= 0:
            raise ValueError(f"max_image_checks must be a positive integer, got {max_image_checks}")

        yaml_path = Path(data_yaml_path).resolve()
        config = self.load_config(yaml_path)

        base_dir = yaml_path.parent
        data_root_raw = config.get("path", "")
        data_root: Optional[Path] = None
        if data_root_raw:
            data_root = Path(data_root_raw)
            if not data_root.is_absolute():
                data_root = base_dir / data_root

        nc = config.get("nc", 2)
        names_raw = config.get("names", {0: "fire", 1: "smoke"})
        if isinstance(names_raw, list):
            class_names = {i: n for i, n in enumerate(names_raw)}
        elif isinstance(names_raw, dict):
            try:
                class_names = {int(k): str(v) for k, v in names_raw.items()}
            except ValueError:
                class_names = {int(v): str(k) for k, v in names_raw.items()}
        else:
            class_names = {0: "fire", 1: "smoke"}

        splits_to_check = ["train", "val", "test"]
        split_stats: Dict[str, SplitStats] = {}
        all_errors: List[str] = []
        all_warnings: List[str] = []

        if len(class_names) != nc:
            all_warnings.append(f"Class count mismatch in {yaml_path.name}: nc={nc} but names dict has {len(class_names)} classes")

        total_images = 0
        total_boxes = 0
        total_fire = 0
        total_smoke = 0
        total_empty = 0

        for s_name in splits_to_check:
            if s_name not in config:
                err = f"Split '{s_name}' not defined in {yaml_path.name}"
                all_errors.append(err)
                split_stats[s_name] = SplitStats(split_name=s_name, errors=[err])
                continue

            try:
                images = self.resolve_split_images(config[s_name], base_dir, data_root)
            except Exception as e:
                err = f"Failed to resolve images for split '{s_name}': {e}"
                all_errors.append(err)
                split_stats[s_name] = SplitStats(split_name=s_name, errors=[err])
                continue

            if len(images) == 0:
                err = f"Split '{s_name}' contains 0 valid images ({config[s_name]})"
                all_errors.append(err)
                split_stats[s_name] = SplitStats(split_name=s_name, errors=[err])
                continue

            stats = SplitStats(split_name=s_name, images_count=len(images))

            # Also check if there are orphaned label files without corresponding images
            if isinstance(config[s_name], (str, Path)):
                target_split_path = Path(config[s_name])
                if not target_split_path.is_absolute():
                    target_split_path = (data_root or base_dir) / target_split_path
                if target_split_path.is_dir():
                    lbl_dir = target_split_path.parent / "labels" if target_split_path.name == "images" else target_split_path
                    if lbl_dir.exists() and lbl_dir.is_dir():
                        img_stems = {img_p.stem for img_p in images}
                        for lbl_file in sorted(lbl_dir.iterdir()):
                            if lbl_file.is_file() and lbl_file.suffix == ".txt":
                                if lbl_file.stem not in img_stems:
                                    err = f"Label file {lbl_file.name} exists but corresponding image is missing on disk in {target_split_path}"
                                    stats.errors.append(err)
                                    all_errors.append(err)

            checked_images = 0
            for img_p in images:
                if not img_p.exists():
                    err = f"Image file missing on disk: {img_p}"
                    stats.errors.append(err)
                    all_errors.append(err)
                    continue

                if check_images_readable and (
                    max_image_checks is None or checked_images < max_image_checks
                ):
                    try:
                        with Image.open(img_p) as img:
                            img.verify()
                        # Re-open to get size (verify clears internal state)
                        with Image.open(img_p) as img:
                            res_key = f"{img.width}x{img.height}"
                            stats.resolutions[res_key] = stats.resolutions.get(res_key, 0) + 1
                        checked_images += 1
                    except Exception as e:
                        err = f"Corrupted image {img_p}: {e}"
                        stats.errors.append(err)
                        all_errors.append(err)

                lbl_p = self.resolve_label_path(img_p)
                if not lbl_p.exists():
                    err = f"Missing label file for image: {img_p} (expected {lbl_p})"
                    stats.errors.append(err)
                    all_errors.append(err)
                    continue

                stats.labels_count += 1
                boxes, lbl_errors = self.validate_label_file(lbl_p, num_classes=nc)
                if lbl_errors:
                    stats.errors.extend(lbl_errors)
                    all_errors.extend(lbl_errors)

                if not boxes:
                    stats.empty_images_count += 1
                else:
                    for b in boxes:
                        stats.boxes_count += 1
                        c_name = class_names.get(b.class_id, f"class_{b.class_id}")
                        stats.class_counts[c_name] = stats.class_counts.get(c_name, 0) + 1

            split_stats[s_name] = stats
            total_images += stats.images_count
            total_boxes += stats.boxes_count
            total_fire += stats.class_counts.get("fire", 0)
            total_smoke += stats.class_counts.get("smoke", 0)
            total_empty += stats.empty_images_count

        status = "PASS" if not all_errors else "FAIL"
        return DatasetVerificationResult(
            status=status,
            dataset_path=str(data_root or base_dir),
            num_classes=nc,
            class_names=class_names,
            splits=split_stats,
            total_images=total_images,
            total_boxes=total_boxes,
            total_fire_boxes=total_fire,
            total_smoke_boxes=total_smoke,
            total_empty_images=total_empty,
            errors=all_errors,
            warnings=all_warnings,
        )


def verify_dataset(
    data_yaml_path: Union[str, Path] = "data.yaml",
    json_out: Optional[Union[str, Path]] = None,
    check_images_readable: bool = True,
    max_image_checks: Optional[int] = None,
) -> DatasetVerificationResult:
    """High-level function to verify dataset and optionally save report."""
    verifier = DatasetVerifier()
    result = verifier.verify(
        data_yaml_path=data_yaml_path,
        check_images_readable=check_images_readable,
        max_image_checks=max_image_checks,
    )
    if json_out is not None:
        out_p = Path(json_out).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
    return result
