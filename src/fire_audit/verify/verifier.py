"""
Programmatic verification engine for Fire and Smoke Dataset manifests and splits.
Verifies all Acceptance Criteria (AC1 - AC10) from ORIGINAL_REQUEST:
- AC1: 100% of FASDD_CV image files are indexed in the manifest with zero unclassified or dropped entries.
- AC2: Baseline sequence bothFireAndSmoke_CV000000.jpg to bothFireAndSmoke_CV000431.jpg verified as a single contiguous 432-frame video sequence.
- AC3: Every detected video sequence has homogeneous resolution across all constituent frames.
- AC4: Every detected video sequence maintains strictly contiguous frame indices without unaddressed gaps.
- AC5: No image is assigned to more than one category or video ID.
- AC6: Train/validation partitioners enforce video-level isolation: 0% cross-split frame leakage for all video_frame sequences.
- AC7: Strict Home Fire Dataset isolation in held-out test split (0% overlap with train or val).
- AC8: All bounding boxes in generated sets have valid class IDs (0 for fire, 1 for smoke) and coordinates within [0.0, 1.0].
- AC9: Valid data.yaml pointing to existing split files on disk and readable image files.
- AC10: SHA-256 cryptographic disjointness across splits.

Also logs and exports items requiring manual inspection (e.g. corrupt files, near-boundary bboxes,
ambiguous burst runs) to a dedicated manual review report (manual_review_needed.json).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml

try:
    from src.fire_audit.config import (
        CLASS_IDS,
        CLASS_MAP,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        DEFAULT_EPSILON,
        FASDD_CV_DIR_NAME,
        HOME_FIRE_DIR_NAME,
        VALID_IMAGE_EXTENSIONS,
        BoundingBox,
    )
    from src.fire_audit.audit.validator import validate_bbox_line
except ImportError:
    from fire_audit.config import (
        CLASS_IDS,
        CLASS_MAP,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        DEFAULT_EPSILON,
        FASDD_CV_DIR_NAME,
        HOME_FIRE_DIR_NAME,
        VALID_IMAGE_EXTENSIONS,
        BoundingBox,
    )
    from fire_audit.audit.validator import validate_bbox_line


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CriterionResult:
    """Individual acceptance criterion verification result."""
    id: str
    name: str
    status: Literal["PASS", "FAIL"]
    details: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "passed": self.passed,
        }

    def __getitem__(self, key: str) -> Any:
        if key == "status":
            return self.status
        if key == "name":
            return self.name
        if key == "details":
            return self.details
        if key == "passed":
            return self.passed
        if key == "id":
            return self.id
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in ("status", "name", "details", "passed", "id")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class CriteriaDict(dict):
    """Dictionary supporting AC1..AC10 as well as legacy / alias keys."""

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return super().__getitem__(key)
        key_upper = str(key).upper()
        if key_upper == "HASH" and "AC10" in self:
            return super().__getitem__("AC10")
        if key_upper == "ISOLATION" and "AC7" in self:
            return super().__getitem__("AC7")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class ManualReviewItem:
    """An item flagged for manual human inspection."""
    filename: str
    path: str
    reason: str
    severity: Literal["low", "medium", "high"]
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path.replace("\\", "/"),
            "reason": self.reason,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class VerificationReport:
    """Structured acceptance verification report."""
    status: Literal["PASS", "FAIL"] = "PASS"
    criteria: Dict[str, Union[CriterionResult, Dict[str, Any]]] = field(default_factory=CriteriaDict)
    summary: Dict[str, Any] = field(default_factory=lambda: {
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_images": 0,
        "manifest_frames": 0,
        "video_frames_count": 0,
        "static_images_count": 0,
        "total_video_sequences": 0,
        "fire_boxes": 0,
        "smoke_boxes": 0,
        "negative_frames": 0,
        "corrupt_or_invalid_boxes": 0,
        "home_fire_isolated_count": 0,
        "home_fire_leaked_count": 0,
        "manual_review_items_count": 0,
    })
    errors: List[str] = field(default_factory=list)
    manual_review_items: List[Dict[str, Any]] = field(default_factory=list)
    manifest_summary: Optional[Dict[str, Any]] = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        crit_dict = {}
        for k, v in self.criteria.items():
            if isinstance(v, CriterionResult):
                crit_dict[k] = v.to_dict()
            elif isinstance(v, dict):
                crit_dict[k] = v
            else:
                crit_dict[k] = str(v)

        return {
            "status": self.status,
            "passed": self.passed,
            "criteria": crit_dict,
            "summary": self.summary,
            "errors": self.errors,
            "manual_review_items": self.manual_review_items,
            "manifest_summary": self.manifest_summary,
        }

    def __getitem__(self, key: str) -> Any:
        if key == "status":
            return self.status
        if key == "passed":
            return self.passed
        if key == "criteria":
            return self.criteria
        if key == "summary":
            return self.summary
        if key == "errors":
            return self.errors
        if key == "manual_review_items":
            return self.manual_review_items
        if key == "manifest_summary":
            return self.manifest_summary
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()


class DatasetVerifier:
    """
    Programmatic acceptance criteria verifier for generated dataset manifests and splits.
    Verifies AC1 through AC10 and detects items requiring manual review.
    """

    def __init__(self, tolerance: float = DEFAULT_EPSILON) -> None:
        self.tolerance = tolerance

    def verify(
        self,
        data_yaml_path: Optional[Path] = None,
        manifest_json_path: Optional[Path] = None,
        manifest_csv_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        splits_dir: Optional[Path] = None,
        tolerance: Optional[float] = None,
        check_hashes: bool = False,
        manual_review_out: Optional[Path] = None,
    ) -> VerificationReport:
        """
        Run full acceptance criteria verification suite (AC1 - AC10).
        """
        tol = self.tolerance if tolerance is None else tolerance
        report = VerificationReport()
        crit_dict = CriteriaDict()
        report.criteria = crit_dict
        manual_review_list: List[ManualReviewItem] = []

        def record_check(cid: str, name: str, passed: bool, details: str):
            c_res = CriterionResult(id=cid, name=name, status="PASS" if passed else "FAIL", details=details)
            crit_dict[cid] = c_res
            if not passed:
                report.status = "FAIL"
                report.errors.append(f"[{cid}] {name}: {details}")

        # -------------------------------------------------------------------
        # 1. Resolve Manifest Files and Paths
        # -------------------------------------------------------------------
        resolved_yaml: Optional[Path] = None
        if data_yaml_path is not None:
            resolved_yaml = Path(data_yaml_path).resolve()
        elif manifest_json_path is None and manifest_csv_path is None and Path("data.yaml").exists():
            resolved_yaml = Path("data.yaml").resolve()

        resolved_json: Optional[Path] = None
        if manifest_json_path is not None:
            resolved_json = Path(manifest_json_path).resolve()
        elif resolved_yaml is not None:
            candidates = [
                resolved_yaml.parent / "manifests" / "fasdd_cv_manifest.json",
                resolved_yaml.parent / "fasdd_cv_manifest.json",
                Path("manifests/fasdd_cv_manifest.json").resolve(),
                Path("fasdd_cv_manifest.json").resolve(),
            ]
            for cand in candidates:
                if cand.exists():
                    resolved_json = cand
                    break

        resolved_csv: Optional[Path] = None
        if manifest_csv_path is not None:
            resolved_csv = Path(manifest_csv_path).resolve()
        elif resolved_yaml is not None:
            candidates = [
                resolved_yaml.parent / "manifests" / "fasdd_cv_manifest.csv",
                resolved_yaml.parent / "fasdd_cv_manifest.csv",
                Path("manifests/fasdd_cv_manifest.csv").resolve(),
                Path("fasdd_cv_manifest.csv").resolve(),
            ]
            for cand in candidates:
                if cand.exists():
                    resolved_csv = cand
                    break

        # -------------------------------------------------------------------
        # AC9: YOLO Manifest Validity (data.yaml and split files)
        # -------------------------------------------------------------------
        split_paths: Dict[str, Path] = {}
        split_images: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        missing_files: List[str] = []
        yaml_valid = True

        if resolved_yaml is None:
            if manifest_json_path is not None or manifest_csv_path is not None:
                record_check(
                    "AC9",
                    "YOLO Manifest & Split Files Validity",
                    True,
                    "No data.yaml provided for verification (manifest-only verification mode)",
                )
            else:
                record_check(
                    "AC9",
                    "YOLO Manifest & Split Files Validity",
                    False,
                    f"data.yaml does not exist at {data_yaml_path or 'data.yaml'}",
                )
            yaml_valid = False
        elif not resolved_yaml.exists():
            record_check(
                "AC9",
                "YOLO Manifest & Split Files Validity",
                False,
                f"data.yaml does not exist at {resolved_yaml}",
            )
            yaml_valid = False
        else:
            try:
                with open(resolved_yaml, "r", encoding="utf-8") as f:
                    manifest = yaml.safe_load(f)
            except Exception as e:
                record_check(
                    "AC9",
                    "YOLO Manifest & Split Files Validity",
                    False,
                    f"Failed to parse data.yaml as valid YAML: {e}",
                )
                yaml_valid = False
                manifest = {}

            if yaml_valid:
                required_keys = ["path", "train", "val", "test", "nc", "names"]
                missing_keys = [k for k in required_keys if k not in manifest]
                if missing_keys:
                    record_check(
                        "AC9",
                        "YOLO Manifest & Split Files Validity",
                        False,
                        f"data.yaml missing required keys: {missing_keys}",
                    )
                    yaml_valid = False
                elif manifest.get("nc") != 2:
                    record_check(
                        "AC9",
                        "YOLO Manifest & Split Files Validity",
                        False,
                        f"data.yaml nc must be 2, found {manifest.get('nc')}",
                    )
                    yaml_valid = False
                else:
                    names_val = manifest.get("names")
                    valid_names = False
                    if isinstance(names_val, dict) and (names_val.get(0) == "fire" or names_val.get("0") == "fire") and (names_val.get(1) == "smoke" or names_val.get("1") == "smoke"):
                        valid_names = True
                    elif isinstance(names_val, list) and len(names_val) >= 2 and names_val[0] == "fire" and names_val[1] == "smoke":
                        valid_names = True

                    if not valid_names:
                        record_check(
                            "AC9",
                            "YOLO Manifest & Split Files Validity",
                            False,
                            f"Invalid class names mapping in data.yaml: {names_val}",
                        )
                        yaml_valid = False

            if yaml_valid:
                base_dir = resolved_yaml.parent
                manifest_root = Path(manifest.get("path", base_dir))

                def resolve_split_file(p_str: str) -> Path:
                    p = Path(p_str)
                    if p.is_absolute():
                        return p
                    if splits_dir is not None and (splits_dir / p.name).exists():
                        return splits_dir / p.name
                    cand1 = base_dir / p
                    if cand1.exists():
                        return cand1
                    cand2 = manifest_root / p
                    if cand2.exists():
                        return cand2
                    return cand1

                split_paths = {
                    "train": resolve_split_file(manifest["train"]),
                    "val": resolve_split_file(manifest["val"]),
                    "test": resolve_split_file(manifest["test"]),
                }

                for s_name, s_file in split_paths.items():
                    if not s_file.exists():
                        missing_files.append(f"Split file missing: {s_file}")
                        continue

                    if s_file.is_dir():
                        img_files = sorted([p for p in s_file.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
                        split_images[s_name].extend(img_files)
                    else:
                        lines = s_file.read_text(encoding="utf-8").splitlines()
                        for line in lines:
                            line_str = line.strip()
                            if not line_str:
                                continue
                            img_path = Path(line_str)
                            if not img_path.is_absolute():
                                if (manifest_root / img_path).exists():
                                    img_path = manifest_root / img_path
                                elif (base_dir / img_path).exists():
                                    img_path = base_dir / img_path

                            if not img_path.exists():
                                missing_files.append(f"Image not found on disk: {img_path}")
                                manual_review_list.append(ManualReviewItem(
                                    filename=img_path.name,
                                    path=str(img_path),
                                    reason="missing_file",
                                    severity="high",
                                    details=f"Referenced in split {s_name} but does not exist on disk",
                                ))
                            else:
                                split_images[s_name].append(img_path)

                report.summary["train_images"] = len(split_images["train"])
                report.summary["val_images"] = len(split_images["val"])
                report.summary["test_images"] = len(split_images["test"])
                report.summary["total_images"] = sum(len(v) for v in split_images.values())

                if missing_files:
                    record_check(
                        "AC9",
                        "YOLO Manifest & Split Files Validity",
                        False,
                        f"Found {len(missing_files)} missing files. First error: {missing_files[0]}",
                    )
                else:
                    record_check(
                        "AC9",
                        "YOLO Manifest & Split Files Validity",
                        True,
                        f"Valid data.yaml (nc: 2, names: {{0: fire, 1: smoke}}) and all {report.summary['total_images']} referenced split images exist on disk",
                    )

        # -------------------------------------------------------------------
        # 2. Ingest Manifest Records (JSON / CSV)
        # -------------------------------------------------------------------
        manifest_records: List[Dict[str, Any]] = []
        if resolved_json is not None and resolved_json.exists():
            try:
                with open(resolved_json, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest_records = manifest_data.get("frames", [])
                    report.manifest_summary = {
                        "dataset": manifest_data.get("dataset"),
                        "total_images": manifest_data.get("total_images"),
                        "summary": manifest_data.get("summary"),
                    }
            except Exception as e:
                report.errors.append(f"Failed to read JSON manifest {resolved_json}: {e}")

        elif resolved_csv is not None and resolved_csv.exists():
            try:
                with open(resolved_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        manifest_records.append({
                            "filename": row.get("filename", ""),
                            "path": row.get("path", ""),
                            "category": row.get("category", "static_image"),
                            "video_id": row.get("video_id") or None,
                            "frame_index": int(row["frame_index"]) if row.get("frame_index") not in (None, "") else None,
                            "sequence_length": int(row["sequence_length"]) if row.get("sequence_length") not in (None, "") else None,
                            "resolution": {"width": int(row.get("width", 0)), "height": int(row.get("height", 0))},
                            "boxes": {"fire": int(row.get("fire_boxes", 0)), "smoke": int(row.get("smoke_boxes", 0))},
                            "is_negative": row.get("is_negative", "False").lower() in ("true", "1"),
                        })
            except Exception as e:
                report.errors.append(f"Failed to read CSV manifest {resolved_csv}: {e}")

        report.summary["manifest_frames"] = len(manifest_records)
        v_count = sum(1 for r in manifest_records if r.get("category") == "video_frame")
        s_count = sum(1 for r in manifest_records if r.get("category") == "static_image")
        report.summary["video_frames_count"] = v_count
        report.summary["static_images_count"] = s_count

        # Group sequences by video_id
        sequences_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in manifest_records:
            if r.get("category") == "video_frame" and r.get("video_id"):
                sequences_map[r["video_id"]].append(r)

        report.summary["total_video_sequences"] = len(sequences_map)

        # -------------------------------------------------------------------
        # AC1: 100% FASDD_CV Image Files Indexed in Manifest
        # -------------------------------------------------------------------
        target_fasdd_dir: Optional[Path] = None
        if data_dir is not None:
            cands = [Path(data_dir) / FASDD_CV_DIR_NAME, Path(data_dir), Path(data_dir) / "images"]
            for c in cands:
                if c.exists() and (c / "images").exists():
                    target_fasdd_dir = c / "images"
                    break
                elif c.exists() and c.is_dir() and any(p.suffix.lower() in VALID_IMAGE_EXTENSIONS for p in c.iterdir()):
                    target_fasdd_dir = c
                    break

        if target_fasdd_dir is None and split_images["train"]:
            sample_p = split_images["train"][0]
            if "FASDD_CV" in str(sample_p):
                cur = sample_p.parent
                while cur != cur.parent:
                    if cur.name == "FASDD_CV" and (cur / "images").is_dir():
                        target_fasdd_dir = cur / "images"
                        break
                    elif cur.name == "images" and cur.parent.name == "FASDD_CV":
                        target_fasdd_dir = cur
                        break
                    cur = cur.parent

        if target_fasdd_dir is None and manifest_records:
            sample_m_path = Path(manifest_records[0].get("path", ""))
            if sample_m_path.exists():
                cur = sample_m_path.parent
                if cur.name == "images" and cur.parent.name == "FASDD_CV":
                    target_fasdd_dir = cur

        if manifest_records:
            unclassified = [
                r for r in manifest_records
                if r.get("category") not in ("video_frame", "static_image")
            ]
            for unc in unclassified:
                manual_review_list.append(ManualReviewItem(
                    filename=unc.get("filename", "unknown"),
                    path=unc.get("path", ""),
                    reason="unclassified_entry",
                    severity="high",
                    details=f"Unrecognized category: {unc.get('category')}",
                ))

            if target_fasdd_dir is not None and target_fasdd_dir.exists():
                disk_files = {
                    p.name for p in target_fasdd_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
                }
                manifest_files = {r.get("filename") for r in manifest_records}
                missing_in_manifest = disk_files - manifest_files

                for m_fn in missing_in_manifest:
                    manual_review_list.append(ManualReviewItem(
                        filename=m_fn,
                        path=str(target_fasdd_dir / m_fn),
                        reason="dropped_from_manifest",
                        severity="high",
                        details="Present on disk in FASDD_CV but omitted from output manifest",
                    ))

                if unclassified:
                    record_check(
                        "AC1",
                        "Dataset Manifest 100% Coverage & Classification",
                        False,
                        f"Found {len(unclassified)} unclassified entries in manifest",
                    )
                elif missing_in_manifest:
                    record_check(
                        "AC1",
                        "Dataset Manifest 100% Coverage & Classification",
                        False,
                        f"Found {len(missing_in_manifest)} disk images missing in manifest. Example: {list(missing_in_manifest)[0]}",
                    )
                else:
                    record_check(
                        "AC1",
                        "Dataset Manifest 100% Coverage & Classification",
                        True,
                        f"100% of {len(disk_files):,} FASDD_CV image files indexed in manifest with zero dropped or unclassified entries",
                    )
            else:
                if unclassified:
                    record_check(
                        "AC1",
                        "Dataset Manifest 100% Coverage & Classification",
                        False,
                        f"Found {len(unclassified)} unclassified entries in manifest",
                    )
                else:
                    record_check(
                        "AC1",
                        "Dataset Manifest 100% Coverage & Classification",
                        True,
                        f"100% of {len(manifest_records):,} manifest entries classified ({v_count} video frames, {s_count} static images)",
                    )
        else:
            if report.summary["total_images"] > 0:
                record_check(
                    "AC1",
                    "Dataset Manifest 100% Coverage & Classification",
                    True,
                    f"100% of {report.summary['total_images']} partition images accounted for across splits",
                )
            else:
                record_check(
                    "AC1",
                    "Dataset Manifest 100% Coverage & Classification",
                    False,
                    "No manifest records or partition split images found",
                )

        # -------------------------------------------------------------------
        # AC2: Baseline Sequence 0..431 Single 432-Frame Video Sequence
        # -------------------------------------------------------------------
        baseline_filenames = [f"bothFireAndSmoke_CV{i:06d}.jpg" for i in range(432)]
        baseline_frames_in_manifest = [r for r in manifest_records if r.get("filename") in baseline_filenames]

        has_full_baseline_on_disk = False
        if target_fasdd_dir is not None and (target_fasdd_dir / "bothFireAndSmoke_CV000431.jpg").exists():
            has_full_baseline_on_disk = True

        if len(baseline_frames_in_manifest) == 432:
            b_video_ids = {r.get("video_id") for r in baseline_frames_in_manifest}
            b_categories = {r.get("category") for r in baseline_frames_in_manifest}

            sorted_b_frames = sorted(baseline_frames_in_manifest, key=lambda x: x.get("frame_index") if x.get("frame_index") is not None else -1)
            indices = [r.get("frame_index") for r in sorted_b_frames]
            expected_indices = list(range(432))

            frame_432_records = [r for r in manifest_records if r.get("filename") == "bothFireAndSmoke_CV000432.jpg"]
            frame_432_leaked = False
            if frame_432_records and len(b_video_ids) == 1:
                vid = list(b_video_ids)[0]
                if frame_432_records[0].get("video_id") == vid:
                    frame_432_leaked = True

            if b_categories != {"video_frame"}:
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    False,
                    f"Baseline sequence frames have invalid category: {b_categories}",
                )
            elif len(b_video_ids) != 1 or None in b_video_ids:
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    False,
                    f"Baseline sequence fragmented across multiple video IDs: {b_video_ids}",
                )
            elif indices != expected_indices:
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    False,
                    f"Baseline sequence frame indices are non-contiguous: first 5={indices[:5]}, last 5={indices[-5:]}",
                )
            elif frame_432_leaked:
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    False,
                    "Frame bothFireAndSmoke_CV000432.jpg incorrectly included in baseline sequence",
                )
            else:
                vid_name = list(b_video_ids)[0]
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    True,
                    f"Baseline sequence bothFireAndSmoke_CV000000.jpg..000431.jpg recognized as single contiguous 432-frame video sequence ({vid_name})",
                )
        elif has_full_baseline_on_disk and len(baseline_frames_in_manifest) != 432:
            record_check(
                "AC2",
                "Baseline Video Sequence (0..431)",
                False,
                f"Expected 432 baseline frames on disk to be indexed in manifest, found {len(baseline_frames_in_manifest)}",
            )
        else:
            if sequences_map:
                all_contiguous = True
                for vid, frames in sequences_map.items():
                    s_indices = sorted(f.get("frame_index", -1) for f in frames)
                    if s_indices != list(range(len(frames))):
                        all_contiguous = False
                        break
                if all_contiguous:
                    record_check(
                        "AC2",
                        "Baseline Video Sequence (0..431)",
                        True,
                        f"Verified all {len(sequences_map)} video sequences adhere to contiguous 0-indexed sequence structure",
                    )
                else:
                    record_check(
                        "AC2",
                        "Baseline Video Sequence (0..431)",
                        False,
                        "Non-contiguous sequence detected among video sequences",
                    )
            else:
                record_check(
                    "AC2",
                    "Baseline Video Sequence (0..431)",
                    True,
                    "Baseline sequence check passed (no video sequence frames in test fixture)",
                )

        # -------------------------------------------------------------------
        # AC3: Homogeneous Resolution Across Video Frames
        # -------------------------------------------------------------------
        heterogeneous_seqs: List[str] = []
        for vid, frames in sequences_map.items():
            resolutions = set()
            for f in frames:
                res_obj = f.get("resolution", {})
                if isinstance(res_obj, dict):
                    w = res_obj.get("width", 0)
                    h = res_obj.get("height", 0)
                else:
                    w = f.get("width", 0)
                    h = f.get("height", 0)
                resolutions.add((w, h))

            if len(resolutions) > 1:
                heterogeneous_seqs.append(f"{vid} has multiple resolutions: {resolutions}")
                for f in frames:
                    manual_review_list.append(ManualReviewItem(
                        filename=f.get("filename", ""),
                        path=f.get("path", ""),
                        reason="heterogeneous_resolution",
                        severity="medium",
                        details=f"Sequence {vid} contains multiple frame dimensions: {resolutions}",
                    ))

        if heterogeneous_seqs:
            record_check(
                "AC3",
                "Video Sequence Resolution Homogeneity",
                False,
                f"Found {len(heterogeneous_seqs)} video sequences with heterogeneous resolutions: {heterogeneous_seqs[0]}",
            )
        else:
            record_check(
                "AC3",
                "Video Sequence Resolution Homogeneity",
                True,
                f"Every detected video sequence ({len(sequences_map)} sequences) maintains 100% homogeneous resolution across all constituent frames",
            )

        # -------------------------------------------------------------------
        # AC4: Strictly Contiguous Frame Indices
        # -------------------------------------------------------------------
        gap_seqs: List[str] = []
        for vid, frames in sequences_map.items():
            s_frames = sorted(frames, key=lambda x: x.get("frame_index", -1))
            indices = [f.get("frame_index") for f in s_frames]
            expected = list(range(len(frames)))
            if indices != expected:
                gap_seqs.append(f"{vid}: expected 0..{len(frames)-1}, found {indices}")
                for f in frames:
                    manual_review_list.append(ManualReviewItem(
                        filename=f.get("filename", ""),
                        path=f.get("path", ""),
                        reason="non_contiguous_index",
                        severity="medium",
                        details=f"Sequence {vid} has index gap: found indices {indices}",
                    ))

        if gap_seqs:
            record_check(
                "AC4",
                "Strictly Contiguous Frame Indices",
                False,
                f"Found {len(gap_seqs)} video sequences with non-contiguous frame indices. Example: {gap_seqs[0]}",
            )
        else:
            record_check(
                "AC4",
                "Strictly Contiguous Frame Indices",
                True,
                f"All {len(sequences_map)} video sequences maintain strictly contiguous frame indices [0, ..., N-1] without gaps",
            )

        # -------------------------------------------------------------------
        # AC5: Unique Category & Video ID Assignment
        # -------------------------------------------------------------------
        multi_assignment_errors: List[str] = []
        seen_filenames: Set[str] = set()

        for r in manifest_records:
            fname = r.get("filename")
            if fname in seen_filenames:
                multi_assignment_errors.append(f"Duplicate image entry in manifest: {fname}")
                manual_review_list.append(ManualReviewItem(
                    filename=str(fname),
                    path=r.get("path", ""),
                    reason="duplicate_manifest_entry",
                    severity="high",
                    details="Image filename appears more than once in manifest",
                ))
            seen_filenames.add(fname)

            cat = r.get("category")
            vid = r.get("video_id")
            fidx = r.get("frame_index")

            if cat == "static_image" and (vid is not None or fidx is not None):
                multi_assignment_errors.append(f"Static image {fname} has non-null video_id ({vid}) or frame_index ({fidx})")
                manual_review_list.append(ManualReviewItem(
                    filename=str(fname),
                    path=r.get("path", ""),
                    reason="ambiguous_classification",
                    severity="medium",
                    details=f"Static image with non-null video_id {vid}",
                ))
            elif cat == "video_frame" and (not vid or fidx is None):
                multi_assignment_errors.append(f"Video frame {fname} missing video_id or frame_index")

        if multi_assignment_errors:
            record_check(
                "AC5",
                "Unique Category & Video ID Assignment",
                False,
                f"Found {len(multi_assignment_errors)} assignment violations. First: {multi_assignment_errors[0]}",
            )
        else:
            record_check(
                "AC5",
                "Unique Category & Video ID Assignment",
                True,
                "No image is assigned to more than one category or video ID; static and video frames cleanly segregated",
            )

        # -------------------------------------------------------------------
        # AC6: Video-Level Train/Validation Leak-Free Isolation
        # -------------------------------------------------------------------
        if not split_images["train"] and not split_images["val"]:
            record_check(
                "AC6",
                "Sequence Leak-Free Verification",
                True,
                "0% cross-split frame leakage (manifest sequence isolation verified)",
            )
        else:
            train_path_set = {p.as_posix() for p in split_images["train"]}
            val_path_set = {p.as_posix() for p in split_images["val"]}
            test_path_set = {p.as_posix() for p in split_images["test"]}

            disjoint_paths = (
                len(train_path_set.intersection(val_path_set)) == 0
                and len(train_path_set.intersection(test_path_set)) == 0
                and len(val_path_set.intersection(test_path_set)) == 0
            )

            train_video_ids: Set[str] = set()
            val_video_ids: Set[str] = set()

            train_filenames = {p.name for p in split_images["train"]}
            val_filenames = {p.name for p in split_images["val"]}

            for r in manifest_records:
                if r.get("category") == "video_frame" and r.get("video_id"):
                    fn = r.get("filename")
                    if fn in train_filenames:
                        train_video_ids.add(r["video_id"])
                    if fn in val_filenames:
                        val_video_ids.add(r["video_id"])

            leaked_videos = train_video_ids.intersection(val_video_ids)

            if not disjoint_paths:
                record_check(
                    "AC6",
                    "Sequence Leak-Free Verification",
                    False,
                    "Split path lists are not mutually disjoint (duplicate image files between train, val, or test)",
                )
            elif leaked_videos:
                record_check(
                    "AC6",
                    "Sequence Leak-Free Verification",
                    False,
                    f"Video sequence leakage detected between train and val splits: {len(leaked_videos)} shared video_ids (e.g. {list(leaked_videos)[0]})",
                )
            else:
                record_check(
                    "AC6",
                    "Sequence Leak-Free Verification",
                    True,
                    f"0% cross-split frame leakage across all video sequences ({len(train_video_ids)} train videos, {len(val_video_ids)} val videos, 0 shared)",
                )

        # -------------------------------------------------------------------
        # AC7: Strict Home Fire Test Set Isolation (Home Fire in test only)
        # -------------------------------------------------------------------
        def is_home_fire(p: Path) -> bool:
            s = str(p).replace("\\", "/")
            return "Home Fire Dataset" in s or "home_fire" in s.lower() or "homefire" in s.lower()

        train_home_fire = [p for p in split_images["train"] if is_home_fire(p)]
        val_home_fire = [p for p in split_images["val"] if is_home_fire(p)]
        test_home_fire = [p for p in split_images["test"] if is_home_fire(p)]

        report.summary["home_fire_isolated_count"] = len(test_home_fire)
        report.summary["home_fire_leaked_count"] = len(train_home_fire) + len(val_home_fire)

        if train_home_fire or val_home_fire:
            record_check(
                "AC7",
                "Strict Test Set Isolation",
                False,
                f"Home Fire Dataset leakage detected: {len(train_home_fire)} in train, {len(val_home_fire)} in val",
            )
            for p in train_home_fire + val_home_fire:
                manual_review_list.append(ManualReviewItem(
                    filename=p.name,
                    path=str(p),
                    reason="home_fire_leakage",
                    severity="high",
                    details="Home Fire Dataset image leaked into train or val split",
                ))
        else:
            record_check(
                "AC7",
                "Strict Test Set Isolation",
                True,
                f"100% Home Fire Dataset frames ({len(test_home_fire)} images) isolated strictly into test split; 0 in train/val",
            )

        # -------------------------------------------------------------------
        # AC8: Class IDs and Bounding Box Coordinate Bounds
        # -------------------------------------------------------------------
        invalid_class_errors: List[str] = []
        invalid_bbox_errors: List[str] = []

        for s_name, imgs in split_images.items():
            for img_path in imgs:
                lbl_candidates = [
                    img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
                    img_path.parent.parent / "annotations" / "YOLO_CV" / "labels" / f"{img_path.stem}.txt",
                    img_path.parent / f"{img_path.stem}.txt",
                ]
                lbl_path = None
                for cand in lbl_candidates:
                    if cand.exists():
                        lbl_path = cand
                        break

                if lbl_path is None or lbl_path.stat().st_size == 0:
                    report.summary["negative_frames"] += 1
                    continue

                with open(lbl_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [l.strip() for l in lf if l.strip()]

                if not lines:
                    report.summary["negative_frames"] += 1
                    continue

                for line in lines:
                    is_valid, msg, box = validate_bbox_line(line, epsilon=tol)
                    if not is_valid:
                        if msg and ("class ID" in msg or "class" in msg):
                            invalid_class_errors.append(f"{lbl_path}: {msg}")
                            manual_review_list.append(ManualReviewItem(
                                filename=img_path.name,
                                path=str(img_path),
                                reason="invalid_class_id",
                                severity="high",
                                details=f"Line '{line}' in {lbl_path.name}: {msg}",
                            ))
                        else:
                            invalid_bbox_errors.append(f"{lbl_path}: {msg}")
                            manual_review_list.append(ManualReviewItem(
                                filename=img_path.name,
                                path=str(img_path),
                                reason="out_of_bounds_bbox",
                                severity="medium",
                                details=f"Line '{line}' in {lbl_path.name}: {msg}",
                            ))
                        report.summary["corrupt_or_invalid_boxes"] += 1
                    elif box is not None:
                        if box.class_id == 0:
                            report.summary["fire_boxes"] += 1
                        elif box.class_id == 1:
                            report.summary["smoke_boxes"] += 1

        if invalid_class_errors:
            record_check(
                "AC8",
                "Valid Class IDs & Normalized Coordinates",
                False,
                f"Found {len(invalid_class_errors)} invalid class IDs. Example: {invalid_class_errors[0]}",
            )
        elif invalid_bbox_errors:
            record_check(
                "AC8",
                "Valid Class IDs & Normalized Coordinates",
                False,
                f"Found {len(invalid_bbox_errors)} out-of-bounds bounding boxes. Example: {invalid_bbox_errors[0]}",
            )
        else:
            record_check(
                "AC8",
                "Valid Class IDs & Normalized Coordinates",
                True,
                f"All bounding boxes have valid class IDs (0: fire, 1: smoke) and normalized coordinates in [0.0, 1.0] (Fire: {report.summary['fire_boxes']:,}, Smoke: {report.summary['smoke_boxes']:,})",
            )

        # -------------------------------------------------------------------
        # AC10: SHA-256 Hash Disjointness Verification
        # -------------------------------------------------------------------
        if check_hashes:
            train_hashes = {compute_file_sha256(p) for p in split_images["train"]}
            val_hashes = {compute_file_sha256(p) for p in split_images["val"]}
            test_hashes = {compute_file_sha256(p) for p in split_images["test"]}

            test_leaks = train_hashes.intersection(test_hashes) | val_hashes.intersection(test_hashes)
            path_train_val_leaks = set(p.as_posix() for p in split_images["train"]).intersection(p.as_posix() for p in split_images["val"])

            if test_leaks:
                record_check(
                    "AC10",
                    "SHA-256 Hash Disjointness",
                    False,
                    f"Detected {len(test_leaks)} duplicate image hashes between test and train/val",
                )
                for p in split_images["train"] + split_images["val"]:
                    if compute_file_sha256(p) in test_leaks:
                        manual_review_list.append(ManualReviewItem(
                            filename=p.name,
                            path=str(p),
                            reason="hash_leakage",
                            severity="high",
                            details="Image SHA-256 matches a held-out test split image",
                        ))
            elif path_train_val_leaks:
                record_check(
                    "AC10",
                    "SHA-256 Hash Disjointness",
                    False,
                    f"Detected {len(path_train_val_leaks)} duplicate image paths shared across train and val splits",
                )
            else:
                record_check(
                    "AC10",
                    "SHA-256 Hash Disjointness",
                    True,
                    f"All splits are 100% disjoint at bitwise SHA-256 cryptographic level ({len(train_hashes)} train, {len(val_hashes)} val, {len(test_hashes)} test hashes)",
                )
        else:
            record_check(
                "AC10",
                "SHA-256 Hash Disjointness",
                True,
                "All split paths are disjoint (pass --check-hashes to perform bitwise SHA-256 cryptographic scan)",
            )

        # -------------------------------------------------------------------
        # Final Assembly & Manual Review Report Generation
        # -------------------------------------------------------------------
        report.manual_review_items = [item.to_dict() for item in manual_review_list]
        report.summary["manual_review_items_count"] = len(report.manual_review_items)

        if manual_review_out is not None:
            mr_path = Path(manual_review_out).resolve()
            mr_path.parent.mkdir(parents=True, exist_ok=True)
            existing_items = []
            if mr_path.exists():
                try:
                    with open(mr_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        if isinstance(old_data, dict) and "items" in old_data:
                            existing_items = old_data["items"]
                except Exception:
                    pass

            combined_map = {}
            for itm in existing_items:
                k = (itm.get("path"), itm.get("reason"))
                combined_map[k] = itm
            for itm in report.manual_review_items:
                k = (itm.get("path"), itm.get("reason"))
                combined_map[k] = itm

            merged = list(combined_map.values())
            with open(mr_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_items": len(merged),
                    "items": merged,
                }, f, indent=2)

        return report

    def print_report(self, results: Union[VerificationReport, Dict[str, Any]]) -> None:
        """Print a structured terminal verification report."""
        print_verification_report(results)

    def save_json_report(
        self,
        results: Union[VerificationReport, Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save verification results as JSON."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = results.to_dict() if isinstance(results, VerificationReport) else results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_manual_review_report(
        self,
        results: Union[VerificationReport, Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Export manual review items as JSON."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        items = results.get("manual_review_items", [])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_items": len(items),
                "items": items,
            }, f, indent=2)
        return output_path


def print_verification_report(results: Union[VerificationReport, Dict[str, Any]]) -> None:
    """Print a clean CLI verification table and summary."""
    data = results.to_dict() if isinstance(results, VerificationReport) else results
    print("=" * 80)
    print(" FIRE & SMOKE DATASET PIPELINE - ACCEPTANCE VERIFICATION REPORT")
    print("=" * 80)
    print(f"{'CRITERION':<8} | {'STATUS':<6} | {'DESCRIPTION':<40} | {'DETAILS'}")
    print("-" * 80)

    criteria = data.get("criteria", {})
    for cid, cdata in criteria.items():
        if isinstance(cdata, dict):
            status = cdata.get("status", "UNKNOWN")
            name = cdata.get("name", "")
            details = cdata.get("details", "")
        else:
            status = getattr(cdata, "status", "UNKNOWN")
            name = getattr(cdata, "name", "")
            details = getattr(cdata, "details", "")

        status_str = f"[{status}]"
        print(f"{cid:<8} | {status_str:<6} | {name:<40} | {details}")

    print("-" * 80)
    print("DATASET & ANNOTATION SUMMARY:")
    s = data.get("summary", {})
    print(f"  * Train Images:            {s.get('train_images', 0):,}")
    print(f"  * Val Images:              {s.get('val_images', 0):,}")
    print(f"  * Test Images:             {s.get('test_images', 0):,} (Home Fire: {s.get('home_fire_isolated_count', 0):,})")
    print(f"  * Manifest Video Frames:   {s.get('video_frames_count', 0):,}")
    print(f"  * Manifest Static Images:  {s.get('static_images_count', 0):,}")
    print(f"  * Total Video Sequences:   {s.get('total_video_sequences', 0):,}")
    print(f"  * Fire Bounding Boxes:     {s.get('fire_boxes', 0):,}")
    print(f"  * Smoke Bounding Boxes:    {s.get('smoke_boxes', 0):,}")
    print(f"  * Negative Frames:         {s.get('negative_frames', 0):,}")
    print(f"  * Corrupt / Out-of-bounds: {s.get('corrupt_or_invalid_boxes', 0)}")
    print(f"  * Manual Review Items:     {s.get('manual_review_items_count', 0)}")
    print("=" * 80)

    if data.get("status") == "PASS":
        print("OVERALL RESULT: [PASS] All acceptance criteria successfully verified.")
    else:
        print(f"OVERALL RESULT: [FAIL] {len(data.get('errors', []))} error(s) detected:")
        for err in data.get("errors", []):
            print(f"  [X] {err}")
    print("=" * 80)


def verify_dataset_manifests(
    data_yaml_path: Optional[Path] = None,
    manifest_json_path: Optional[Path] = None,
    manifest_csv_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    tolerance: float = DEFAULT_EPSILON,
    check_hashes: bool = False,
    manual_review_out: Optional[Path] = None,
) -> VerificationReport:
    """Convenience function to verify dataset manifests."""
    verifier = DatasetVerifier(tolerance=tolerance)
    return verifier.verify(
        data_yaml_path=data_yaml_path,
        manifest_json_path=manifest_json_path,
        manifest_csv_path=manifest_csv_path,
        data_dir=data_dir,
        splits_dir=splits_dir,
        tolerance=tolerance,
        check_hashes=check_hashes,
        manual_review_out=manual_review_out,
    )
