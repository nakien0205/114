"""
Manifest writer module for serializing video sequence segmentation metadata.

Exports structured manifests in both JSON and CSV formats for downstream consumers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from src.fire_audit.config import SegmentedFrame, SegmentationResult
except ImportError:
    from fire_audit.config import SegmentedFrame, SegmentationResult



class ManifestWriter:
    """
    Serializes SegmentationResult and SegmentedFrame collections into standardized
    JSON and CSV metadata manifests.
    """

    CSV_FIELDNAMES: List[str] = [
        "filename",
        "path",
        "category",
        "video_id",
        "frame_index",
        "sequence_length",
        "width",
        "height",
        "fire_boxes",
        "smoke_boxes",
        "is_negative",
    ]

    def __init__(self, dataset_name: str = "dataset") -> None:
        self.dataset_name = dataset_name

    def write_json(
        self,
        records: Sequence[Union[SegmentedFrame, Dict[str, Any]]],
        output_path: Union[Path, str],
        dataset_name: Optional[str] = None,
    ) -> Path:
        """
        Export frame records to structured JSON manifest.

        Schema:
        {
             "dataset": "dataset",
            "total_images": int,
            "summary": {
                "video_frames": int,
                "static_images": int,
                "total_video_sequences": int
            },
            "frames": [
                {
                    "filename": str,
                    "path": str (POSIX forward slashes),
                    "category": "video_frame" | "static_image",
                    "video_id": str | null,
                    "frame_index": int | null,
                    "sequence_length": int | null,
                    "resolution": {"width": int, "height": int},
                    "boxes": {"fire": int, "smoke": int},
                    "is_negative": bool
                }, ...
            ]
        }
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ds_name = dataset_name or self.dataset_name

        frame_dicts: List[Dict[str, Any]] = []
        video_ids: set[str] = set()
        video_count = 0
        static_count = 0

        for r in records:
            if isinstance(r, SegmentedFrame) or (hasattr(r, "to_dict") and hasattr(r, "category")):
                d = r.to_dict()
            elif isinstance(r, dict):
                # Normalize dict representation
                path_str = str(r.get("path", "")).replace("\\", "/")
                cat = r.get("category", "static_image")
                vid_id = r.get("video_id")
                f_idx = r.get("frame_index")
                seq_len = r.get("sequence_length")
                w = int(r.get("width", 0))
                h = int(r.get("height", 0))
                fb = int(r.get("fire_boxes", 0))
                sb = int(r.get("smoke_boxes", 0))
                is_neg = bool(r.get("is_negative", (fb == 0 and sb == 0)))

                # Support nested resolution and boxes if provided
                if "resolution" in r and isinstance(r["resolution"], dict):
                    w = int(r["resolution"].get("width", w))
                    h = int(r["resolution"].get("height", h))
                if "boxes" in r and isinstance(r["boxes"], dict):
                    fb = int(r["boxes"].get("fire", fb))
                    sb = int(r["boxes"].get("smoke", sb))

                d = {
                    "filename": str(r.get("filename", Path(path_str).name if path_str else "")),
                    "path": path_str,
                    "category": cat,
                    "video_id": vid_id if vid_id else None,
                    "frame_index": f_idx if f_idx is not None else None,
                    "sequence_length": seq_len if seq_len is not None else None,
                    "resolution": {"width": w, "height": h},
                    "boxes": {"fire": fb, "smoke": sb},
                    "is_negative": is_neg,
                }
            else:
                continue

            frame_dicts.append(d)
            if d["category"] == "video_frame":
                video_count += 1
                if d["video_id"]:
                    video_ids.add(d["video_id"])
            else:
                static_count += 1

        manifest_data = {
            "dataset": ds_name,
            "total_images": len(frame_dicts),
            "summary": {
                "video_frames": video_count,
                "static_images": static_count,
                "total_video_sequences": len(video_ids),
            },
            "frames": frame_dicts,
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return out_p

    def write_csv(
        self,
        records: Sequence[Union[SegmentedFrame, Dict[str, Any]]],
        output_path: Union[Path, str],
    ) -> Path:
        """
        Export frame records to flat CSV manifest.

        Header:
        filename,path,category,video_id,frame_index,sequence_length,width,height,fire_boxes,smoke_boxes,is_negative
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
            writer.writeheader()

            for r in records:
                if isinstance(r, SegmentedFrame) or (hasattr(r, "to_csv_row") and hasattr(r, "category")):
                    row = r.to_csv_row()
                elif isinstance(r, dict):
                    path_str = str(r.get("path", "")).replace("\\", "/")
                    w = int(r.get("width", 0))
                    h = int(r.get("height", 0))
                    fb = int(r.get("fire_boxes", 0))
                    sb = int(r.get("smoke_boxes", 0))
                    if "resolution" in r and isinstance(r["resolution"], dict):
                        w = int(r["resolution"].get("width", w))
                        h = int(r["resolution"].get("height", h))
                    if "boxes" in r and isinstance(r["boxes"], dict):
                        fb = int(r["boxes"].get("fire", fb))
                        sb = int(r["boxes"].get("smoke", sb))

                    f_idx = r.get("frame_index")
                    seq_len = r.get("sequence_length")
                    is_neg = r.get("is_negative")
                    if is_neg is None:
                        is_neg = (fb == 0 and sb == 0)

                    row = {
                        "filename": str(r.get("filename", Path(path_str).name if path_str else "")),
                        "path": path_str,
                        "category": str(r.get("category", "static_image")),
                        "video_id": str(r.get("video_id") or ""),
                        "frame_index": f_idx if f_idx is not None else "",
                        "sequence_length": seq_len if seq_len is not None else "",
                        "width": w,
                        "height": h,
                        "fire_boxes": fb,
                        "smoke_boxes": sb,
                        "is_negative": bool(is_neg),
                    }
                else:
                    continue

                writer.writerow(row)

        return out_p

    def write_manifests(
        self,
        result: SegmentationResult,
        output_dir: Union[Path, str],
        dataset_name: Optional[str] = None,
        base_name: str = "dataset_manifest",
    ) -> Tuple[Path, Path]:
        """
        Write dual JSON and CSV manifests for a complete SegmentationResult.
        Updates result.manifest_json_path and result.manifest_csv_path in place.
        """
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)
        ds_name = dataset_name or self.dataset_name

        json_path = out_d / f"{base_name}.json"
        csv_path = out_d / f"{base_name}.csv"

        self.write_json(result.records, json_path, dataset_name=ds_name)
        self.write_csv(result.records, csv_path)

        result.manifest_json_path = json_path
        result.manifest_csv_path = csv_path

        return json_path, csv_path


def export_manifest_json(
    records: Sequence[Union[SegmentedFrame, Dict[str, Any]]],
    output_path: Union[Path, str],
    dataset_name: str = "dataset",
) -> Path:
    """Convenience function to write a JSON manifest."""
    writer = ManifestWriter(dataset_name=dataset_name)
    return writer.write_json(records=records, output_path=output_path)


def export_manifest_csv(
    records: Sequence[Union[SegmentedFrame, Dict[str, Any]]],
    output_path: Union[Path, str],
) -> Path:
    """Convenience function to write a CSV manifest."""
    writer = ManifestWriter()
    return writer.write_csv(records=records, output_path=output_path)


def export_manifests(
    result: SegmentationResult,
    output_dir: Union[Path, str],
    dataset_name: str = "dataset",
    base_name: str = "dataset_manifest",
) -> Tuple[Path, Path]:
    """Convenience function to write dual JSON and CSV manifests from SegmentationResult."""
    writer = ManifestWriter(dataset_name=dataset_name)
    return writer.write_manifests(result=result, output_dir=output_dir, base_name=base_name)
