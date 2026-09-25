#!/usr/bin/env python3
"""
Standalone Dataset Acceptance Verification Runner for Fire and Smoke Detection Pipeline.
Validates:
- YOLO dataset config (data.yaml) resolution (train, val, test splits).
- 100% census check on image file existence and integrity.
- 100% census check on YOLO label files and annotations (classes 0: fire, 1: smoke).
- Normalized coordinate bounds [0.0, 1.0] and non-zero dimensions.
- Per-split summary metrics and class distribution.
- Optional AC1-AC10 audit mode for manifest-level sequence checks.

Usage:
    python verify_dataset.py [--data-yaml data.yaml] [--json-out report.json]
    python verify_dataset.py --audit [--manifest-json manifests/fasdd_cv_manifest.json] [--check-hashes]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root and src/ are on python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from src.fire_detection.dataset import DatasetVerifier, verify_dataset
except ImportError:
    from fire_detection.dataset import DatasetVerifier, verify_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for dataset verification."""
    parser = argparse.ArgumentParser(
        prog="verify_dataset",
        description="Programmatic Dataset Verifier for Fire and Smoke Detection (R1 Acceptance Check)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data.yaml"),
        help="Path to data.yaml dataset specification (default: data.yaml)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Coordinate boundary epsilon tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output verification report as JSON",
    )
    parser.add_argument(
        "--check-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Perform image file header / integrity checks (default: True, disable with --no-check-images)",
    )
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=None,
        help="Optional limit on number of image integrity checks for faster verification",
    )
    # Legacy / audit compatibility flags
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run legacy multi-criteria audit verifier (AC1-AC10)",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Path to fasdd_cv_manifest.json (optional for audit mode)",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Path to fasdd_cv_manifest.csv (optional for audit mode)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset root for 100%% census verification (optional for audit mode)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory containing split text files (optional for audit mode)",
    )
    parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Perform bitwise SHA-256 duplicate verification across splits",
    )
    parser.add_argument(
        "--manual-review-out",
        type=Path,
        default=None,
        help="Optional path to output manual review items JSON (audit mode)",
    )
    return parser


def run_detection_verification(args: argparse.Namespace) -> int:
    """Run primary YOLO dataset verification for Home Fire Dataset (R1)."""
    verifier = DatasetVerifier(tolerance=args.tolerance)
    print("=" * 70)
    print(" FIRE & SMOKE DATASET VERIFICATION (R1)")
    print("=" * 70)
    print(f"Dataset config: {args.data_yaml}")

    try:
        result = verifier.verify(
            data_yaml_path=args.data_yaml,
            check_images_readable=args.check_images,
            max_image_checks=args.max_image_checks,
        )
    except Exception as e:
        print(f"\n[FAIL] Dataset verification encountered fatal error: {e}")
        return 1

    print(f"\nStatus: {result.status}")
    print(f"Dataset Root: {result.dataset_path}")
    print(f"Number of Classes: {result.num_classes} ({result.class_names})")
    print(f"Total Images: {result.total_images:,}")
    print(f"Total Bounding Boxes: {result.total_boxes:,} (Fire: {result.total_fire_boxes:,}, Smoke: {result.total_smoke_boxes:,})")
    print(f"Total Background (Empty) Images: {result.total_empty_images:,}")

    print("\nSplit Details:")
    for s_name, s_stat in result.splits.items():
        print(
            f"  - Split '{s_name}': {s_stat.images_count:,} images, {s_stat.labels_count:,} labels, "
            f"{s_stat.boxes_count:,} boxes "
            f"(fire: {s_stat.class_counts.get('fire', 0):,}, smoke: {s_stat.class_counts.get('smoke', 0):,}, "
            f"empty: {s_stat.empty_images_count:,})"
        )
        if s_stat.errors:
            print(f"    Errors ({len(s_stat.errors)}): {s_stat.errors[:3]}")

    if result.errors:
        print(f"\n[FAIL] Found {len(result.errors)} total verification errors:")
        for err in result.errors[:10]:
            print(f"  - {err}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more errors")
    else:
        print("\n[PASS] All images and label annotations verified successfully with 0 errors.")

    if args.json_out:
        out_p = Path(args.json_out).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved verification JSON report to: {out_p}")

    print("=" * 70)
    return 0 if result.status == "PASS" else 1


def run_audit_verification(args: argparse.Namespace) -> int:
    """Run legacy multi-criteria audit verifier."""
    try:
        from src.fire_audit.verify.verifier import DatasetVerifier as AuditVerifier
    except ImportError:
        from fire_audit.verify.verifier import DatasetVerifier as AuditVerifier

    verifier = AuditVerifier(tolerance=args.tolerance)
    report = verifier.verify(
        data_yaml_path=args.data_yaml,
        manifest_json_path=args.manifest_json,
        manifest_csv_path=args.manifest_csv,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        tolerance=args.tolerance,
        check_hashes=args.check_hashes,
        manual_review_out=args.manual_review_out,
    )
    verifier.print_report(report)

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        verifier.save_json_report(report, out_path)
        print(f"Verification JSON report saved to: {out_path}")

    return 0 if report.status == "PASS" else 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main verification entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if (
        args.audit
        or args.check_hashes
        or args.manifest_json is not None
        or args.manifest_csv is not None
        or args.data_dir is not None
        or args.splits_dir is not None
    ):
        return run_audit_verification(args)
    else:
        return run_detection_verification(args)


if __name__ == "__main__":
    sys.exit(main())
