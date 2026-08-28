"""
Unified Command-Line Interface (CLI) for the Fire and Smoke Dataset Pipeline.
Supports subcommands: audit, segment, prepare, verify, all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from src.fire_audit.config import (
        DEFAULT_EPSILON,
        get_configured_data_root,
        MANIFEST_DIR_NAME,
        SPLITS_DIR_NAME,
        SegmentationConfig,
        SegmentationResult,
    )
    from src.fire_audit.audit.scanner import (
        DatasetScanner,
    )
    from src.fire_audit.audit.stats import calculate_dataset_stats
    from src.fire_audit.audit.reporter import (
        ReportGenerator,
        save_reports,
    )
    from src.fire_audit.segment.engine import VideoSegmentationEngine
    from src.fire_audit.segment.manifest_writer import ManifestWriter
    from src.fire_audit.prepare.partitioner import partition_dataset_by_sequence
    from src.fire_audit.prepare.manifest import (
        DataPreparer,
        ManifestGenerator,
        dataset_artifact_root,
        prepare_pipeline_datasets,
        resolve_dataset_roots,
    )
    from src.fire_audit.verify.verifier import (
        DatasetVerifier,
        VerificationReport,
        print_verification_report,
        verify_dataset_manifests,
    )
except ImportError:
    from fire_audit.config import (
        DEFAULT_EPSILON,
        get_configured_data_root,
        MANIFEST_DIR_NAME,
        SPLITS_DIR_NAME,
        SegmentationConfig,
        SegmentationResult,
    )
    from fire_audit.audit.scanner import (
        DatasetScanner,
    )
    from fire_audit.audit.stats import calculate_dataset_stats
    from fire_audit.audit.reporter import (
        ReportGenerator,
        save_reports,
    )
    from fire_audit.segment.engine import VideoSegmentationEngine
    from fire_audit.segment.manifest_writer import ManifestWriter
    from fire_audit.prepare.partitioner import partition_dataset_by_sequence
    from fire_audit.prepare.manifest import (
        DataPreparer,
        ManifestGenerator,
        dataset_artifact_root,
        prepare_pipeline_datasets,
        resolve_dataset_roots,
    )
    from fire_audit.verify.verifier import (
        DatasetVerifier,
        VerificationReport,
        print_verification_report,
        verify_dataset_manifests,
    )


def resolve_data_dir(data_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve a CLI data directory from an explicit value or user config."""
    configured = get_configured_data_root()
    return Path(data_dir or configured or Path.cwd()).resolve()


def run_audit(
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("."),
    epsilon: float = DEFAULT_EPSILON,
    validate_images: bool = True,
    manual_review_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the multi-dataset audit and report generation."""
    print("=" * 80)
    print(" PIPELINE STAGE: AUDIT & VALIDATION")
    print("=" * 80)
    data_dir = resolve_data_dir(data_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = []
    for root in resolve_dataset_roots(data_dir):
        if root not in dataset_roots:
            dataset_roots.append(root)
    scanner = DatasetScanner(epsilon=epsilon)
    scan_results = {}
    for dataset_root in dataset_roots:
        dataset_name = dataset_root.name or "dataset"
        print(f"[*] Scanning {dataset_name} from: {dataset_root}")
        scan_result = scanner.scan_dataset(
            dataset_root,
            dataset_name=dataset_name,
            validate_images=validate_images,
        )
        scan_results[dataset_name] = scan_result
        print(f"    -> Found {len(scan_result.records):,} records ({len(scan_result.corrupt_images)} corrupt images, {len(scan_result.corrupt_labels)} corrupt labels)")

    print("[*] Computing statistical distributions and co-occurrences...")
    stats_map = {
        name: calculate_dataset_stats(scan_result)
        for name, scan_result in scan_results.items()
    }

    print(f"[*] Writing audit reports to: {output_dir}")
    md_path, json_path = save_reports(stats_map, scan_results=scan_results, output_dir=output_dir)
    print(f"    -> Markdown report: {md_path}")
    print(f"    -> JSON report:     {json_path}")

    # Collect corrupt / invalid items for manual review
    manual_review_items = []
    for scan_result in scan_results.values():
        for reason, paths, details in (
            ("corrupt_image", scan_result.corrupt_images, "Unreadable or truncated image"),
            ("corrupt_label", scan_result.corrupt_labels, "Unparseable or out-of-bounds YOLO bounding box"),
        ):
            for path in paths:
                manual_review_items.append({
                    "filename": path.name,
                    "path": str(path).replace("\\", "/"),
                    "reason": reason,
                    "severity": "high",
                    "details": details,
                })

    mr_file = manual_review_out or (output_dir / "manual_review_needed.json")
    with open(mr_file, "w", encoding="utf-8") as f:
        json.dump({"total_items": len(manual_review_items), "items": manual_review_items}, f, indent=2)
    print(f"    -> Manual review:   {mr_file} ({len(manual_review_items)} items)")

    print("[*] Audit stage completed successfully.\n")

    return {
        "dataset_stats": {name: stats.to_dict() for name, stats in stats_map.items()},
        "md_path": str(md_path),
        "json_path": str(json_path),
        "manual_review_path": str(mr_file),
    }


def run_segment(
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("."),
    dhash_threshold: int = 18,
    min_length: int = 5,
    lookahead_k: int = 2,
    workers: int = 16,
    manual_review_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute video sequence segmentation on one dataset and export manifests."""
    print("=" * 80)
    print(" PIPELINE STAGE: VIDEO SEQUENCE SEGMENTATION & MANIFEST GENERATION")
    print("=" * 80)
    data_dir = resolve_data_dir(data_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, target_dir = resolve_dataset_roots(data_dir)
    if not target_dir.exists():
        target_dir = data_dir

    print(f"[*] Input Dataset Directory: {target_dir}")
    print(f"[*] Configuration: min_len={min_length}, dhash_thresh={dhash_threshold}, lookahead_k={lookahead_k}, workers={workers}")

    cfg = SegmentationConfig(
        min_sequence_length=min_length,
        dhash_threshold=dhash_threshold,
        lookahead_k=lookahead_k,
        max_workers=workers,
        dataset_path=target_dir,
        output_dir=output_dir,
    )
    engine = VideoSegmentationEngine(config=cfg)

    print("[*] Running two-stage segmentation orchestrator...")
    seg_res = engine.segment_dataset(target_dir)

    print(f"    -> Total images processed:   {seg_res.total_images:,}")
    print(f"    -> Video frames detected:   {seg_res.video_frames_count:,}")
    print(f"    -> Static images classified: {seg_res.static_images_count:,}")
    print(f"    -> Video sequences created:  {seg_res.total_video_sequences:,}")

    # Keep segmentation metadata alongside the dataset it describes.
    manifest_dir = dataset_artifact_root(
        target_dir,
        output_dir=output_dir,
        data_dir=data_dir,
    ) / MANIFEST_DIR_NAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(dataset_name=target_dir.name or "dataset")
    json_path, csv_path = writer.write_manifests(seg_res, manifest_dir)

    # Collect borderline / uncertain sequences and corrupt items for manual review
    manual_review_items = []
    for r in seg_res.records:
        if r.is_corrupt or r.width <= 0 or r.height <= 0:
            manual_review_items.append({
                "filename": r.filename,
                "path": str(r.path).replace("\\", "/"),
                "reason": "corrupt_label" if (r.is_corrupt and r.width > 0) else "unreadable_dimensions",
                "severity": "high",
                "details": r.error_message or "Zero or unreadable image dimensions or corrupt label",
            })

    mr_file = manual_review_out or (output_dir / "manual_review_needed.json")
    with open(mr_file, "w", encoding="utf-8") as f:
        json.dump({"total_items": len(manual_review_items), "items": manual_review_items}, f, indent=2)

    print(f"[*] Serialized dual metadata manifests:")
    print(f"    • JSON Manifest: {json_path}")
    print(f"    • CSV Manifest:  {csv_path}")
    print(f"    • Manual Review: {mr_file}")
    print("[*] Video segmentation stage completed successfully.\n")

    return {
        "total_images": seg_res.total_images,
        "video_frames_count": seg_res.video_frames_count,
        "static_images_count": seg_res.static_images_count,
        "total_video_sequences": seg_res.total_video_sequences,
        "manifest_json_path": str(json_path),
        "manifest_csv_path": str(csv_path),
        "manual_review_path": str(mr_file),
    }


def run_prepare(
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("."),
    train_ratio: float = 0.8,
    seed: int = 42,
    epsilon: float = DEFAULT_EPSILON,
    validate_images: bool = True,
    compute_hashes: bool = False,
    use_segmentation: bool = True,
    manual_review_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute dataset isolation, sequence partitioning, and manifest generation."""
    print("=" * 80)
    print(" PIPELINE STAGE: ISOLATION, PARTITIONING & MANIFEST GENERATION")
    print("=" * 80)
    data_dir = resolve_data_dir(data_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Input Data Directory:  {data_dir}")
    print(f"[*] Output Directory:      {output_dir}")
    print(f"[*] Train/Val Ratio:       {train_ratio:.2f} / {1.0 - train_ratio:.2f}")
    print(f"[*] Deterministic Seed:    {seed}")
    print(f"[*] Video Segmentation:    {'Enabled' if use_segmentation else 'Disabled'}")

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

    iso_res, part_res, manifest_summary = preparer.prepare()

    print(f"[+] Isolated {len(iso_res.test_images):,} holdout images strictly to test split.")
    print(f"[+] Partitioned {len(part_res.train_images):,} train and {len(part_res.val_images):,} val images.")
    if part_res.video_train_count > 0 or part_res.video_val_count > 0:
        print(f"    (Video frames: {part_res.video_train_count:,} train, {part_res.video_val_count:,} val; Static: {part_res.static_train_count:,} train, {part_res.static_val_count:,} val)")
    print(f"[+] Generated dataset-local manifests:")
    summaries = manifest_summary.dataset_summaries or {"dataset": manifest_summary}
    for dataset_name, summary in summaries.items():
        print(f"    • {dataset_name} data.yaml: {summary.data_yaml_path}")
        print(f"      train.txt: {summary.splits_dir / 'train.txt'} ({summary.train_count:,} images)")
        print(f"      val.txt:   {summary.splits_dir / 'val.txt'} ({summary.val_count:,} images)")
        print(f"      test.txt:  {summary.splits_dir / 'test.txt'} ({summary.test_count:,} images)")
        if summary.manifest_json_path:
            print(f"      manifest:  {summary.manifest_json_path}")

    # Quarantined files go to manual review
    manual_review_items = []
    for p in part_res.quarantined_images:
        manual_review_items.append({
            "filename": p.name,
            "path": str(p).replace("\\", "/"),
            "reason": "quarantined_image",
            "severity": "high",
            "details": "Corrupt or invalid frame quarantined from train/val splits",
        })
    mr_file = manual_review_out or (output_dir / "manual_review_needed.json")
    with open(mr_file, "w", encoding="utf-8") as f:
        json.dump({"total_items": len(manual_review_items), "items": manual_review_items}, f, indent=2)

    print(f"    • Manual Review: {mr_file}")
    print("[*] Prepare stage completed successfully.\n")

    return manifest_summary.to_dict()


def run_verify(
    data_yaml_path: Optional[Path] = None,
    manifest_json: Optional[Path] = None,
    manifest_csv: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    tolerance: float = DEFAULT_EPSILON,
    check_hashes: bool = False,
    json_out: Optional[Path] = None,
    manual_review_out: Optional[Path] = None,
) -> int:
    """Execute acceptance criteria verification suite."""
    print("=" * 80)
    print(" PIPELINE STAGE: ACCEPTANCE VERIFICATION")
    print("=" * 80)

    if data_yaml_path is None:
        data_yaml_path = Path("data.yaml")
    elif Path(data_yaml_path) == Path("data.yaml") and not Path(data_yaml_path).exists():
        local_manifest_yaml = Path(MANIFEST_DIR_NAME) / "data.yaml"
        if local_manifest_yaml.exists():
            data_yaml_path = local_manifest_yaml
    data_yaml_path = Path(data_yaml_path).resolve()

    verifier = DatasetVerifier(tolerance=tolerance)
    report = verifier.verify(
        data_yaml_path=data_yaml_path,
        manifest_json_path=Path(manifest_json).resolve() if manifest_json else None,
        manifest_csv_path=Path(manifest_csv).resolve() if manifest_csv else None,
        data_dir=Path(data_dir).resolve() if data_dir else None,
        splits_dir=Path(splits_dir).resolve() if splits_dir else None,
        tolerance=tolerance,
        check_hashes=check_hashes,
        manual_review_out=manual_review_out,
    )

    verifier.print_report(report)

    if json_out:
        json_out = Path(json_out).resolve()
        verifier.save_json_report(report, json_out)
        print(f"[*] Verification report JSON saved to: {json_out}")

    return 0 if report.status == "PASS" else 1


def run_all(
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("."),
    train_ratio: float = 0.8,
    seed: int = 42,
    tolerance: float = DEFAULT_EPSILON,
    validate_images: bool = True,
    check_hashes: bool = False,
    json_out: Optional[Path] = None,
    manual_review_out: Optional[Path] = None,
) -> int:
    """Execute audit -> segment -> prepare -> verify stages end-to-end."""
    data_dir = resolve_data_dir(data_dir)
    output_dir = Path(output_dir).resolve()
    mr_path = manual_review_out or (output_dir / "manual_review_needed.json")

    # 1. Audit
    run_audit(
        data_dir=data_dir,
        output_dir=output_dir,
        epsilon=tolerance,
        validate_images=validate_images,
        manual_review_out=mr_path,
    )

    # 2. Segment
    run_segment(
        data_dir=data_dir,
        output_dir=output_dir,
        manual_review_out=mr_path,
    )

    # 3. Prepare
    run_prepare(
        data_dir=data_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
        epsilon=tolerance,
        validate_images=validate_images,
        compute_hashes=check_hashes,
        use_segmentation=True,
        manual_review_out=mr_path,
    )

    # 4. Verify the dataset-local artifacts generated by prepare.
    resolved_data_dir = Path(data_dir).resolve()
    _, target_root = resolve_dataset_roots(resolved_data_dir)
    if not target_root.exists():
        target_root = resolved_data_dir
    artifact_root = dataset_artifact_root(
        target_root,
        output_dir=output_dir,
        data_dir=resolved_data_dir,
    )
    data_yaml_path = (artifact_root / MANIFEST_DIR_NAME / "data.yaml").resolve()
    manifest_json_path = (artifact_root / MANIFEST_DIR_NAME / "dataset_manifest.json").resolve()
    manifest_csv_path = (artifact_root / MANIFEST_DIR_NAME / "dataset_manifest.csv").resolve()

    code = run_verify(
        data_yaml_path=data_yaml_path,
        manifest_json=manifest_json_path if manifest_json_path.exists() else None,
        manifest_csv=manifest_csv_path if manifest_csv_path.exists() else None,
        data_dir=data_dir,
        splits_dir=(artifact_root / SPLITS_DIR_NAME).resolve(),
        tolerance=tolerance,
        check_hashes=check_hashes,
        json_out=json_out,
        manual_review_out=mr_path,
    )
    return code


def build_parser() -> argparse.ArgumentParser:
    """Build the unified command line parser."""
    parser = argparse.ArgumentParser(
        prog="fire_audit",
        description="Fire and Smoke Detection Dataset Audit, Video Segmentation, Partitioning & Verification Pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # Common options
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory containing raw datasets (default: data_path from config.yaml)",
    )
    parent_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Report/output parent; artifacts are grouped under one directory per dataset (default: .)",
    )
    parent_parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Epsilon tolerance for coordinate bounds (default: {DEFAULT_EPSILON})",
    )
    parent_parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Skip PIL image header decode to accelerate scanning",
    )
    parent_parser.add_argument(
        "--manual-review-out",
        type=Path,
        default=None,
        help="Destination path for manual review report JSON (default: output_dir/manual_review_needed.json)",
    )

    # Subcommand: audit
    subparsers.add_parser(
        "audit",
        parents=[parent_parser],
        help="Scan raw datasets and generate audit reports (Markdown & JSON)",
    )

    # Subcommand: segment
    segment_parser = subparsers.add_parser(
        "segment",
        parents=[parent_parser],
        help="Run VideoSegmentationEngine on a dataset and export JSON/CSV manifests",
    )
    segment_parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=18,
        help="Hamming distance threshold for 64-bit dHash temporal coherence (default: 18)",
    )
    segment_parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum sequence length to classify as video_frame (default: 5)",
    )
    segment_parser.add_argument(
        "--lookahead-k",
        type=int,
        default=2,
        help="Temporal lookahead bridging window k (default: 2)",
    )
    segment_parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread count for parallel perceptual hashing (default: 16)",
    )

    # Subcommand: prepare
    prep_parser = subparsers.add_parser(
        "prepare",
        parents=[parent_parser],
        help="Create a held-out test split, partition a dataset, and generate manifests",
    )
    prep_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of training sequences (default: 0.8)",
    )
    prep_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sequence partitioning (default: 42)",
    )
    prep_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Compute SHA-256 hashes during isolation",
    )
    prep_parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Bypass video sequence segmentation and use standard partitioner",
    )

    # Subcommand: verify
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify dataset manifests, split isolation, class IDs, and bounding box bounds",
    )
    verify_parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data.yaml"),
        help="Path to data.yaml manifest (default: data.yaml)",
    )
    verify_parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Path to dataset_manifest.json (optional)",
    )
    verify_parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Path to dataset_manifest.csv (optional)",
    )
    verify_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset root for 100% census verification (optional)",
    )
    verify_parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Optional directory containing split text files",
    )
    verify_parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Epsilon tolerance for bounding box coordinates (default: {DEFAULT_EPSILON})",
    )
    verify_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Verify SHA-256 bitwise hash disjointness across splits",
    )
    verify_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output verification JSON",
    )
    verify_parser.add_argument(
        "--manual-review-out",
        type=Path,
        default=None,
        help="Optional path to output manual review items JSON",
    )

    # Subcommand: all
    all_parser = subparsers.add_parser(
        "all",
        parents=[parent_parser],
        help="Run audit, segment, prepare, and verify stages end-to-end",
    )
    all_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of training sequences (default: 0.8)",
    )
    all_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sequence partitioning (default: 42)",
    )
    all_parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Verify SHA-256 bitwise hash disjointness across splits",
    )
    all_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output verification JSON",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main execution entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    validate_imgs = not getattr(args, "skip_image_decode", False)
    mr_out = getattr(args, "manual_review_out", None)

    if args.command == "audit":
        run_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epsilon=args.tolerance,
            validate_images=validate_imgs,
            manual_review_out=mr_out,
        )
        return 0

    elif args.command == "segment":
        run_segment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dhash_threshold=args.dhash_threshold,
            min_length=args.min_length,
            lookahead_k=args.lookahead_k,
            workers=args.workers,
            manual_review_out=mr_out,
        )
        return 0

    elif args.command == "prepare":
        run_prepare(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            epsilon=args.tolerance,
            validate_images=validate_imgs,
            compute_hashes=getattr(args, "check_hashes", False),
            use_segmentation=not getattr(args, "no_segmentation", False),
            manual_review_out=mr_out,
        )
        return 0

    elif args.command == "verify":
        return run_verify(
            data_yaml_path=args.data_yaml,
            manifest_json=args.manifest_json,
            manifest_csv=args.manifest_csv,
            data_dir=args.data_dir,
            splits_dir=args.splits_dir,
            tolerance=args.tolerance,
            check_hashes=args.check_hashes,
            json_out=args.json_out,
            manual_review_out=mr_out,
        )

    elif args.command == "all":
        return run_all(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            tolerance=args.tolerance,
            validate_images=validate_imgs,
            check_hashes=args.check_hashes,
            json_out=args.json_out,
            manual_review_out=mr_out,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
