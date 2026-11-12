"""Read-only cross-split audit of the local Home Fire image/label folders."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path.home() / "Downloads" / "Fire" / "Home Fire Dataset"
OUT = Path(__file__).resolve().parent / "reports" / "home_fire_split_audit"
SPLITS = ("train", "val", "test")
BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def dhash(image: Image.Image) -> int:
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(small[:, :-1] > small[:, 1:]).tobytes(), "big")


def records_for(split: str) -> tuple[list[dict], dict]:
    images = sorted((ROOT / split / "images").iterdir())
    labels = ROOT / split / "labels"
    records = []
    stats = Counter()
    for image_path in images:
        if not image_path.is_file():
            continue
        label_path = labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            stats["missing_label"] += 1
            continue
        classes = Counter()
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                fields = line.split()
                if len(fields) != 5:
                    stats["malformed_label_line"] += 1
                    continue
                classes[fields[0]] += 1
        with Image.open(image_path) as image:
            size = image.size
            perceptual_hash = dhash(image)
        records.append(
            {
                "split": split,
                "name": image_path.name,
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "dhash": perceptual_hash,
                "size": size,
                "classes": dict(classes),
            }
        )
        stats["images"] += 1
        stats["boxes"] += sum(classes.values())
        stats["empty_labels"] += not classes
        for class_id, count in classes.items():
            stats[f"class_{class_id}_boxes"] += count
    stats["orphan_labels"] = sum(1 for p in labels.glob("*.txt")) - stats["images"]
    return records, dict(stats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_split = {}
    stats = {}
    for split in SPLITS:
        by_split[split], stats[split] = records_for(split)

    exact_groups = defaultdict(list)
    for split in SPLITS:
        for record in by_split[split]:
            exact_groups[record["sha256"]].append(record)
    exact_cross_split = [
        group for group in exact_groups.values() if len({r["split"] for r in group}) > 1
    ]

    near_counts = {}
    candidates = []
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        left = by_split[left_split]
        right = by_split[right_split]
        right_hashes = np.array([r["dhash"] for r in right], dtype=np.uint64)
        counts = Counter()
        for record in left:
            xor_bytes = np.bitwise_xor(right_hashes, np.uint64(record["dhash"])).view(np.uint8).reshape(-1, 8)
            distances = BIT_COUNTS[xor_bytes].sum(axis=1)
            counts["dhash_le_8"] += int(np.count_nonzero(distances <= 8))
            counts["dhash_le_4"] += int(np.count_nonzero(distances <= 4))
            for index in np.flatnonzero(distances <= 4):
                other = right[int(index)]
                if record["size"] != other["size"]:
                    continue
                candidates.append(
                    {
                        "left_split": left_split,
                        "left_image": record["name"],
                        "right_split": right_split,
                        "right_image": other["name"],
                        "dhash_distance": int(distances[index]),
                        "exact_sha256": record["sha256"] == other["sha256"],
                        "left_classes": json.dumps(record["classes"]),
                        "right_classes": json.dumps(other["classes"]),
                    }
                )
        near_counts[f"{left_split}_{right_split}"] = dict(counts)

    candidates.sort(key=lambda row: (row["dhash_distance"], row["left_split"], row["left_image"]))
    with (OUT / "same_size_dhash_le4_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]) if candidates else [
            "left_split", "left_image", "right_split", "right_image", "dhash_distance", "exact_sha256", "left_classes", "right_classes"
        ])
        writer.writeheader()
        writer.writerows(candidates)

    summary = {
        "dataset_root": str(ROOT),
        "split_stats": stats,
        "exact_cross_split_groups": len(exact_cross_split),
        "exact_cross_split_images": sum(len(group) for group in exact_cross_split),
        "exact_examples": [
            [f"{r['split']}/{r['name']}" for r in group] for group in exact_cross_split[:30]
        ],
        "dhash_pair_counts": near_counts,
        "same_size_dhash_le4_candidates": len(candidates),
        "interpretation": "dHash pairs are screening candidates, not proven shared scenes; no source event IDs are available from these folders.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
