"""Read-only audit of adjacent FASDD_CV image IDs across supplied YOLO splits.

Outputs CSVs beside this script. A consecutive ID or close perceptual hash is
only a candidate related scene, not proof of shared source video or leakage.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
import re

from PIL import Image


DEFAULT_ROOT = Path(r"C:\Users\phong\Downloads\Fire\FASDD_CV")
NAME_RE = re.compile(r"^(?P<prefix>.+_CV)(?P<number>\d{6})\.jpg$", re.I)


def read_splits(root: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    lists = root / "annotations" / "YOLO_CV"
    for split in ("train", "val", "test"):
        for raw in (lists / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            name = Path(raw.strip().replace("\\", "/")).name
            if not name:
                continue
            if name in assignments:
                raise ValueError(f"Duplicate filename across/within splits: {name}")
            if not (root / "images" / name).is_file():
                raise FileNotFoundError(root / "images" / name)
            assignments[name] = split
    return assignments


def adjacent_pairs(assignments: dict[str, str]):
    by_key = {}
    for name, split in assignments.items():
        match = NAME_RE.match(name)
        if not match:
            continue
        by_key[(match["prefix"], int(match["number"]))] = (name, split)
    for (prefix, number), (name, split) in sorted(by_key.items()):
        following = by_key.get((prefix, number + 1))
        if following and following[1] != split:
            yield name, split, following[0], following[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--threshold", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assignments = read_splits(args.root)

    @lru_cache(maxsize=65536)
    def dhash(name: str) -> int:
        with Image.open(args.root / "images" / name) as image:
            pixels = image.convert("L").resize((9, 8)).tobytes()
        result = 0
        for row in range(8):
            for column in range(8):
                result = (result << 1) | int(
                    pixels[row * 9 + column] > pixels[row * 9 + column + 1]
                )
        return result

    fields = ("image_a", "split_a", "image_b", "split_b", "dhash_distance")
    all_path = args.out / "FASDD_CV_adjacent_cross_split.csv"
    close_path = args.out / "FASDD_CV_dhash8_cross_split.csv"
    counts: dict[tuple[str, str], int] = {}
    close_counts: dict[tuple[str, str], int] = {}
    with all_path.open("w", newline="", encoding="utf-8") as all_file, close_path.open(
        "w", newline="", encoding="utf-8"
    ) as close_file:
        all_writer = csv.DictWriter(all_file, fieldnames=fields)
        close_writer = csv.DictWriter(close_file, fieldnames=fields)
        all_writer.writeheader()
        close_writer.writeheader()
        for index, (a, sa, b, sb) in enumerate(adjacent_pairs(assignments), 1):
            distance = (dhash(a) ^ dhash(b)).bit_count()
            row = dict(image_a=a, split_a=sa, image_b=b, split_b=sb, dhash_distance=distance)
            all_writer.writerow(row)
            key = tuple(sorted((sa, sb)))
            counts[key] = counts.get(key, 0) + 1
            if distance <= args.threshold:
                close_writer.writerow(row)
                close_counts[key] = close_counts.get(key, 0) + 1
            if index % 5000 == 0:
                print(f"Audited {index:,} cross-split adjacent pairs", flush=True)

    print(f"Images in supplied split lists: {len(assignments):,}")
    for key in sorted(counts):
        print(f"{'+'.join(key)}: {counts[key]:,} adjacent pairs, {close_counts.get(key, 0):,} hash-close")
    print(f"All adjacent pairs: {sum(counts.values()):,}")
    print(f"Hash-close pairs (distance <= {args.threshold}): {sum(close_counts.values()):,}")
    print(f"CSV files: {all_path}; {close_path}")


if __name__ == "__main__":
    main()
