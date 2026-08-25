"""
Classifier module for video sequence detection and perceptual hashing.

Provides:
- Fast JPEG binary SOF dimension parser with PIL fallback.
- 64-bit dHash perceptual hashing using cv2.IMREAD_REDUCED_GRAYSCALE_8 with PIL fallback.
- Multithreaded parallel hash computation.
- Temporal lookahead bridge (k=2) for absorbing keyframe/compression spikes.
- Temporal coherence verification for video sequence qualification.
- Sequence splitting into verified contiguous video segments.
"""

from __future__ import annotations

import statistics
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

from PIL import Image


# Standard JPEG SOF markers
JPEG_SOF_MARKERS = {
    0xC0,  # SOF0: Baseline DCT
    0xC1,  # SOF1: Extended Sequential DCT
    0xC2,  # SOF2: Progressive DCT
    0xC3,  # SOF3: Lossless Sequential
    0xC5,  # SOF5: Differential Sequential DCT
    0xC6,  # SOF6: Differential Progressive DCT
    0xC7,  # SOF7: Differential Lossless
    0xC9,  # SOF9: Extended Sequential DCT, Arithmetic Coding
    0xCA,  # SOF10: Progressive DCT, Arithmetic Coding
    0xCB,  # SOF11: Lossless Sequential, Arithmetic Coding
    0xCD,  # SOF13: Differential Sequential DCT, Arithmetic Coding
    0xCE,  # SOF14: Differential Progressive DCT, Arithmetic Coding
    0xCF,  # SOF15: Differential Lossless, Arithmetic Coding
}


def get_jpeg_dimensions(file_path: Union[Path, str]) -> Optional[Tuple[int, int]]:
    """
    Extract (width, height) from JPEG headers without full image decompression.

    Parses the binary stream for Start Of Frame (SOF) markers.
    Returns (width, height) tuple or None if parsing fails.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
            if header != b"\xff\xd8":
                return None
            while True:
                marker_bytes = f.read(2)
                if len(marker_bytes) < 2:
                    return None
                if marker_bytes[0] != 0xFF:
                    while marker_bytes and marker_bytes[0] != 0xFF:
                        marker_bytes = f.read(1)
                    if not marker_bytes:
                        return None
                    next_byte = f.read(1)
                    if not next_byte:
                        return None
                    marker = next_byte[0]
                else:
                    while marker_bytes[1] == 0xFF:
                        b = f.read(1)
                        if not b:
                            return None
                        marker_bytes = b"\xff" + b
                    marker = marker_bytes[1]

                if marker in (0xD8, 0xD9):  # SOI, EOI
                    if marker == 0xD9:
                        return None
                    continue
                if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # RST0-RST7, TEM
                    continue

                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    return None
                length = struct.unpack(">H", length_bytes)[0]
                if length < 2:
                    return None

                if marker in JPEG_SOF_MARKERS:
                    sof_data = f.read(5)
                    if len(sof_data) < 5:
                        return None
                    _precision, height, width = struct.unpack(">BHH", sof_data)
                    if width > 0 and height > 0:
                        return (width, height)
                    return None
                else:
                    f.seek(length - 2, 1)
    except Exception:
        return None


def get_image_dimensions(file_path: Union[Path, str]) -> Tuple[int, int]:
    """
    Extract (width, height) of an image with fast binary JPEG header scanning
    and PIL fallback for non-JPEG or atypical formats.

    Returns (width, height) or (0, 0) if unreadable.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        dims = get_jpeg_dimensions(path)
        if dims is not None:
            return dims

    try:
        with Image.open(path) as img:
            return img.size  # returns (width, height)
    except Exception:
        return (0, 0)


def compute_aspect_ratio_str(width: int, height: int) -> str:
    """Calculate aspect ratio string formatted to 3 decimal places."""
    if width <= 0 or height <= 0:
        return "0.000"
    return f"{width / height:.3f}"


def compute_dhash(image_path: Union[Path, str]) -> int:
    """
    Compute a 64-bit difference hash (dHash) for an image.

    Uses cv2.IMREAD_REDUCED_GRAYSCALE_8 for high-throughput decode
    with PIL fallback if OpenCV is unavailable or fails.
    """
    path_str = str(image_path)
    if cv2 is not None:
        try:
            img = cv2.imread(path_str, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if img is None:
                img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
                diff = resized[:, 1:] > resized[:, :-1]
                h = 0
                for b in diff.flat:
                    h = (h << 1) | int(b)
                return h
        except Exception:
            pass

    # PIL Fallback
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            raw_bytes = img_gray.tobytes()
            h = 0
            for row in range(8):
                row_bytes = raw_bytes[row * 9 : (row + 1) * 9]
                for col in range(8):
                    diff = 1 if row_bytes[col + 1] > row_bytes[col] else 0
                    h = (h << 1) | diff
            return h
    except Exception:
        return 0


def compute_dhashes_parallel(
    image_paths: Sequence[Union[Path, str]],
    max_workers: int = 16,
) -> Dict[str, int]:
    """
    Compute 64-bit dHashes for a sequence of image paths concurrently.

    Returns mapping from str(path) -> 64-bit int hash.
    """
    if not image_paths:
        return {}

    num_workers = min(max_workers, len(image_paths)) if len(image_paths) > 0 else 1
    paths_str = [str(p) for p in image_paths]

    if num_workers > 1 and len(paths_str) > 1:
        chunksize = max(1, len(paths_str) // (num_workers * 4))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            hash_vals = list(executor.map(compute_dhash, paths_str, chunksize=chunksize))
        return dict(zip(paths_str, hash_vals))
    else:
        return {p: compute_dhash(p) for p in paths_str}


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculate the Hamming distance (number of differing bits) between two hashes."""
    return (hash1 ^ hash2).bit_count()


def is_transition_smooth(
    hashes: Sequence[int],
    idx: int,
    threshold: int = 18,
    lookahead_k: int = 2,
) -> bool:
    """
    Determine if transition starting at idx is smooth or bridged across keyframe flicker.

    Checks:
    1. Direct consecutive transition: d(idx, idx + 1) <= threshold.
    2. Forward lookahead bridge: d(idx, idx + step) <= threshold for step in [2, lookahead_k + 1].
    3. Backward lookahead bridge: d(idx - step, idx + 1) <= threshold for step in [1, lookahead_k].
    """
    n = len(hashes)
    if idx < 0 or idx >= n - 1:
        return False

    # 1. Direct consecutive transition
    if hamming_distance(hashes[idx], hashes[idx + 1]) <= threshold:
        return True

    # 2. Forward lookahead bridge (absorbs flicker at idx + 1)
    max_forward = min(idx + 1 + lookahead_k + 1, n)
    for fwd_idx in range(idx + 2, max_forward):
        if hamming_distance(hashes[idx], hashes[fwd_idx]) <= threshold:
            return True

    # 3. Backward lookahead bridge (absorbs flicker at idx)
    min_back = max(0, idx - lookahead_k)
    for bck_idx in range(min_back, idx):
        if hamming_distance(hashes[bck_idx], hashes[idx + 1]) <= threshold:
            return True

    return False


def verify_temporal_coherence(
    hashes: Sequence[int],
    threshold: int = 18,
    lookahead_k: int = 2,
    min_length: int = 5,
    smooth_ratio_threshold: float = 0.60,
) -> bool:
    """
    Verify whether a sequence of frame hashes exhibits smooth temporal video flow.

    Returns True if:
    - Sequence length >= min_length.
    - Median consecutive distance <= threshold OR smooth transition ratio >= smooth_ratio_threshold.
    - The sequence contains a sustained smooth run of at least min_length - 1 transitions.
    """
    n = len(hashes)
    if n < min_length:
        return False

    diffs = [hamming_distance(hashes[i], hashes[i + 1]) for i in range(n - 1)]
    if not diffs:
        return False

    med_diff = statistics.median(diffs)
    smooth_flags = [is_transition_smooth(hashes, i, threshold, lookahead_k) for i in range(n - 1)]
    smooth_count = sum(1 for s in smooth_flags if s)
    smooth_ratio = smooth_count / len(diffs)

    # Check for max consecutive smooth run
    max_streak = 0
    curr_streak = 0
    for s in smooth_flags:
        if s:
            curr_streak += 1
            if curr_streak > max_streak:
                max_streak = curr_streak
        else:
            curr_streak = 0

    required_streak = min(min_length - 1, len(diffs))
    streak_ok = max_streak >= required_streak

    return (med_diff <= threshold or smooth_ratio >= smooth_ratio_threshold) and streak_ok


def split_into_coherent_segments(
    hashes: Sequence[int],
    threshold: int = 18,
    lookahead_k: int = 2,
    min_length: int = 5,
    smooth_ratio_threshold: float = 0.60,
) -> List[Tuple[int, int]]:
    """
    Split a sequence of frame hashes into 0-indexed [start, end) ranges of coherent video clips.

    First checks if the entire contiguous run is coherent as a whole.
    If not, detects hard scene cuts and validates temporal coherence on candidate sub-segments.
    Segments shorter than min_length or failing coherence checks are excluded.
    """
    n = len(hashes)
    if n < min_length:
        return []

    # If the candidate run as a whole passes temporal coherence, treat as a single sequence
    if verify_temporal_coherence(
        hashes,
        threshold=threshold,
        lookahead_k=lookahead_k,
        min_length=min_length,
        smooth_ratio_threshold=smooth_ratio_threshold,
    ):
        return [(0, n)]

    # Otherwise, identify hard scene cut boundary indices to search for valid sub-sequences
    cut_indices = [0]
    for i in range(n - 1):
        if not is_transition_smooth(hashes, i, threshold, lookahead_k):
            cut_indices.append(i + 1)
    cut_indices.append(n)

    accepted_segments: List[Tuple[int, int]] = []
    for s, e in zip(cut_indices[:-1], cut_indices[1:]):
        if e - s >= min_length:
            sub_hashes = hashes[s:e]
            if verify_temporal_coherence(
                sub_hashes,
                threshold=threshold,
                lookahead_k=lookahead_k,
                min_length=min_length,
                smooth_ratio_threshold=smooth_ratio_threshold,
            ):
                accepted_segments.append((s, e))

    return accepted_segments
