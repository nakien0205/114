"""
Annotation and image file validation engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

from src.fire_audit.config import (
    CLASS_IDS,
    DEFAULT_EPSILON,
    BoundingBox,
    ValidationResult,
)


def validate_bbox_line(
    line: str, epsilon: float = DEFAULT_EPSILON
) -> Tuple[bool, Optional[str], Optional[BoundingBox]]:
    """
    Validate and parse a single YOLO bounding box string.
    Expected format: <class_id> <x_center> <y_center> <width> <height>

    Parameters
    ----------
    line : str
        The raw annotation line.
    epsilon : float
        Numerical tolerance for floating point boundary clamping.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[BoundingBox]]
        (is_valid, error_message, parsed_clamped_box)
    """
    stripped = line.strip()
    if not stripped:
        return True, None, None

    tokens = stripped.split()
    if len(tokens) != 5:
        return False, f"Malformed token count: expected 5 tokens, got {len(tokens)} ('{line}')", None

    try:
        class_id = int(tokens[0])
        xc = float(tokens[1])
        yc = float(tokens[2])
        w = float(tokens[3])
        h = float(tokens[4])
    except ValueError as e:
        return False, f"Non-numeric token in annotation: {e} ('{line}')", None

    # Validate class ID
    if class_id not in CLASS_IDS:
        return False, f"Invalid class ID: {class_id}, expected 0 (fire) or 1 (smoke)", None

    # Validate box dimensions
    if w <= 0.0 or h <= 0.0:
        return False, f"Degenerate box with zero or negative dimension: w={w}, h={h}", None

    if w > 1.0 + epsilon or h > 1.0 + epsilon:
        return False, f"Dimensions exceed 1.0: w={w}, h={h}", None

    # Validate center coordinates
    if xc < -epsilon or xc > 1.0 + epsilon or yc < -epsilon or yc > 1.0 + epsilon:
        return False, f"Center coordinate out of bounds [0, 1]: ({xc}, {yc})", None

    # Validate box extents
    xmin = xc - (w / 2.0)
    xmax = xc + (w / 2.0)
    ymin = yc - (h / 2.0)
    ymax = yc + (h / 2.0)

    if xmin < -epsilon or xmax > 1.0 + epsilon or ymin < -epsilon or ymax > 1.0 + epsilon:
        return False, (
            f"Extents [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}] "
            f"exceed normalized bounds [0, 1]"
        ), None

    # Clamp coordinates to exact [0.0, 1.0] interval
    clamped_xc = max(0.0, min(1.0, xc))
    clamped_yc = max(0.0, min(1.0, yc))
    clamped_w = max(0.0, min(1.0, w))
    clamped_h = max(0.0, min(1.0, h))

    box = BoundingBox(
        class_id=class_id,
        x_center=clamped_xc,
        y_center=clamped_yc,
        width=clamped_w,
        height=clamped_h,
    )
    return True, None, box


def validate_label_file(
    label_path: Path, epsilon: float = DEFAULT_EPSILON
) -> ValidationResult:
    """
    Validate a YOLO label file. Handles empty / 0-byte files as valid negative/background frames.

    Parameters
    ----------
    label_path : Path
        Path to the label file.
    epsilon : float
        Numerical tolerance.

    Returns
    -------
    ValidationResult
        Detailed validation status, parsed boxes, and errors if any.
    """
    if not label_path.exists():
        return ValidationResult(
            valid=False,
            errors=[f"Label file does not exist: {label_path}"],
        )

    # 0-byte file is a valid negative background frame
    if label_path.stat().st_size == 0:
        return ValidationResult(valid=True, boxes=[], is_negative=True)

    try:
        content = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = label_path.read_text(encoding="latin-1")
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to read label file encoding: {e}"],
            )

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ValidationResult(valid=True, boxes=[], is_negative=True)

    boxes = []
    errors = []
    for line_idx, line in enumerate(lines, start=1):
        is_valid, err_msg, box = validate_bbox_line(line, epsilon=epsilon)
        if not is_valid:
            errors.append(f"Line {line_idx}: {err_msg}")
        elif box is not None:
            boxes.append(box)

    if errors:
        return ValidationResult(valid=False, boxes=boxes, errors=errors)

    is_neg = (len(boxes) == 0)
    return ValidationResult(valid=True, boxes=boxes, is_negative=is_neg)


def validate_image_file(image_path: Path) -> Tuple[bool, Optional[str], int, int]:
    """
    Validate an image file header and extract pixel dimensions.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    Tuple[bool, Optional[str], int, int]
        (is_valid, error_message, width, height)
    """
    if not image_path.exists():
        return False, f"Image file does not exist: {image_path}", 0, 0

    if image_path.stat().st_size == 0:
        return False, f"Image file is 0 bytes: {image_path}", 0, 0

    try:
        try:
            from src.fire_audit.segment.classifier import get_image_dimensions
        except ImportError:
            from fire_audit.segment.classifier import get_image_dimensions
        w, h = get_image_dimensions(image_path)
        if w > 0 and h > 0:
            return True, None, w, h
    except Exception:
        pass

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, f"Invalid image dimensions: {width}x{height}", 0, 0
        return True, None, width, height
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        return False, f"Corrupt image file: {e}", 0, 0
    except Exception as e:
        return False, f"Unexpected error reading image: {e}", 0, 0
