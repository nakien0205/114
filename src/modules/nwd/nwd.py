import torch
import math

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy to xywh.

    Args:
        boxes: (..., 4)
               [x1, y1, x2, y2]

    Returns:
        (..., 4)
        [cx, cy, w, h]
    """

    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    w = x2 - x1
    h = y2 - y1

    return torch.stack((cx, cy, w, h), dim=-1)


def wasserstein_distance(
    box1: torch.Tensor,
    box2: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian Wasserstein Distance used by NWD.

    Args:
        box1: (..., 4), xywh
        box2: (..., 4), xywh

    Returns:
        (...,)
    """

    x1, y1, w1, h1 = box1.unbind(dim=-1)
    x2, y2, w2, h2 = box2.unbind(dim=-1)

    # Center distance
    center_distance = ((x1 - x2).pow(2) + (y1 - y2).pow(2))

    # Width/height distance
    size_distance = ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4.0

    return center_distance + size_distance


def nwd(
    box1: torch.Tensor,
    box2: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Normalized Wasserstein Distance.

    Args:
        box1: (..., 4), xywh
        box2: (..., 4), xywh
        constant: normalization constant

    Returns:
        NWD similarity.
    """
    if constant <= 0:
        raise ValueError(f"NWD constant must be > 0, got {constant}")

    distance_squared = wasserstein_distance(box1, box2).clamp_min(0.0)
    distance = (torch.sqrt(distance_squared + eps) - math.sqrt(eps)).clamp_min(0.0)

    score = torch.exp(-distance / constant)

    identical = distance_squared <= eps

    return torch.where(
        identical,
        torch.ones_like(score),
        score,
    )