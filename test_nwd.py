"""
Test NWD implementation.

Usage:

    python test_nwd.py

or, after training:

    python test_nwd.py \
        --weights runs/train/yolo11n_nwd/weights/best.pt \
        --data data.yaml \
        --device 0
"""

from __future__ import annotations

import argparse
import logging

import torch

from src.modules.nwd import (
    nwd, xyxy_to_xywh
)
from src.modules.bbox_loss import NWDBboxLoss
from src.modules.detection_loss import NWDDetectionLoss
from src.modules.model import NWDDetectionModel
from src.fire_detection.train import (
    check_ultralytics_version,
    check_ultralytics_version,
    setup_logging, 
)


logger = logging.getLogger(__name__)


# ============================================================
# TEST 1
# ============================================================

def test_xyxy_to_xywh():

    boxes = torch.tensor(
        [
            [
                10.0,
                20.0,
                30.0,
                60.0,
            ]
        ]
    )

    result = xyxy_to_xywh(
        boxes
    )

    expected = torch.tensor(
        [
            [
                20.0,
                40.0,
                20.0,
                40.0,
            ]
        ]
    )

    assert torch.allclose(
        result,
        expected,
    )

    logger.info(
        "PASS: xyxy_to_xywh"
    )


# ============================================================
# TEST 2
# ============================================================

def test_nwd_identity():

    boxes = torch.tensor(
        [
            [
                20.0,
                40.0,
                20.0,
                40.0,
            ]
        ]
    )

    score = nwd(
        boxes,
        boxes,
    )

    assert torch.allclose(
        score,
        torch.ones_like(score),
        atol=1e-6,
    )

    logger.info(
        "PASS: identical boxes -> NWD = 1"
    )


# ============================================================
# TEST 3
# ============================================================

def test_nwd_monotonicity():

    target = torch.tensor(
        [
            [
                20.0,
                40.0,
                20.0,
                40.0,
            ]
        ]
    )

    close = torch.tensor(
        [
            [
                21.0,
                40.0,
                20.0,
                40.0,
            ]
        ]
    )

    far = torch.tensor(
        [
            [
                30.0,
                40.0,
                20.0,
                40.0,
            ]
        ]
    )

    close_score = nwd(
        close,
        target,
    )

    far_score = nwd(
        far,
        target,
    )

    assert (
        close_score.item()
        > far_score.item()
    )

    logger.info(
        "PASS: NWD monotonicity "
        "close=%.6f far=%.6f",
        close_score.item(),
        far_score.item(),
    )


# ============================================================
# TEST 4
# ============================================================

def test_nwd_bbox_loss():

    loss_fn = NWDBboxLoss(
        reg_max=16,
        nwd_weight=0.25,
        nwd_constant=12.8,
    )

    pred_bboxes = torch.tensor(
        [
            [
                [
                    10.0,
                    10.0,
                    30.0,
                    30.0,
                ]
            ]
        ],
        requires_grad=True,
    )

    target_bboxes = torch.tensor(
        [
            [
                [
                    11.0,
                    10.0,
                    31.0,
                    30.0,
                ]
            ]
        ]
    )

    pred_dist = torch.zeros(
        1,
        1,
        64,
        requires_grad=True,
    )

    anchor_points = torch.tensor(
        [
            [
                20.0,
                20.0,
            ]
        ]
    )

    target_scores = torch.ones(
        1,
        1,
        2,
    )

    target_scores_sum = torch.tensor(
        1.0
    )

    fg_mask = torch.ones(
        1,
        1,
        dtype=torch.bool,
    )

    box_loss, dfl_loss = loss_fn(
        pred_dist=pred_dist,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        target_bboxes=target_bboxes,
        target_scores=target_scores,
        target_scores_sum=target_scores_sum,
        fg_mask=fg_mask,
    )

    total_loss = (
        box_loss
        + dfl_loss
    )

    total_loss.backward()

    assert torch.isfinite(
        total_loss
    )

    assert (
        pred_dist.grad
        is not None
    )

    logger.info(
        "PASS: NWDBboxLoss forward/backward "
        "box=%.6f dfl=%.6f",
        box_loss.item(),
        dfl_loss.item(),
    )


# ============================================================
# TEST 5
# ============================================================

def test_model_criterion(
    weights=None,
):

    if weights is None:

        logger.info(
            "SKIP: model criterion test "
            "because --weights was not supplied"
        )

        return

    from ultralytics import YOLO

    base_model = YOLO(
        weights
    )

    model = NWDDetectionModel(
        cfg=base_model.model.yaml,
        nc=base_model.model.nc,
        verbose=False,
        nwd_weight=0.25,
        nwd_constant=12.8,
    )

    model.load(
        weights
    )

    criterion = (
        model.init_criterion()
    )

    assert isinstance(
        criterion,
        NWDDetectionLoss,
    )

    assert isinstance(
        criterion.bbox_loss,
        NWDBboxLoss,
    )

    assert (
        criterion.bbox_loss.nwd_weight
        == 0.25
    )

    assert (
        criterion.bbox_loss.nwd_constant
        == 12.8
    )

    logger.info(
        "PASS: model -> "
        "NWDDetectionLoss -> "
        "NWDBboxLoss"
    )


# ============================================================
# OPTIONAL VALIDATION
# ============================================================

def test_validation(
    weights,
    data,
    device,
):

    from ultralytics import YOLO

    logger.info(
        "Running validation..."
    )

    model = YOLO(
        weights
    )

    results = model.val(
        data=data,
        device=device,
        verbose=True,
    )

    if hasattr(
        results,
        "results_dict",
    ):
        logger.info(
            "Validation results:"
        )

        for key, value in (
            results.results_dict.items()
        ):
            logger.info(
                "%s = %s",
                key,
                value,
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test custom NWD implementation"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained .pt checkpoint",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/logs",
    )

    args = parser.parse_args()

    setup_logging(
        args.log_dir
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "NWD TEST"
    )

    logger.info(
        "Expected Ultralytics: 8.4.140"
    )

    logger.info(
        "=" * 70
    )

    try:
        test_xyxy_to_xywh()

        test_nwd_identity()

        test_nwd_monotonicity()

        test_nwd_bbox_loss()

        test_model_criterion(
            args.weights
        )

        if (
            args.weights
            and args.data
        ):

            test_validation(
                args.weights,
                args.data,
                args.device,
            )

        elif (
            args.weights
            or args.data
        ):

            logger.warning(
                "Validation skipped. "
                "Provide BOTH --weights and --data."
            )

        logger.info(
            "=" * 70
        )

        logger.info(
            "ALL NWD TESTS PASSED"
        )

        logger.info(
            "=" * 70
        )

        return 0

    except Exception as e:

        logger.exception(
            "NWD TEST FAILED: %s",
            e,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )