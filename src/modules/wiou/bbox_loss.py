from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, bbox2dist

from .wiou import WIoU


class WIoUBboxLoss(nn.Module):
    """
    Bounding-box regression loss using WIoU.

    Compared with Ultralytics BboxLoss:

        Original:
            CIoU + DFL

        Custom:
            WIoU + DFL

    Everything else remains unchanged.
    """

    def __init__(
        self,
        reg_max: int = 16,
        wiou_monotonous: bool = False,
        wiou_alpha: float = 1.9,
        wiou_delta: float = 3.0,
        wiou_momentum: float = 0.01,
    ) -> None:
        super().__init__()

        self.reg_max = int(reg_max)

        self.dfl_loss = (
            DFLoss(self.reg_max)
            if self.reg_max > 1
            else None
        )

        self.wiou = WIoU(
            monotonous=wiou_monotonous,
            alpha=wiou_alpha,
            delta=wiou_delta,
            momentum=wiou_momentum,
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WIoU and DFL/L1 losses.

        Args:
            pred_dist:
                Raw regression distribution predictions.

            pred_bboxes:
                Decoded predicted bounding boxes, xyxy.

            anchor_points:
                Anchor points.

            target_bboxes:
                Target bounding boxes, xyxy.

            target_scores:
                Scores produced by Task-Aligned Assignment.

            target_scores_sum:
                Sum of target scores.

            fg_mask:
                Foreground mask.

            imgsz:
                Image size tensor.

            stride:
                Prediction stride.

        Returns:
            loss_iou:
                WIoU box regression loss.

            loss_dfl:
                DFL or L1 regression loss.
        """

        # ----------------------------------------------------------
        # Weight from Task-Aligned Assigner.
        #
        # Keep exactly the same weighting semantics as the
        # Ultralytics BboxLoss.
        # ----------------------------------------------------------
        weight = target_scores[fg_mask].sum(
            -1,
            keepdim=True,
        )

        # ----------------------------------------------------------
        # WIoU
        # ----------------------------------------------------------
        loss_wiou = self.wiou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
        )

        if loss_wiou.ndim == 1:
            loss_wiou = loss_wiou.unsqueeze(-1)

        if not isinstance(target_scores_sum, torch.Tensor):
                    target_scores_sum = torch.as_tensor(
                        target_scores_sum,
                        device=pred_bboxes.device,
                        dtype=pred_bboxes.dtype,
                    )

        normalizer = target_scores_sum.clamp_min(
            torch.finfo(
                target_scores_sum.dtype
            ).eps
        )

        loss_iou = (
            loss_wiou * weight
        ).sum() / normalizer

        # ----------------------------------------------------------
        # DFL
        # ----------------------------------------------------------
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )

            pred_dist_fg = pred_dist[fg_mask].view(
                -1,
                self.dfl_loss.reg_max,
            )

            target_ltrb_fg = target_ltrb[fg_mask]

            loss_dfl = (
                self.dfl_loss(
                    pred_dist_fg,
                    target_ltrb_fg,
                )
                * weight
            )

            loss_dfl = (
                loss_dfl.sum()
                / normalizer
            )

        else:
            # ------------------------------------------------------
            # DFL-free L1 fallback.
            # Keep Ultralytics normalization semantics.
            # ------------------------------------------------------
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
            )

            target_ltrb = target_ltrb * stride

            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]

            pred_dist_scaled = pred_dist * stride

            pred_dist_scaled[..., 0::2] /= imgsz[1]
            pred_dist_scaled[..., 1::2] /= imgsz[0]

            loss_dfl = (
                F.l1_loss(
                    pred_dist_scaled[fg_mask],
                    target_ltrb[fg_mask],
                    reduction="none",
                )
                .mean(-1, keepdim=True)
                * weight
            )

            loss_dfl = (
                loss_dfl.sum()
                / normalizer
            )

        return loss_iou, loss_dfl