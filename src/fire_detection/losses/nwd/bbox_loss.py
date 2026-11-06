import torch
import torch.nn as nn

from ultralytics.utils.loss import DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

from .nwd import xyxy_to_xywh, nwd


class NWDBboxLoss(nn.Module):
    """
    Bbox loss for YOLOv8 / YOLO11.

    Loss:

        L_box =
            (1 - nwd_weight) * L_CIoU
            + nwd_weight * L_NWD

    DFL remains unchanged.
    """

    def __init__(
        self,
        reg_max: int = 16,
        nwd_weight: float = 0.25,
        nwd_constant: float = 12.8,
    ):
        super().__init__()

        if not 0.0 <= nwd_weight <= 1.0:
            raise ValueError(f"nwd_weight must be in [0, 1], got {nwd_weight}")

        if nwd_constant <= 0:
            raise ValueError(f"nwd_constant must be > 0, got {nwd_constant}")

        self.reg_max = reg_max
        self.nwd_weight = nwd_weight
        self.nwd_constant = nwd_constant

        # YOLOv8 / YOLO11 use DFL when reg_max > 1.
        self.dfl_loss = (
            DFLoss(reg_max)
            if reg_max > 1
            else None
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
        *args,
        **kwargs,
    ):
        """
        Args:
            pred_dist:
                Predicted distributions.

            pred_bboxes:
                Decoded predicted boxes.
                Shape: [B, A, 4]
                Format: xyxy

            anchor_points:
                Anchor points.

            target_bboxes:
                Assigned target boxes.
                Shape: [B, A, 4]
                Format: xyxy

            target_scores:
                Target classification scores.

            target_scores_sum:
                Sum of target scores.

            fg_mask:
                Foreground mask.

            *args, **kwargs:
                Keep compatibility with Ultralytics 8.4.x.
        """

        # ==========================================================
        # 1. Positive samples
        # ==========================================================

        if not fg_mask.any():
            zero = pred_dist.sum() * 0.0
            return zero, zero

        fg_mask_idx = fg_mask.nonzero(
            as_tuple=True
        )

        pred_fg = pred_bboxes[fg_mask_idx]
        target_fg = target_bboxes[fg_mask_idx]

        # ==========================================================
        # 2. Target score weights
        # ==========================================================

        weight = target_scores[fg_mask_idx].sum(dim=-1,keepdim=True,)

        # ==========================================================
        # 3. CIoU
        # ==========================================================

        iou = bbox_iou(
            pred_fg,
            target_fg,
            xywh=False,
            CIoU=True,
        )

        ciou_loss = 1.0 - iou

        # ==========================================================
        # 4. NWD
        # ==========================================================

        pred_xywh = xyxy_to_xywh(pred_fg)

        target_xywh = xyxy_to_xywh(target_fg)

        nwd_score = nwd(
            pred_xywh,
            target_xywh,
            constant=self.nwd_constant,
        )

        nwd_regression_loss = (
            1.0 - nwd_score
        )

        # ==========================================================
        # 5. Hybrid CIoU + NWD
        # ==========================================================

        box_loss = (
            (1.0 - self.nwd_weight) * ciou_loss
            + self.nwd_weight * nwd_regression_loss
        )

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

        # ==========================================================
        # 6. Weighted box loss
        # ==========================================================

        loss_iou = (
            box_loss * weight
        ).sum() / normalizer

        # ==========================================================
        # 7. DFL
        # ==========================================================

        if self.dfl_loss is not None:

            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.reg_max - 1,
            )

            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask_idx].view(-1, self.reg_max),
                target_ltrb[fg_mask_idx],
            )

            loss_dfl = (
                loss_dfl * weight
            ).sum() / normalizer

        else:

            loss_dfl = torch.zeros(
                (),
                device=pred_dist.device,
                dtype=pred_dist.dtype,
            )

        return loss_iou, loss_dfl