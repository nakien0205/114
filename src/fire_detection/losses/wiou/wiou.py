"""
Wise-IoU v3 loss.

Reference:
    Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic
    Focusing Mechanism", arXiv:2301.10051

Official implementation:
    https://github.com/Instinct323/Wise-IoU

Important:
    - Does NOT modify Ultralytics source code.
    - Designed for xyxy bounding boxes.
    - Keeps iou_mean as a persistent registered buffer.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ultralytics.utils.metrics import bbox_iou


class WIoU(nn.Module):
    """
    Wise-IoU v3 with non-monotonic focusing.

    Expected input:
        pred_boxes:   (..., 4), xyxy
        target_boxes: (..., 4), xyxy

    Main WIoU formulation:

        WIoU = exp(l2_center / l2_box.detach()) * IoU

    Non-monotonic focusing:

        beta = IoU / iou_mean

        r = beta / (delta * alpha ** (beta - delta))

        loss = WIoU * r
    """

    def __init__(
        self,
        monotonous: bool = False,
        alpha: float = 1.9,
        delta: float = 3.0,
        momentum: float = 0.01,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.monotonous = monotonous
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.momentum = float(momentum)
        self.eps = float(eps)

        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha}"
            )

        if self.delta <= 0:
            raise ValueError(
                f"delta must be > 0, got {self.delta}"
            )

        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError(
                f"momentum must be in [0, 1], got {self.momentum}"
            )

        # Persistent state.
        #
        # This is intentionally a registered buffer:
        # - moves with the model to CUDA/CPU
        # - appears in state_dict
        # - survives checkpoint save/load
        self.register_buffer(
            "iou_mean",
            torch.tensor(1.0, dtype=torch.float32),
        )

    @staticmethod
    def _center_and_wh(
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert xyxy boxes to center coordinates and width/height.

        Returns:
            center: (..., 2)
            wh:     (..., 2)
        """

        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        w = (x2 - x1).clamp_min(0.0)
        h = (y2 - y1).clamp_min(0.0)

        center = torch.stack((cx, cy), dim=-1)
        wh = torch.stack((w, h), dim=-1)

        return center, wh

    def _calculate_components(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ):
        """
        Calculate the quantities required by WIoU.
        """

        pred_center, pred_wh = self._center_and_wh(pred_boxes)
        target_center, target_wh = self._center_and_wh(target_boxes)

        # ----------------------------------------------------------
        # Intersection / union
        # ----------------------------------------------------------
        pred_x1 = pred_boxes[..., 0]
        pred_y1 = pred_boxes[..., 1]
        pred_x2 = pred_boxes[..., 2]
        pred_y2 = pred_boxes[..., 3]

        target_x1 = target_boxes[..., 0]
        target_y1 = target_boxes[..., 1]
        target_x2 = target_boxes[..., 2]
        target_y2 = target_boxes[..., 3]

        inter_x1 = torch.maximum(pred_x1, target_x1)
        inter_y1 = torch.maximum(pred_y1, target_y1)
        inter_x2 = torch.minimum(pred_x2, target_x2)
        inter_y2 = torch.minimum(pred_y2, target_y2)

        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)

        inter_area = inter_w * inter_h

        pred_area = (
            pred_wh[..., 0] *
            pred_wh[..., 1]
        )

        target_area = (
            target_wh[..., 0] *
            target_wh[..., 1]
        )

        union_area = (
            pred_area
            + target_area
            - inter_area
        ).clamp_min(self.eps)

        # ----------------------------------------------------------
        # IoU
        # ----------------------------------------------------------
        iou = inter_area / union_area

        iou = iou.clamp(
            min=0.0,
            max=1.0,
        )

        # ----------------------------------------------------------
        # Center distance
        # ----------------------------------------------------------
        d_center = (
            pred_center
            - target_center
        )

        l2_center = torch.square(
            d_center
        ).sum(dim=-1)

        # ----------------------------------------------------------
        # Enclosing box
        # ----------------------------------------------------------
        enclosing_x1 = torch.minimum(
            pred_x1,
            target_x1,
        )

        enclosing_y1 = torch.minimum(
            pred_y1,
            target_y1,
        )

        enclosing_x2 = torch.maximum(
            pred_x2,
            target_x2,
        )

        enclosing_y2 = torch.maximum(
            pred_y2,
            target_y2,
        )

        enclosing_w = (
            enclosing_x2 - enclosing_x1
        ).clamp_min(self.eps)

        enclosing_h = (
            enclosing_y2 - enclosing_y1
        ).clamp_min(self.eps)

        # This corresponds to the diagonal squared of the enclosing box.
        l2_box = (
            torch.square(enclosing_w)
            + torch.square(enclosing_h)
        ).clamp_min(self.eps)

        return iou, l2_center, l2_box

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        ret_iou: bool = False,
    ):
        """
        Compute WIoU loss.

        Args:
            pred_boxes:
                Predicted boxes in xyxy format.

            target_boxes:
                Ground-truth boxes in xyxy format.

            ret_iou:
                If True, return:
                    (loss, iou)

                Otherwise:
                    loss

        Returns:
            Tensor or tuple[Tensor, Tensor]
        """

        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(
                "pred_boxes and target_boxes must have identical shapes. "
                f"Got {pred_boxes.shape} and {target_boxes.shape}."
            )

        if pred_boxes.ndim < 1 or pred_boxes.shape[-1] != 4:
            raise ValueError(
                "Boxes must have shape (..., 4) in xyxy format. "
                f"Got {pred_boxes.shape}."
            )

        # No positive samples.
        if pred_boxes.numel() == 0:
            empty = pred_boxes.new_zeros(
                pred_boxes.shape[:-1]
            )

            if ret_iou:
                return empty, empty

            return empty

        # ----------------------------------------------------------
        # IoU
        #
        # Use Ultralytics bbox_iou for consistency with the
        # installed framework.
        # ----------------------------------------------------------
        iou = bbox_iou(
            pred_boxes,
            target_boxes,
            xywh=False,
            CIoU=False,
        )

        iou = iou.clamp(
            min=0.0,
            max=1.0,
        )

        # ----------------------------------------------------------
        # WIoU geometric quantities
        # ----------------------------------------------------------
        _, l2_center, l2_box = self._calculate_components(
            pred_boxes,
            target_boxes,
        )

        # ----------------------------------------------------------
        # Running IoU mean
        # ----------------------------------------------------------
        if self.training:
            with torch.no_grad():
                batch_iou_mean = iou.detach().mean()

                self.iou_mean.mul_(
                    1.0 - self.momentum
                )

                self.iou_mean.add_(
                    self.momentum * batch_iou_mean
                )

        current_iou_mean = self.iou_mean.to(
            device=iou.device,
            dtype=iou.dtype,
        ).clamp_min(self.eps)

        # ----------------------------------------------------------
        # Base WIoU
        #
        # Official implementation:
        #
        #     dist = exp(l2_center / l2_box.detach())
        #     WIoU = dist * IoU
        # ----------------------------------------------------------
        distance_term = torch.exp(
            l2_center
            / l2_box.detach().clamp_min(self.eps)
        )

        # Prevent numerical overflow.
        distance_term = torch.clamp(
            distance_term,
            max=1e4,
        )

        wiou = distance_term * iou

        # ----------------------------------------------------------
        # Dynamic focusing
        # ----------------------------------------------------------
        beta = (
            iou.detach()
            / current_iou_mean
        ).clamp_min(self.eps)

        if self.monotonous:
            # Monotonic version.
            #
            # Official implementation uses sqrt(beta).
            focusing_factor = torch.sqrt(beta)

        else:
            # WIoU v3 non-monotonic focusing.
            divisor = (
                self.delta
                * torch.pow(
                    torch.tensor(
                        self.alpha,
                        device=beta.device,
                        dtype=beta.dtype,
                    ),
                    beta - self.delta,
                )
            )

            focusing_factor = (
                beta
                / divisor.clamp_min(self.eps)
            )

        loss = wiou * focusing_factor

        # Numerical protection.
        loss = torch.nan_to_num(
            loss,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )

        if ret_iou:
            return loss, iou

        return loss