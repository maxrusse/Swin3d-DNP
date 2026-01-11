"""Focal losses for Swin3D-DNP.

This module provides focal loss variants for keypoint/lesion detection,
including CornerNet-style focal loss for heatmap regression.

Tensor conventions:
- Spatial order: (D, H, W)
- valid_mask: 1.0 for valid voxels, 0.0 for invalid/padded
"""

import torch
import torch.nn.functional as F
from typing import Optional

from swin3d_dnp.constants import EPS_LOG


def focal_heatmap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = EPS_LOG,
) -> torch.Tensor:
    """CornerNet-style focal loss for keypoint/lesion heatmaps.

    This loss is specifically designed for heatmap regression where:
    - Positive locations (peaks) have target value 1.0
    - Negative locations have target values in [0, 1) based on
      distance from peaks (typically Gaussian-blurred)

    The loss focuses learning on hard examples:
    - Easy negatives (far from peaks) are down-weighted by (1-target)^beta
    - Easy positives (confident predictions) are down-weighted by (1-pred)^alpha

    Args:
        pred: (B, C, D, H, W) predicted heatmap values (after sigmoid)
        target: (B, C, D, H, W) ground truth heatmap (0-1, peaks at 1)
        valid_mask: (B, 1, D, H, W) binary mask for valid regions
        alpha: Focusing parameter for positive samples (default: 2.0)
        beta: Focusing parameter for negative samples (default: 4.0)
        eps: Small constant for numerical stability in log

    Returns:
        Scalar focal heatmap loss

    Notes:
        - pred should be sigmoid outputs (not raw logits)
        - target peaks should be exactly 1.0 (for proper masking)
        - Non-peak target values should be Gaussian-blurred [0, 1)
        - Loss is normalized by number of positive locations

    Reference:
        Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints"
    """
    # Clamp predictions for numerical stability
    pred = pred.clamp(eps, 1 - eps)

    # Identify positive (peak) and negative locations
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()

    # Apply valid_mask (broadcast across channels)
    pos_mask = pos_mask * valid_mask
    neg_mask = neg_mask * valid_mask

    # Positive loss: -((1-p)^alpha) * log(p) at peak locations
    # Down-weights easy positives (high confidence predictions)
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask

    # Negative loss: -((1-t)^beta) * (p^alpha) * log(1-p) at non-peak locations
    # Down-weights:
    #   - Easy negatives far from peaks (small target, large (1-t)^beta factor inverted - wait that's wrong)
    #   - Actually: (1-t)^beta is SMALL when t is large (near peak), so near-peak negatives
    #     contribute MORE, which makes sense - they're harder examples
    #   - p^alpha down-weights easy negatives with low predictions
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    # Total loss
    loss = pos_loss + neg_loss

    # Normalize by number of positive locations
    num_pos = pos_mask.sum()
    return loss.sum() / (num_pos + 1e-4)


def focal_heatmap_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = EPS_LOG,
) -> torch.Tensor:
    """CornerNet-style focal loss accepting raw logits.

    Same as focal_heatmap_loss but applies sigmoid internally.

    Args:
        logits: (B, C, D, H, W) raw network outputs (before sigmoid)
        target: (B, C, D, H, W) ground truth heatmap (0-1, peaks at 1)
        valid_mask: (B, 1, D, H, W) binary mask for valid regions
        alpha: Focusing parameter for positive samples (default: 2.0)
        beta: Focusing parameter for negative samples (default: 4.0)
        eps: Small constant for numerical stability

    Returns:
        Scalar focal heatmap loss
    """
    pred = torch.sigmoid(logits)
    return focal_heatmap_loss(pred, target, valid_mask, alpha, beta, eps)


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal loss for classification (Lin et al. style).

    Standard focal loss that down-weights well-classified examples
    to focus learning on hard examples.

    Args:
        logits: (B, C, D, H, W) raw network outputs
        target: (B, D, H, W) class indices (long tensor)
        valid_mask: (B, 1, D, H, W) binary mask
        gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)
        alpha: Optional (C,) class weights for addressing class imbalance

    Returns:
        Scalar focal cross-entropy loss

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection"
    """
    B, C, D, H, W = logits.shape
    vm = valid_mask[:, 0]  # (B, D, H, W)

    # Compute cross-entropy per voxel
    ce = F.cross_entropy(logits, target, weight=alpha, reduction="none")  # (B, D, H, W)

    # Get probability of correct class for focal modulation
    probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)

    # Gather probabilities for target class
    target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, D, H, W)

    # Focal modulation: (1 - p_t)^gamma
    focal_weight = (1 - target_probs) ** gamma

    # Apply focal weight and valid mask
    focal_loss = focal_weight * ce * vm

    return focal_loss.sum() / (vm.sum() + 1e-8)


def quality_focal_loss(
    pred: torch.Tensor,
    target_class: torch.Tensor,
    target_quality: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    """Quality Focal Loss for joint classification and quality estimation.

    QFL unifies classification and IoU/quality prediction by using
    a continuous quality score as the target instead of hard labels.

    Args:
        pred: (B, C, D, H, W) predicted class scores (after sigmoid)
        target_class: (B, D, H, W) class indices for positive locations
        target_quality: (B, D, H, W) quality scores [0, 1] (e.g., IoU with GT)
        valid_mask: (B, 1, D, H, W) binary mask
        beta: Focusing parameter (default: 2.0)

    Returns:
        Scalar quality focal loss

    Reference:
        Li et al., "Generalized Focal Loss"
    """
    B, C, D, H, W = pred.shape
    pred = pred.clamp(EPS_LOG, 1 - EPS_LOG)
    vm = valid_mask[:, 0]  # (B, D, H, W)

    # Create soft target: quality at target_class, 0 elsewhere
    target = torch.zeros_like(pred)
    target.scatter_(1, target_class.unsqueeze(1).long(), target_quality.unsqueeze(1))

    # QFL: |y - pred|^beta * BCE(pred, y)
    scale_factor = (target - pred).abs() ** beta

    # Binary cross-entropy
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

    loss = scale_factor * bce * vm.unsqueeze(1)

    return loss.sum() / (vm.sum() * C + 1e-8)


def offset_loss(
    pred_offset: torch.Tensor,
    target_offset: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Offset regression loss for sub-voxel localization.

    Used to predict sub-voxel offsets from discretized heatmap peaks.
    Only computed at positive (peak) locations.

    Args:
        pred_offset: (B, 3, D, H, W) predicted offsets (z, y, x)
        target_offset: (B, 3, D, H, W) ground truth offsets
        valid_mask: (B, 1, D, H, W) binary mask
        pos_mask: (B, 1, D, H, W) mask for peak locations only
        loss_type: "smooth_l1" or "l1"

    Returns:
        Scalar offset loss
    """
    # Only compute loss at positive locations
    mask = valid_mask * pos_mask  # (B, 1, D, H, W)

    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_offset, target_offset, reduction="none")
    else:  # l1
        loss = F.l1_loss(pred_offset, target_offset, reduction="none")

    # Sum over offset dimensions (z, y, x)
    loss = (loss * mask).sum()
    num_pos = mask.sum() * 3  # 3 offset dimensions

    return loss / (num_pos + 1e-8)
