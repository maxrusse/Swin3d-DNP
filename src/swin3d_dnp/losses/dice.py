"""Masked Dice loss for Swin3D-DNP.

This module provides Dice-based losses that respect valid_mask,
ensuring padded/out-of-bounds regions do not contribute to the loss.

Tensor conventions:
- Spatial order: (D, H, W)
- valid_mask: 1.0 for valid voxels, 0.0 for invalid/padded
"""

import torch
import torch.nn.functional as F
from typing import Optional

from swin3d_dnp.constants import EPS_DICE


def _ensure_one_hot(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert target to one-hot encoding if needed.

    Args:
        pred: (B, C, D, H, W) prediction tensor (used for shape reference).
        target: (B, 1, D, H, W) class indices OR (B, C, D, H, W) one-hot.

    Returns:
        (B, C, D, H, W) one-hot encoded target.
    """
    if target.shape[1] == 1:
        target_oh = torch.zeros_like(pred)
        target_oh.scatter_(1, target.long(), 1.0)
        return target_oh
    return target.float()


def _logits_to_probs(logits: torch.Tensor, apply_softmax: bool) -> torch.Tensor:
    """Convert logits to probabilities if requested.

    Args:
        logits: (B, C, D, H, W) raw logits or probabilities.
        apply_softmax: If True, apply softmax (multi-class) or sigmoid (binary).

    Returns:
        (B, C, D, H, W) probability tensor.
    """
    if not apply_softmax:
        return logits
    C = logits.shape[1]
    if C > 1:
        return torch.softmax(logits, dim=1)
    return torch.sigmoid(logits)


def masked_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = EPS_DICE,
    apply_softmax: bool = True,
    exclude_background: bool = False,
) -> torch.Tensor:
    """Compute Dice loss masked by valid regions.

    Dice coefficient: 2 * |P ∩ G| / (|P| + |G|)
    Dice loss: 1 - Dice

    Only voxels where valid_mask == 1 contribute to intersection
    and union calculations.

    Args:
        pred: (B, C, D, H, W) raw logits or probabilities
        target: (B, 1, D, H, W) class indices OR (B, C, D, H, W) one-hot
        valid_mask: (B, 1, D, H, W) binary mask, 1.0 = valid
        smooth: Smoothing factor for numerical stability (default: EPS_DICE)
        apply_softmax: If True, apply softmax to pred (set False if already probabilities)
        exclude_background: If True, exclude class 0 from loss

    Returns:
        Scalar loss tensor (mean Dice loss across valid classes)

    Notes:
        - Classes with no valid voxels are excluded from the mean
        - Perfect prediction gives Dice ≈ 1, loss ≈ 0
        - smooth factor prevents division by zero and provides gradient
          when both pred and target are zero
    """
    B, C, D, H, W = pred.shape

    target_oh = _ensure_one_hot(pred, target)
    pred_prob = _logits_to_probs(pred, apply_softmax)

    # Apply valid mask - broadcast across channels
    vm = valid_mask  # (B, 1, D, H, W)
    pred_masked = pred_prob * vm  # (B, C, D, H, W)
    target_masked = target_oh * vm  # (B, C, D, H, W)

    # Compute Dice per class
    # Sum over spatial dimensions (D, H, W)
    dims = (2, 3, 4)

    intersection = (pred_masked * target_masked).sum(dim=dims)  # (B, C)
    pred_sum = pred_masked.sum(dim=dims)  # (B, C)
    target_sum = target_masked.sum(dim=dims)  # (B, C)

    # Dice coefficient: 2 * |P ∩ G| / (|P| + |G|)
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)  # (B, C)

    # Dice loss
    dice_loss = 1.0 - dice  # (B, C)

    # Determine which classes have valid voxels (for weighted mean)
    # A class is valid if it has any valid voxels in the batch
    valid_per_class = vm.expand_as(pred_prob).sum(dim=dims)  # (B, C)
    class_weight = (valid_per_class > 0).float()  # (B, C)

    # Optionally exclude background
    if exclude_background and C > 1:
        class_weight[:, 0] = 0.0

    # Weighted mean across batch and classes
    weighted_loss = (dice_loss * class_weight).sum()
    weight_sum = class_weight.sum()

    return weighted_loss / (weight_sum + 1e-8)


def masked_dice_loss_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = EPS_DICE,
    apply_softmax: bool = True,
) -> torch.Tensor:
    """Compute per-class Dice losses for analysis.

    Useful for monitoring class-specific segmentation quality.

    Args:
        pred: (B, C, D, H, W) raw logits or probabilities
        target: (B, 1, D, H, W) class indices OR (B, C, D, H, W) one-hot
        valid_mask: (B, 1, D, H, W) binary mask
        smooth: Smoothing factor for numerical stability
        apply_softmax: If True, apply softmax to pred

    Returns:
        (C,) tensor with per-class Dice losses
    """
    target_oh = _ensure_one_hot(pred, target)
    pred_prob = _logits_to_probs(pred, apply_softmax)

    pred_masked = pred_prob * valid_mask
    target_masked = target_oh * valid_mask

    dims = (0, 2, 3, 4)  # Sum over batch and spatial

    intersection = (pred_masked * target_masked).sum(dim=dims)  # (C,)
    pred_sum = pred_masked.sum(dim=dims)
    target_sum = target_masked.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1.0 - dice


def masked_generalized_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = EPS_DICE,
    apply_softmax: bool = True,
    weight_type: str = "square",
) -> torch.Tensor:
    """Compute Generalized Dice Loss with class volume weighting.

    GDL addresses class imbalance by weighting each class inversely
    proportional to its volume (number of voxels).

    Args:
        pred: (B, C, D, H, W) raw logits or probabilities
        target: (B, 1, D, H, W) class indices OR (B, C, D, H, W) one-hot
        valid_mask: (B, 1, D, H, W) binary mask
        smooth: Smoothing factor for numerical stability
        apply_softmax: If True, apply softmax to pred
        weight_type: "square" for 1/volume^2, "simple" for 1/volume

    Returns:
        Scalar Generalized Dice Loss

    Reference:
        Sudre et al., "Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations"
    """
    target_oh = _ensure_one_hot(pred, target)
    pred_prob = _logits_to_probs(pred, apply_softmax)

    pred_masked = pred_prob * valid_mask
    target_masked = target_oh * valid_mask

    dims = (2, 3, 4)  # Sum over spatial

    # Class volumes for weighting
    target_volume = target_masked.sum(dim=dims)  # (B, C)

    if weight_type == "square":
        class_weights = 1.0 / (target_volume ** 2 + smooth)
    else:  # simple
        class_weights = 1.0 / (target_volume + smooth)

    # Weighted intersection and union
    intersection = (pred_masked * target_masked).sum(dim=dims)  # (B, C)
    pred_sum = pred_masked.sum(dim=dims)
    target_sum = target_masked.sum(dim=dims)

    # Weighted Dice
    numerator = 2.0 * (class_weights * intersection).sum(dim=1) + smooth  # (B,)
    denominator = (class_weights * (pred_sum + target_sum)).sum(dim=1) + smooth  # (B,)

    gdl = 1.0 - (numerator / denominator)
    return gdl.mean()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    smooth: float = EPS_DICE,
) -> torch.Tensor:
    """Compute Dice score (coefficient) for evaluation.

    Unlike dice_loss, this returns the actual Dice coefficient
    (higher is better, range [0, 1]).

    Args:
        pred: (B, C, D, H, W) predicted probabilities or binary mask
        target: (B, C, D, H, W) ground truth (same format as pred)
        valid_mask: Optional (B, 1, D, H, W) binary mask
        smooth: Smoothing factor

    Returns:
        (C,) tensor with per-class Dice scores
    """
    B, C, D, H, W = pred.shape

    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask

    dims = (0, 2, 3, 4)  # Sum over batch and spatial

    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice
