"""Masked cross-entropy loss for Swin3D-DNP.

This module provides a cross-entropy loss that respects valid_mask,
ensuring padded/out-of-bounds regions do not contribute to the loss.

Tensor conventions:
- Spatial order: (D, H, W)
- valid_mask: 1.0 for valid voxels, 0.0 for invalid/padded
"""

import torch
import torch.nn.functional as F
from typing import Optional


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute cross-entropy loss masked by valid regions.

    Only voxels where valid_mask == 1 contribute to the loss.
    This is critical for handling patches that extend beyond
    volume boundaries.

    Args:
        logits: (B, C, D, H, W) raw network outputs (before softmax)
        target: (B, D, H, W) ground truth class indices (long tensor)
        valid_mask: (B, 1, D, H, W) binary mask, 1.0 = valid, 0.0 = invalid
        class_weights: Optional (C,) tensor for class weighting
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar loss tensor

    Notes:
        - Division by valid count includes small epsilon for stability
        - When valid_mask is all zeros, returns 0.0 (no loss)
        - target values in invalid regions are ignored
    """
    # Validate inputs
    B, C, D, H, W = logits.shape
    assert target.shape == (B, D, H, W), f"target shape {target.shape} != expected ({B}, {D}, {H}, {W})"
    assert valid_mask.shape == (B, 1, D, H, W), f"valid_mask shape {valid_mask.shape} != expected ({B}, 1, {D}, {H}, {W})"

    # Compute per-voxel cross-entropy
    # F.cross_entropy with reduction="none" gives (B, D, H, W)
    loss_per_voxel = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # (B, D, H, W)

    # Apply valid mask (squeeze the channel dimension)
    vm = valid_mask[:, 0]  # (B, D, H, W)

    # Masked mean: sum(loss * mask) / sum(mask)
    masked_loss = (loss_per_voxel * vm).sum()
    valid_count = vm.sum()

    # Epsilon prevents division by zero when mask is all zeros
    return masked_loss / (valid_count + 1e-8)


def masked_cross_entropy_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute per-class cross-entropy losses for analysis.

    Useful for monitoring class-specific training progress.

    Args:
        logits: (B, C, D, H, W) raw network outputs
        target: (B, D, H, W) ground truth class indices
        valid_mask: (B, 1, D, H, W) binary mask
        num_classes: Number of classes (C)

    Returns:
        (C,) tensor with per-class mean losses
    """
    B, C, D, H, W = logits.shape
    assert C == num_classes

    vm = valid_mask[:, 0]  # (B, D, H, W)

    # Per-voxel loss
    loss_per_voxel = F.cross_entropy(logits, target, reduction="none")  # (B, D, H, W)

    per_class_losses = []
    for c in range(num_classes):
        class_mask = (target == c).float() * vm
        class_loss = (loss_per_voxel * class_mask).sum()
        class_count = class_mask.sum()
        per_class_losses.append(class_loss / (class_count + 1e-8))

    return torch.stack(per_class_losses)
