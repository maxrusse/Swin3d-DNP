"""Loss functions for Swin3D-DNP.

This module provides all loss functions used in the hierarchical
coarse-to-fine training pipeline.

All losses support valid_mask to handle padded/out-of-bounds regions,
ensuring only valid voxels contribute to gradient computation.

Main exports:
    - masked_cross_entropy: CE loss with valid mask
    - masked_dice_loss: Dice loss with valid mask
    - focal_heatmap_loss: CornerNet-style focal for heatmaps
    - focal_cross_entropy_loss: Lin et al. focal for classification
"""

from swin3d_dnp.losses.ce import (
    masked_cross_entropy,
    masked_cross_entropy_per_class,
)
from swin3d_dnp.losses.dice import (
    masked_dice_loss,
    masked_dice_loss_per_class,
    masked_generalized_dice_loss,
    dice_score,
)
from swin3d_dnp.losses.focal import (
    focal_heatmap_loss,
    focal_heatmap_loss_from_logits,
    focal_cross_entropy_loss,
    quality_focal_loss,
    offset_loss,
)

__all__ = [
    # Cross-entropy losses
    "masked_cross_entropy",
    "masked_cross_entropy_per_class",
    # Dice losses
    "masked_dice_loss",
    "masked_dice_loss_per_class",
    "masked_generalized_dice_loss",
    "dice_score",
    # Focal losses
    "focal_heatmap_loss",
    "focal_heatmap_loss_from_logits",
    "focal_cross_entropy_loss",
    "quality_focal_loss",
    "offset_loss",
]
