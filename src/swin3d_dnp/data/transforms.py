"""Data transforms for Swin3D-DNP.

This module provides transforms for:
- Label downsampling for coarse supervision
- Data augmentation
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def downsample_label_coarse(
    label_full: Tensor,
    coarse_shape: tuple[int, int, int],
    is_binary_lesion: bool = False,
) -> Tensor:
    """Downsample full-resolution labels to coarse resolution.

    For multi-class labels, uses nearest-neighbor interpolation.
    For binary lesion masks, uses max pooling to preserve small lesions
    that might be lost with nearest interpolation.

    Args:
        label_full: (D, H, W) label tensor (long).
        coarse_shape: (Dc, Hc, Wc) target coarse shape.
        is_binary_lesion: If True, use max pooling to preserve small objects.

    Returns:
        (Dc, Hc, Wc) downsampled label tensor (long).
    """
    D, H, W = label_full.shape
    Dc, Hc, Wc = coarse_shape

    if is_binary_lesion:
        # For binary lesions, maxpool preserves small objects
        mask = (label_full > 0).float()[None, None]  # (1, 1, D, H, W)

        # Compute stride for each dimension
        sd = max(D // Dc, 1)
        sh = max(H // Hc, 1)
        sw = max(W // Wc, 1)

        pooled = F.max_pool3d(mask, kernel_size=(sd, sh, sw), stride=(sd, sh, sw))

        # Handle size mismatch (if not exactly divisible)
        if pooled.shape[2:] != (Dc, Hc, Wc):
            pooled = F.interpolate(pooled, size=(Dc, Hc, Wc), mode="nearest")

        return (pooled[0, 0] > 0.5).long()
    else:
        # Multi-class: use nearest interpolation
        label_5d = label_full.float()[None, None]  # (1, 1, D, H, W)
        downsampled = F.interpolate(label_5d, size=(Dc, Hc, Wc), mode="nearest")
        return downsampled[0, 0].long()


def downsample_image_coarse(
    image_full: Tensor,
    coarse_shape: tuple[int, int, int],
) -> Tensor:
    """Downsample full-resolution image to coarse resolution.

    Uses trilinear interpolation for smooth downsampling.

    Args:
        image_full: (C, D, H, W) or (D, H, W) image tensor (float).
        coarse_shape: (Dc, Hc, Wc) target coarse shape.

    Returns:
        Downsampled image tensor with same number of dimensions.
    """
    if image_full.dim() == 3:
        img_5d = image_full[None, None]
        result = F.interpolate(
            img_5d, size=coarse_shape, mode="trilinear", align_corners=False
        )
        return result[0, 0]
    elif image_full.dim() == 4:
        img_5d = image_full[None]
        result = F.interpolate(
            img_5d, size=coarse_shape, mode="trilinear", align_corners=False
        )
        return result[0]
    else:
        raise ValueError(f"Expected 3D or 4D input, got {image_full.dim()}D")


def random_rotation_matrix_3d(
    max_angle_deg: float = 15.0,
    device: torch.device | None = None,
) -> Tensor:
    """Generate a random 3D rotation matrix.

    Creates a rotation matrix with random angles up to max_angle_deg
    around each axis (x, y, z).

    Args:
        max_angle_deg: Maximum rotation angle in degrees.
        device: Device for output tensor.

    Returns:
        (3, 3) rotation matrix.
    """
    import math

    max_rad = max_angle_deg * math.pi / 180.0

    # Random angles for each axis
    angles = (torch.rand(3, device=device) * 2 - 1) * max_rad

    # Rotation matrices for each axis
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Rotation around x-axis
    Rx = torch.eye(3, device=device)
    Rx[1, 1] = cos[0]
    Rx[1, 2] = -sin[0]
    Rx[2, 1] = sin[0]
    Rx[2, 2] = cos[0]

    # Rotation around y-axis
    Ry = torch.eye(3, device=device)
    Ry[0, 0] = cos[1]
    Ry[0, 2] = sin[1]
    Ry[2, 0] = -sin[1]
    Ry[2, 2] = cos[1]

    # Rotation around z-axis
    Rz = torch.eye(3, device=device)
    Rz[0, 0] = cos[2]
    Rz[0, 1] = -sin[2]
    Rz[1, 0] = sin[2]
    Rz[1, 1] = cos[2]

    # Combined rotation: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def random_scale_factors(
    min_scale: float = 0.9,
    max_scale: float = 1.1,
    isotropic: bool = False,
    device: torch.device | None = None,
) -> Tensor:
    """Generate random scale factors for augmentation.

    Args:
        min_scale: Minimum scale factor.
        max_scale: Maximum scale factor.
        isotropic: If True, use same scale for all axes.
        device: Device for output tensor.

    Returns:
        (3,) scale factors for (x, y, z).
    """
    if isotropic:
        scale = torch.rand(1, device=device) * (max_scale - min_scale) + min_scale
        return scale.expand(3)
    else:
        return torch.rand(3, device=device) * (max_scale - min_scale) + min_scale


def random_translation_mm(
    max_translation_mm: float = 10.0,
    device: torch.device | None = None,
) -> Tensor:
    """Generate random translation vector for augmentation.

    Args:
        max_translation_mm: Maximum translation in mm for each axis.
        device: Device for output tensor.

    Returns:
        (3,) translation vector in mm for (x, y, z).
    """
    return (torch.rand(3, device=device) * 2 - 1) * max_translation_mm
