"""Coordinate transformation utilities.

This module implements the coordinate transformations between:
- Index space (voxel centers): u ∈ [0, N-1]
- Normalized grid space (align_corners=False): n ∈ [-1, 1]

CRITICAL: All functions assume align_corners=False convention.
"""

import torch
from torch import Tensor


def index_to_norm_acfalse(u: Tensor | float, N: int | Tensor) -> Tensor:
    """Convert index coordinates to normalized coordinates (align_corners=False).

    Maps voxel center indices to the [-1, 1] range used by grid_sample.

    Formula: n(u) = 2.0 * (u + 0.5) / N - 1.0

    Args:
        u: Index coordinates (voxel centers), can be any shape.
        N: Size of the dimension (number of voxels).

    Returns:
        Normalized coordinates in [-1, 1].

    Example:
        >>> index_to_norm_acfalse(torch.tensor([0.0, 4.5, 9.0]), 10)
        tensor([-0.9000, 0.0000, 0.9000])
    """
    return 2.0 * (u + 0.5) / float(N) - 1.0


def norm_to_index_acfalse(n: Tensor | float, N: int | Tensor) -> Tensor:
    """Convert normalized coordinates to index coordinates (align_corners=False).

    Maps [-1, 1] normalized coordinates back to voxel center indices.

    Formula: u(n) = ((n + 1.0) * N) / 2.0 - 0.5

    Args:
        n: Normalized coordinates in [-1, 1].
        N: Size of the dimension (number of voxels).

    Returns:
        Index coordinates (voxel centers).

    Example:
        >>> norm_to_index_acfalse(torch.tensor([-0.9, 0.0, 0.9]), 10)
        tensor([0.0000, 4.5000, 9.0000])
    """
    return ((n + 1.0) * float(N)) / 2.0 - 0.5


def patch_center_index(start: Tensor, size: Tensor) -> Tensor:
    """Compute patch center in index space.

    For a patch with start index s and size S, the center is:
    c = s + (S - 1) / 2 = s + S / 2 - 0.5

    Args:
        start: Start indices of the patch (z, y, x).
        size: Size of the patch (D, H, W).

    Returns:
        Center indices (z, y, x) as float.
    """
    return start.float() + (size.float() - 1) / 2.0


def patch_center_norm(start: Tensor, size: Tensor, volume_size: Tensor) -> Tensor:
    """Compute patch center in normalized coordinates.

    Formula: n(c) = 2 * (c + 0.5) / N - 1 = 2 * (s + S/2) / N - 1

    Args:
        start: Start indices of the patch (z, y, x).
        size: Size of the patch (D, H, W).
        volume_size: Size of the full volume (D, H, W).

    Returns:
        Normalized center coordinates.
    """
    center = patch_center_index(start, size)
    return 2.0 * (center + 0.5) / volume_size.float() - 1.0
