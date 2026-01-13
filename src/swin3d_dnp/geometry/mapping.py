"""Coordinate mapping between full and coarse volumes via world space.

CRITICAL CONVENTION:
- Internal tensors use (z, y, x) order (D, H, W)
- NIfTI affines use (x, y, z) order (i, j, k)
- Convert zyx -> xyz before applying affine, xyz -> zyx after
"""

import torch
from torch import Tensor


def center_full_to_coarse_norm(
    center_full_index_zyx: Tensor,
    affine_full: Tensor,
    affine_coarse: Tensor,
    shape_coarse: tuple[int, int, int],
) -> Tensor:
    """Map full volume index to coarse normalized coordinates.

    This function converts a center point from full-resolution voxel indices
    to normalized [-1, 1] coordinates in the coarse volume space by going
    through world (mm) coordinates.

    Args:
        center_full_index_zyx: (B, 3) continuous indices in full volume (z, y, x).
        affine_full: (B, 4, 4) NIfTI affine for full volume (maps xyz indices to world).
        affine_coarse: (B, 4, 4) NIfTI affine for coarse volume.
        shape_coarse: (Dc, Hc, Wc) shape of coarse volume.

    Returns:
        (B, 3) normalized coordinates in coarse volume space (d, h, w order).
    """
    B = center_full_index_zyx.shape[0]
    device = center_full_index_zyx.device
    dtype = torch.float32

    # zyx -> xyz for affine multiply
    center_xyz = torch.stack(
        [
            center_full_index_zyx[:, 2],
            center_full_index_zyx[:, 1],
            center_full_index_zyx[:, 0],
        ],
        dim=1,
    ).to(device=device, dtype=dtype)

    # Homogeneous coordinates
    ones = torch.ones((B, 1), device=device, dtype=dtype)
    center_h = torch.cat([center_xyz, ones], dim=1)  # (B, 4)

    # Full index -> world
    A_full = affine_full.to(device=device, dtype=dtype)
    world = (A_full @ center_h[:, :, None])[:, :, 0]  # (B, 4)

    # World -> coarse index
    A_coarse = affine_coarse.to(device=device, dtype=dtype)
    A_coarse_inv = torch.linalg.inv(A_coarse)
    coarse_xyz = (A_coarse_inv @ world[:, :, None])[:, :3, 0]  # (B, 3) in xyz

    # xyz -> zyx (dhw)
    coarse_zyx = torch.stack(
        [coarse_xyz[:, 2], coarse_xyz[:, 1], coarse_xyz[:, 0]], dim=1
    )

    # Convert to normalized coordinates
    Dc, Hc, Wc = shape_coarse
    shape = torch.tensor([Dc, Hc, Wc], device=device, dtype=dtype)

    coarse_norm_dhw = 2.0 * (coarse_zyx + 0.5) / shape - 1.0
    return coarse_norm_dhw


def center_coarse_to_full_index(
    center_coarse_index_zyx: Tensor,
    affine_coarse: Tensor,
    affine_full: Tensor,
) -> Tensor:
    """Map coarse voxel index to full volume index via world space.

    This is the inverse of center_full_to_coarse_norm (without normalization).
    Used during inference to map coarse proposals back to full resolution.

    Args:
        center_coarse_index_zyx: (B, 3) continuous index in coarse volume (z, y, x).
        affine_coarse: (B, 4, 4) NIfTI affine for coarse volume.
        affine_full: (B, 4, 4) NIfTI affine for full volume.

    Returns:
        (B, 3) continuous index in full volume (z, y, x order).
    """
    B = center_coarse_index_zyx.shape[0]
    device = center_coarse_index_zyx.device
    dtype = torch.float32

    # zyx -> xyz for affine multiply
    center_xyz = torch.stack(
        [
            center_coarse_index_zyx[:, 2],
            center_coarse_index_zyx[:, 1],
            center_coarse_index_zyx[:, 0],
        ],
        dim=1,
    ).to(dtype)

    # Homogeneous coordinates
    ones = torch.ones((B, 1), device=device, dtype=dtype)
    center_h = torch.cat([center_xyz, ones], dim=1)  # (B, 4)

    # Coarse index -> world
    world = (affine_coarse.to(dtype) @ center_h[:, :, None])[:, :, 0]  # (B, 4)

    # World -> full index
    A_full_inv = torch.linalg.inv(affine_full.to(dtype))
    full_xyz = (A_full_inv @ world[:, :, None])[:, :3, 0]  # (B, 3)

    # xyz -> zyx
    return torch.stack([full_xyz[:, 2], full_xyz[:, 1], full_xyz[:, 0]], dim=1)


def center_coarse_norm_to_index(
    center_coarse_norm_dhw: Tensor,
    shape_coarse: tuple[int, int, int],
) -> Tensor:
    """Convert coarse normalized coordinates to coarse index coordinates.

    Args:
        center_coarse_norm_dhw: (B, 3) normalized coordinates (d, h, w).
        shape_coarse: (Dc, Hc, Wc) shape of coarse volume.

    Returns:
        (B, 3) index coordinates in coarse volume (z, y, x).
    """
    Dc, Hc, Wc = shape_coarse
    device = center_coarse_norm_dhw.device
    shape = torch.tensor([Dc, Hc, Wc], device=device, dtype=torch.float32)

    # n -> u: u = ((n + 1) * N) / 2 - 0.5
    return ((center_coarse_norm_dhw + 1.0) * shape) / 2.0 - 0.5
