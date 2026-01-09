"""Non-Maximum Suppression for 3D lesion/landmark detection.

This module implements anisotropic NMS operating in mm space,
handling different voxel spacings across dimensions.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from swin3d_dnp.constants import (
    DEFAULT_NMS_MIN_DIST_MM,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_NMS_TOPK,
)


def nms_3d_aniso_mm(
    prob: Tensor,
    spacing_mm: tuple[float, float, float] | Tensor,
    min_dist_mm: float = DEFAULT_NMS_MIN_DIST_MM,
    threshold: float = DEFAULT_NMS_THRESHOLD,
    topk: int = DEFAULT_NMS_TOPK,
) -> tuple[Tensor, Tensor]:
    """Anisotropic 3D NMS operating in mm-based distances.

    Performs non-maximum suppression on a 3D probability map using
    distances computed in physical (mm) space rather than voxel space.
    This handles anisotropic voxel spacing correctly.

    Algorithm:
    1. Apply local maxima detection using max pooling with anisotropic kernel
    2. Threshold candidates
    3. Sort by score descending
    4. Greedily suppress based on mm-distance

    Args:
        prob: (D, H, W) probability map.
        spacing_mm: (d, h, w) voxel spacing in mm.
        min_dist_mm: Minimum distance in mm between kept proposals.
        threshold: Probability threshold for candidate selection.
        topk: Maximum number of proposals to return.

    Returns:
        coords: (N, 3) tensor of kept proposal indices (z, y, x).
        scores: (N,) tensor of corresponding probability scores.
    """
    device = prob.device

    # Convert spacing to tensor if needed
    if isinstance(spacing_mm, (tuple, list)):
        spacing = torch.tensor(spacing_mm, device=device, dtype=torch.float32)
    else:
        spacing = spacing_mm.to(device=device, dtype=torch.float32)

    # Compute kernel size in voxels for each dimension
    # kernel should cover 2 * min_dist_mm in each direction
    k_vox = (2.0 * min_dist_mm / spacing).ceil().long()
    k_vox = torch.clamp(k_vox, min=1)
    # Ensure odd kernel size for symmetric pooling
    k_vox = k_vox + (1 - k_vox % 2)
    kd, kh, kw = k_vox.tolist()

    # Add batch and channel dims for F.max_pool3d
    p = prob[None, None]  # (1, 1, D, H, W)

    # Pad to maintain size after max pooling
    pad = (kw // 2, kw // 2, kh // 2, kh // 2, kd // 2, kd // 2)
    p_padded = F.pad(p, pad, mode="replicate")

    # Local maxima detection via max pooling
    mx = F.max_pool3d(p_padded, kernel_size=(kd, kh, kw), stride=1, padding=0)

    # Find points that are local maxima AND above threshold
    cand = (p[0, 0] == mx[0, 0]) & (prob > threshold)
    coords = cand.nonzero(as_tuple=False)  # (N, 3) with columns [z, y, x]

    if coords.numel() == 0:
        return coords, prob.new_zeros((0,))

    # Get scores for candidates
    scores = prob[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    coords = coords[order]
    scores = scores[order]

    # Convert to mm for distance computation
    coords_mm = coords.float() * spacing[None, :]

    # Greedy suppression in mm space
    keep_idx: list[int] = []
    keep_coords_mm: list[Tensor] = []

    for i in range(coords.shape[0]):
        if len(keep_idx) >= topk:
            break

        c_mm = coords_mm[i]
        suppressed = False

        for kc_mm in keep_coords_mm:
            dist_sq = ((c_mm - kc_mm) ** 2).sum()
            if dist_sq < min_dist_mm * min_dist_mm:
                suppressed = True
                break

        if not suppressed:
            keep_idx.append(i)
            keep_coords_mm.append(c_mm)

    if len(keep_idx) == 0:
        return prob.new_zeros((0, 3), dtype=torch.long), prob.new_zeros((0,))

    keep_idx_tensor = torch.tensor(keep_idx, device=device, dtype=torch.long)
    return coords[keep_idx_tensor], scores[keep_idx_tensor]


def nms_3d_isotropic(
    prob: Tensor,
    min_dist_vox: int = 5,
    threshold: float = DEFAULT_NMS_THRESHOLD,
    topk: int = DEFAULT_NMS_TOPK,
) -> tuple[Tensor, Tensor]:
    """Isotropic 3D NMS operating in voxel space.

    Simplified NMS for isotropic volumes where voxel distance equals mm distance
    (up to a constant scale factor).

    Args:
        prob: (D, H, W) probability map.
        min_dist_vox: Minimum distance in voxels between kept proposals.
        threshold: Probability threshold for candidate selection.
        topk: Maximum number of proposals to return.

    Returns:
        coords: (N, 3) tensor of kept proposal indices (z, y, x).
        scores: (N,) tensor of corresponding probability scores.
    """
    return nms_3d_aniso_mm(
        prob=prob,
        spacing_mm=(1.0, 1.0, 1.0),
        min_dist_mm=float(min_dist_vox),
        threshold=threshold,
        topk=topk,
    )
