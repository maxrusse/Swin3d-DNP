"""Patch stitching utilities for inference.

This module implements weighted overlap-add stitching for combining
overlapping patch predictions into a full volume output.

The cos^2 window provides smooth blending at patch boundaries while
ensuring that overlapping regions sum to approximately 1.
"""

import math
from typing import Sequence

import torch
from torch import Tensor

from swin3d_dnp.constants import EPS_STITCH


def cos2_window_1d(n: int, device: torch.device | None = None) -> Tensor:
    """Create a 1D cos^2 window for smooth blending.

    The window is computed as sin^2(pi * (i + 0.5) / n), which gives:
    - Values near 0 at edges (but non-zero for numerical stability)
    - Maximum value of 1 at the center
    - Smooth transition suitable for overlap-add

    Args:
        n: Length of the window.
        device: Device to create tensor on.

    Returns:
        (n,) tensor of window values in [0, 1].
    """
    i = torch.arange(n, device=device, dtype=torch.float32)
    x = math.pi * (i + 0.5) / float(n)
    w = torch.sin(x)
    return w * w


def cos2_window_3d(
    shape: tuple[int, int, int], device: torch.device | None = None
) -> Tensor:
    """Create a 3D cos^2 window by outer product of 1D windows.

    The window is separable: w_3d[d,h,w] = w_d[d] * w_h[h] * w_w[w].

    Args:
        shape: (D, H, W) shape of the output window.
        device: Device to create tensor on.

    Returns:
        (D, H, W) tensor of window values.
    """
    d, h, w = shape
    wz = cos2_window_1d(d, device)
    wy = cos2_window_1d(h, device)
    wx = cos2_window_1d(w, device)
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def stitch_patches_to_volume(
    patch_logits_list: Sequence[Tensor],
    patch_starts_zyx_list: Sequence[Tensor],
    full_shape: tuple[int, int, int],
    num_classes: int,
    window: Tensor,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Stitch overlapping patches into a full volume using weighted overlap-add.

    This function combines overlapping patch predictions by weighting each
    patch contribution with a window function and normalizing by the sum
    of weights at each location.

    For a location covered by patches P_1, P_2, ..., P_k with windows w_i:
        output = sum(w_i * logits_i) / sum(w_i)

    Args:
        patch_logits_list: List of (C, Df, Hf, Wf) tensors with patch predictions.
        patch_starts_zyx_list: List of (3,) tensors with patch start indices (z, y, x).
        full_shape: (D, H, W) shape of the output volume.
        num_classes: Number of output classes (C).
        window: (Df, Hf, Wf) weighting window (typically from cos2_window_3d).
        device: Device for output tensor.

    Returns:
        (num_classes, D, H, W) stitched output volume.
    """
    D, H, W = full_shape
    Df, Hf, Wf = window.shape

    # Accumulators for weighted sum
    num = torch.zeros((num_classes, D, H, W), device=device, dtype=torch.float32)
    den = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    window = window.to(device=device, dtype=torch.float32)

    for logits, start in zip(patch_logits_list, patch_starts_zyx_list):
        sz, sy, sx = int(start[0]), int(start[1]), int(start[2])

        # Compute source slices (region of patch to use)
        src_s = [max(0, -sz), max(0, -sy), max(0, -sx)]
        src_e = [
            Df - max(0, sz + Df - D),
            Hf - max(0, sy + Hf - H),
            Wf - max(0, sx + Wf - W),
        ]

        # Compute destination slices (region in output volume)
        dst_s = [max(0, sz), max(0, sy), max(0, sx)]
        dst_e = [min(D, sz + Df), min(H, sy + Hf), min(W, sx + Wf)]

        # Extract valid regions
        w = window[src_s[0] : src_e[0], src_s[1] : src_e[1], src_s[2] : src_e[2]]
        logits_dev = logits.to(device=device, dtype=torch.float32)
        patch_region = logits_dev[
            :, src_s[0] : src_e[0], src_s[1] : src_e[1], src_s[2] : src_e[2]
        ]

        # Accumulate weighted contributions
        num[:, dst_s[0] : dst_e[0], dst_s[1] : dst_e[1], dst_s[2] : dst_e[2]] += (
            patch_region * w
        )
        den[:, dst_s[0] : dst_e[0], dst_s[1] : dst_e[1], dst_s[2] : dst_e[2]] += w

    # Normalize by weight sum (with epsilon for numerical stability)
    return num / (den + EPS_STITCH)


def generate_tile_positions(
    full_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    stride: tuple[int, int, int] | None = None,
    overlap_fraction: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Generate patch start positions for dense tiling.

    Creates a regular grid of patch positions that covers the entire volume
    with the specified overlap.

    Args:
        full_shape: (D, H, W) shape of the full volume.
        patch_shape: (Df, Hf, Wf) shape of each patch.
        stride: (sd, sh, sw) step between patches. If None, computed from overlap_fraction.
        overlap_fraction: Fraction of overlap (0.5 = 50% overlap). Used if stride is None.

    Returns:
        List of (z, y, x) start positions for each patch.
    """
    D, H, W = full_shape
    Df, Hf, Wf = patch_shape

    if stride is None:
        # Compute stride from overlap fraction
        stride = (
            max(1, int(Df * (1 - overlap_fraction))),
            max(1, int(Hf * (1 - overlap_fraction))),
            max(1, int(Wf * (1 - overlap_fraction))),
        )

    sd, sh, sw = stride

    positions = []
    for sz in range(0, max(D - Df + 1, 1), sd):
        for sy in range(0, max(H - Hf + 1, 1), sh):
            for sx in range(0, max(W - Wf + 1, 1), sw):
                positions.append((sz, sy, sx))

    # Ensure we cover the edges by adding boundary patches if needed
    # (only if the last patch doesn't reach the edge)
    last_z = D - Df
    last_y = H - Hf
    last_x = W - Wf

    # Check if we need to add edge patches
    if last_z > 0:
        existing_z = {p[0] for p in positions}
        if last_z not in existing_z:
            for sy in range(0, max(H - Hf + 1, 1), sh):
                for sx in range(0, max(W - Wf + 1, 1), sw):
                    positions.append((last_z, sy, sx))

    if last_y > 0:
        existing_y = {p[1] for p in positions}
        if last_y not in existing_y:
            for sz in range(0, max(D - Df + 1, 1), sd):
                for sx in range(0, max(W - Wf + 1, 1), sw):
                    positions.append((sz, last_y, sx))

    if last_x > 0:
        existing_x = {p[2] for p in positions}
        if last_x not in existing_x:
            for sz in range(0, max(D - Df + 1, 1), sd):
                for sy in range(0, max(H - Hf + 1, 1), sh):
                    positions.append((sz, sy, last_x))

    # Add corner patches
    if last_z > 0 and last_y > 0:
        for sx in range(0, max(W - Wf + 1, 1), sw):
            positions.append((last_z, last_y, sx))
    if last_z > 0 and last_x > 0:
        for sy in range(0, max(H - Hf + 1, 1), sh):
            positions.append((last_z, sy, last_x))
    if last_y > 0 and last_x > 0:
        for sz in range(0, max(D - Df + 1, 1), sd):
            positions.append((sz, last_y, last_x))

    # Add final corner
    if last_z > 0 and last_y > 0 and last_x > 0:
        positions.append((last_z, last_y, last_x))

    # Remove duplicates while preserving order
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)

    return unique_positions
