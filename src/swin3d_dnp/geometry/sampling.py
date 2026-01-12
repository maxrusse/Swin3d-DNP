"""Patch and context sampling utilities.

This module implements:
- Fine patch sampling from full volume with augmentation
- Differentiable coarse context sampling

CRITICAL INVARIANTS:
- align_corners=False for all grid_sample calls
- Labels use mode="nearest", padding_mode="zeros"
- Images use mode="bilinear", padding_mode="border"
- valid_mask produced for out-of-bounds regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from swin3d_dnp.geometry.coordinates import index_to_norm_acfalse


def extent_vox_in_src_from_spacings(
    out_shape: tuple[int, int, int],
    fine_spacing_dhw_mm: tuple[float, float, float] | Tensor,
    coarse_spacing_dhw_mm: tuple[float, float, float] | Tensor,
) -> tuple[float, float, float]:
    """Compute extent in source (coarse) voxels covering same physical FOV as output.

    The fine patch at fine spacing covers a certain physical FOV.
    This function computes how many coarse voxels span the same physical extent.

    Args:
        out_shape: (Df, Hf, Wf) output patch shape.
        fine_spacing_dhw_mm: (3,) or (B, 3) spacing of fine volume in mm.
            If batched, uses first sample (assumes uniform spacing across batch).
        coarse_spacing_dhw_mm: (3,) or (B, 3) spacing of coarse volume in mm.
            If batched, uses first sample.

    Returns:
        (Ld, Lh, Lw) extent in coarse voxels (float).
    """
    Df, Hf, Wf = out_shape

    if isinstance(fine_spacing_dhw_mm, Tensor):
        # Handle batch dimension: if (B, 3), take first sample
        if fine_spacing_dhw_mm.dim() == 2:
            fine_spacing_dhw_mm = fine_spacing_dhw_mm[0]
        sd, sh, sw = fine_spacing_dhw_mm.tolist()
    else:
        sd, sh, sw = fine_spacing_dhw_mm

    if isinstance(coarse_spacing_dhw_mm, Tensor):
        # Handle batch dimension: if (B, 3), take first sample
        if coarse_spacing_dhw_mm.dim() == 2:
            coarse_spacing_dhw_mm = coarse_spacing_dhw_mm[0]
        cd, ch, cw = coarse_spacing_dhw_mm.tolist()
    else:
        cd, ch, cw = coarse_spacing_dhw_mm

    return ((Df * sd) / cd, (Hf * sh) / ch, (Wf * sw) / cw)


def sample_patch_from_full(
    image_full: Tensor,
    label_full: Tensor | None,
    affine_full: Tensor,
    center_full_index_zyx: Tensor,
    out_shape: tuple[int, int, int],
    spacing_fine_dhw_mm: Tensor,
    R_world: Tensor | None = None,
    S_world: Tensor | None = None,
    t_world_mm: Tensor | None = None,
    clamp_grid: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Sample fine patches from full volume with optional augmentation.

    This is the canonical patch sampler. Augmentation is applied in world space:
    1. Scale the voxel offsets
    2. Rotate in world space
    3. Translate in world space

    Args:
        image_full: (B, 1, D, H, W) full resolution image.
        label_full: (B, 1, D, H, W) full resolution label or None.
        affine_full: (B, 4, 4) NIfTI affine for full volume.
        center_full_index_zyx: (B, 3) patch center in full volume indices (z, y, x).
        out_shape: (Df, Hf, Wf) output patch shape.
        spacing_fine_dhw_mm: (B, 3) fine voxel spacing in mm (d, h, w).
        R_world: (B, 3, 3) rotation matrix in world xyz axes. Default: identity.
        S_world: (B, 3) scale factors in xyz. Default: ones.
        t_world_mm: (B, 3) translation in xyz mm. Default: zeros.
        clamp_grid: Whether to clamp grid to [-1, 1]. Default: True.

    Returns:
        image_fine: (B, 1, Df, Hf, Wf) sampled image patch.
        label_fine: (B, 1, Df, Hf, Wf) sampled label or None.
        valid_mask: (B, 1, Df, Hf, Wf) mask of valid (in-bounds) voxels.
    """
    B, _, D, H, W = image_full.shape
    Df, Hf, Wf = out_shape
    device = image_full.device

    A = affine_full.to(device=device, dtype=torch.float32)
    A_inv = torch.linalg.inv(A)

    centers_zyx = center_full_index_zyx.to(device=device, dtype=torch.float32)

    # zyx -> xyz for affine
    centers_xyz = torch.stack(
        [centers_zyx[:, 2], centers_zyx[:, 1], centers_zyx[:, 0]], dim=1
    )

    sf_dhw = spacing_fine_dhw_mm.to(device=device, dtype=torch.float32)
    # spacing in xyz order for world perturbations
    sf_xyz = torch.stack([sf_dhw[:, 2], sf_dhw[:, 1], sf_dhw[:, 0]], dim=1)

    # Default augmentation parameters
    if R_world is None:
        R_world = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(B, 1, 1)
    else:
        R_world = R_world.to(device=device, dtype=torch.float32)

    if S_world is None:
        S_world = torch.ones((B, 3), device=device, dtype=torch.float32)
    else:
        S_world = S_world.to(device=device, dtype=torch.float32)

    if t_world_mm is None:
        t_world_mm = torch.zeros((B, 3), device=device, dtype=torch.float32)
    else:
        t_world_mm = t_world_mm.to(device=device, dtype=torch.float32)

    # World center in xyz
    ones = torch.ones((B, 1), device=device, dtype=torch.float32)
    center_h = torch.cat([centers_xyz, ones], dim=1)  # (B, 4)
    world_center = (A @ center_h[:, :, None])[:, :, 0]  # (B, 4) xyz world

    # Patch grid in zyx index coordinates
    z = torch.arange(Df, device=device, dtype=torch.float32)
    y = torch.arange(Hf, device=device, dtype=torch.float32)
    x = torch.arange(Wf, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    z0 = (Df - 1) / 2.0
    y0 = (Hf - 1) / 2.0
    x0 = (Wf - 1) / 2.0

    # Centered voxel offsets (zyx)
    t_zyx = torch.stack([zz - z0, yy - y0, xx - x0], dim=-1)  # (Df, Hf, Wf, 3)

    # Convert to xyz mm using spacing_fine
    t_xyz = torch.stack([t_zyx[..., 2], t_zyx[..., 1], t_zyx[..., 0]], dim=-1)
    t_mm = t_xyz[None] * sf_xyz[:, None, None, None, :]  # (B, Df, Hf, Wf, 3)

    # Scale then rotate then translate in world xyz
    t_mm = t_mm * S_world[:, None, None, None, :]
    t_mm = torch.einsum("bij,bdhwj->bdhwi", R_world, t_mm)
    t_mm = t_mm + t_world_mm[:, None, None, None, :]

    # World coords (xyz, 1)
    world = torch.zeros((B, Df, Hf, Wf, 4), device=device, dtype=torch.float32)
    world[..., 0:3] = world_center[:, None, None, None, 0:3] + t_mm
    world[..., 3] = 1.0

    # Map world xyz to full index xyz
    u_xyz = torch.einsum("bij,bdhwj->bdhwi", A_inv, world)[..., 0:3]

    # xyz -> normalized full coords (x, y, z), align_corners=False
    nx = index_to_norm_acfalse(u_xyz[..., 0], W)  # x uses W
    ny = index_to_norm_acfalse(u_xyz[..., 1], H)
    nz = index_to_norm_acfalse(u_xyz[..., 2], D)

    # Valid mask
    valid = (nx.abs() <= 1.0) & (ny.abs() <= 1.0) & (nz.abs() <= 1.0)
    valid_mask = valid.to(torch.float32)[:, None]

    # grid_sample grid is (x, y, z)
    grid = torch.stack([nx, ny, nz], dim=-1)
    if clamp_grid:
        grid = grid.clamp(-1.0, 1.0)

    img = F.grid_sample(
        image_full,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    lbl = None
    if label_full is not None:
        lblf = label_full.to(device=device, dtype=torch.float32)
        lbl = F.grid_sample(
            lblf,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

    return img, lbl, valid_mask


class DifferentiableContextSampler(nn.Module):
    """Sample coarse context features at the same physical FOV as fine patch.

    This module samples from coarse feature maps using differentiable grid_sample,
    allowing gradients to flow back through the coarse network.
    """

    def __init__(self, padding_mode: str = "border", clamp_grid: bool = True):
        """Initialize context sampler.

        Args:
            padding_mode: Padding mode for grid_sample ("border" or "zeros").
            clamp_grid: Whether to clamp grid coordinates to [-1, 1].
        """
        super().__init__()
        self.padding_mode = padding_mode
        self.clamp_grid = clamp_grid

    @staticmethod
    def _axis_grid(
        center_norm_f32: Tensor,
        out_len: int,
        src_len: int,
        extent_vox_f32: float,
    ) -> Tensor:
        """Build sampling grid for one axis.

        Args:
            center_norm_f32: (B,) normalized center coordinates.
            out_len: Output size for this axis.
            src_len: Source size for this axis.
            extent_vox_f32: Extent in source voxels.

        Returns:
            (B, out_len) grid coordinates for this axis.
        """
        device = center_norm_f32.device
        i = torch.arange(out_len, device=device, dtype=torch.float32)
        i0 = (out_len - 1) / 2.0
        du = (i - i0) * (float(extent_vox_f32) / float(out_len))
        dn = 2.0 * du / float(src_len)
        return center_norm_f32[:, None] + dn[None, :]

    def forward(
        self,
        src: Tensor,
        centers_dhw_norm: Tensor,
        out_shape: tuple[int, int, int],
        extent_vox_in_src: tuple[float, float, float],
    ) -> Tensor:
        """Sample context from source features.

        Args:
            src: (B, C, Dg, Hg, Wg) source feature map.
            centers_dhw_norm: (B, 3) normalized centers in [-1, 1] (d, h, w order).
            out_shape: (Df, Hf, Wf) output shape.
            extent_vox_in_src: (Ld, Lh, Lw) extent in source voxels.

        Returns:
            (B, C, Df, Hf, Wf) sampled context features.
        """
        B, C, Dg, Hg, Wg = src.shape
        Df, Hf, Wf = out_shape
        Ld, Lh, Lw = extent_vox_in_src

        centers = centers_dhw_norm.to(device=src.device, dtype=torch.float32)
        z = self._axis_grid(centers[:, 0], Df, Dg, Ld)
        y = self._axis_grid(centers[:, 1], Hf, Hg, Lh)
        x = self._axis_grid(centers[:, 2], Wf, Wg, Lw)

        zz = z[:, :, None, None].expand(B, Df, Hf, Wf)
        yy = y[:, None, :, None].expand(B, Df, Hf, Wf)
        xx = x[:, None, None, :].expand(B, Df, Hf, Wf)

        grid = torch.stack([xx, yy, zz], dim=-1)  # (x, y, z) for grid_sample
        if self.clamp_grid:
            grid = grid.clamp(-1.0, 1.0)

        return F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
