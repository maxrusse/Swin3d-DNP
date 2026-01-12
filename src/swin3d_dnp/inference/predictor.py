"""Inference predictor for Swin3D-DNP.

This module implements two inference modes:
1. Proposal-driven inference: For lesions/landmarks using NMS proposals
2. Dense tiling inference: For organs using overlapping grid of patches

Both modes use the coarse network for initial predictions and global context,
then refine with the fine network at selected locations.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from swin3d_dnp.geometry.mapping import (
    center_coarse_to_full_index,
    center_full_to_coarse_norm,
)
from swin3d_dnp.geometry.sampling import (
    DifferentiableContextSampler,
    extent_vox_in_src_from_spacings,
    sample_patch_from_full,
)
from swin3d_dnp.inference.nms import nms_3d_aniso_mm
from swin3d_dnp.inference.stitching import (
    cos2_window_3d,
    generate_tile_positions,
    stitch_patches_to_volume,
)


@dataclass
class InferenceConfig:
    """Configuration for inference predictor.

    Attributes:
        mode: Inference mode ("proposal" or "dense").
        fine_patch_shape: Shape of fine patches (D, H, W).
        batch_size: Number of patches to process simultaneously.
        nms_min_dist_mm: Minimum distance between NMS proposals (proposal mode).
        nms_threshold: Probability threshold for NMS candidates (proposal mode).
        nms_topk: Maximum number of proposals (proposal mode).
        overlap_fraction: Overlap between tiles (dense mode).
        use_amp: Whether to use automatic mixed precision.
    """

    mode: Literal["proposal", "dense"] = "proposal"
    fine_patch_shape: tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 4
    nms_min_dist_mm: float = 10.0
    nms_threshold: float = 0.5
    nms_topk: int = 64
    overlap_fraction: float = 0.5
    use_amp: bool = True


class Predictor:
    """Inference predictor for Swin3D-DNP model.

    Supports two inference modes:
    1. Proposal-driven: Run NMS on coarse predictions, refine at proposal locations
    2. Dense tiling: Tile the volume with overlapping patches, stitch results

    Example:
        >>> model = build_swin3d_dnp(...)
        >>> predictor = Predictor(model, config)
        >>> result = predictor.predict(image_full, affine_full, spacing_full)
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: torch.device | str = "cuda",
    ):
        """Initialize predictor.

        Args:
            model: Trained Swin3D-DNP model.
            config: Inference configuration.
            device: Device for inference.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Context sampler for fine patch extraction
        self.context_sampler = DifferentiableContextSampler()

        # Precompute stitching window
        self.window = cos2_window_3d(config.fine_patch_shape, device=self.device)

    def _downsample_to_coarse(
        self,
        image_full: Tensor,
        coarse_shape: tuple[int, int, int],
    ) -> Tensor:
        """Downsample full image to coarse resolution.

        Args:
            image_full: (1, 1, D, H, W) full resolution image.
            coarse_shape: (Dc, Hc, Wc) target coarse shape.

        Returns:
            (1, 1, Dc, Hc, Wc) downsampled image.
        """
        return torch.nn.functional.interpolate(
            image_full,
            size=coarse_shape,
            mode="trilinear",
            align_corners=False,
        )

    def _compute_coarse_affine(
        self,
        affine_full: Tensor,
        full_shape: tuple[int, int, int],
        coarse_shape: tuple[int, int, int],
    ) -> Tensor:
        """Compute coarse volume affine from full volume affine.

        Args:
            affine_full: (4, 4) NIfTI affine for full volume.
            full_shape: (D, H, W) full volume shape.
            coarse_shape: (Dc, Hc, Wc) coarse volume shape.

        Returns:
            (4, 4) affine for coarse volume.
        """
        D, H, W = full_shape
        Dc, Hc, Wc = coarse_shape

        # Scale factors (in xyz order for NIfTI affine)
        scale = torch.tensor(
            [W / Wc, H / Hc, D / Dc, 1.0],
            device=affine_full.device,
            dtype=affine_full.dtype,
        )

        # Scale the affine (multiply columns by scale)
        affine_coarse = affine_full.clone()
        affine_coarse[:, 0] *= scale[0]
        affine_coarse[:, 1] *= scale[1]
        affine_coarse[:, 2] *= scale[2]

        return affine_coarse

    def _compute_spacings(
        self,
        affine_full: Tensor,
        full_shape: tuple[int, int, int],
        coarse_shape: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        """Compute spacings for full and coarse volumes.

        Args:
            affine_full: (4, 4) NIfTI affine.
            full_shape: (D, H, W) full shape.
            coarse_shape: (Dc, Hc, Wc) coarse shape.

        Returns:
            Tuple of (spacing_full_dhw_mm, spacing_coarse_dhw_mm).
        """
        # Spacing from affine (norm of each column vector)
        spacing_xyz = torch.norm(affine_full[:3, :3], dim=0)

        # Convert xyz to dhw (zyx)
        spacing_full_dhw = torch.stack([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])

        # Coarse spacing
        D, H, W = full_shape
        Dc, Hc, Wc = coarse_shape
        scale_dhw = torch.tensor(
            [D / Dc, H / Hc, W / Wc],
            device=affine_full.device,
            dtype=torch.float32,
        )
        spacing_coarse_dhw = spacing_full_dhw * scale_dhw

        return spacing_full_dhw, spacing_coarse_dhw

    @torch.no_grad()
    def predict_proposal(
        self,
        image_full: Tensor,
        affine_full: Tensor,
        coarse_shape: tuple[int, int, int] = (128, 128, 128),
        target_channel: int = 0,
    ) -> dict[str, Tensor]:
        """Proposal-driven inference for lesions/landmarks.

        Steps:
        1. Downsample to coarse resolution
        2. Run coarse network
        3. NMS on coarse predictions to get proposals
        4. Map proposals to full resolution
        5. Sample and process fine patches at proposal locations
        6. Stitch results

        Args:
            image_full: (1, 1, D, H, W) full resolution image.
            affine_full: (4, 4) NIfTI affine.
            coarse_shape: Target shape for coarse network.
            target_channel: Channel index for NMS (0 for binary, class index for multi).

        Returns:
            Dictionary with:
                - "fine_logits": (C, D, H, W) stitched fine predictions
                - "coarse_logits": (C, Dc, Hc, Wc) coarse predictions
                - "proposals_zyx": (N, 3) proposal coordinates in full volume
                - "proposal_scores": (N,) proposal scores
        """
        full_shape = image_full.shape[2:]  # (D, H, W)
        D, H, W = full_shape

        # Move inputs to device
        image_full = image_full.to(self.device)
        affine_full = affine_full.to(self.device)

        # Compute coarse image and affine
        image_coarse = self._downsample_to_coarse(image_full, coarse_shape)
        affine_coarse = self._compute_coarse_affine(affine_full, full_shape, coarse_shape)

        spacing_full_dhw, spacing_coarse_dhw = self._compute_spacings(
            affine_full, full_shape, coarse_shape
        )

        # Run coarse network
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            coarse_logits, coarse_feat = self.model.forward_coarse_only(image_coarse)

        # Get probability map for NMS
        if coarse_logits.shape[1] > 1:
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            prob_map = coarse_probs[0, target_channel]
        else:
            prob_map = torch.sigmoid(coarse_logits)[0, 0]

        # Run NMS in mm space
        coords_coarse, scores = nms_3d_aniso_mm(
            prob_map,
            spacing_mm=tuple(spacing_coarse_dhw.tolist()),
            min_dist_mm=self.config.nms_min_dist_mm,
            threshold=self.config.nms_threshold,
            topk=self.config.nms_topk,
        )

        if coords_coarse.shape[0] == 0:
            # No proposals found - return coarse only
            return {
                "fine_logits": torch.zeros(
                    (coarse_logits.shape[1], D, H, W), device=self.device
                ),
                "coarse_logits": coarse_logits[0],
                "proposals_zyx": coords_coarse,
                "proposal_scores": scores,
            }

        # Map proposals from coarse to full indices
        n_proposals = coords_coarse.shape[0]
        proposals_full = center_coarse_to_full_index(
            coords_coarse.float(),
            affine_coarse[None].expand(n_proposals, -1, -1),
            affine_full[None].expand(n_proposals, -1, -1),
        )

        # Process fine patches in batches
        patch_logits_list = []
        patch_starts_list = []

        Df, Hf, Wf = self.config.fine_patch_shape

        for i in range(0, n_proposals, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, n_proposals)
            batch_centers = proposals_full[i:batch_end]  # (B, 3)
            B = batch_centers.shape[0]

            # Sample fine patches
            image_fine, _, valid_mask = sample_patch_from_full(
                image_full.expand(B, -1, -1, -1, -1),
                label_full=None,
                affine_full=affine_full[None].expand(B, -1, -1),
                center_full_index_zyx=batch_centers,
                out_shape=self.config.fine_patch_shape,
                spacing_fine_dhw_mm=spacing_full_dhw[None].expand(B, -1),
            )

            # Compute normalized centers in coarse space
            centers_coarse_norm = center_full_to_coarse_norm(
                batch_centers,
                affine_full[None].expand(B, -1, -1),
                affine_coarse[None].expand(B, -1, -1),
                coarse_shape,
            )

            # Run fine network with coarse context
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                fine_logits = self.model.forward_fine_with_context(
                    image_fine,
                    coarse_feat,
                    coarse_logits,
                    centers_coarse_norm,
                    self.config.fine_patch_shape,
                    spacing_full_dhw,
                    spacing_coarse_dhw,
                )

            # Compute patch start positions for stitching
            for j in range(B):
                center = batch_centers[j]
                start = torch.tensor(
                    [
                        int(center[0] - (Df - 1) / 2),
                        int(center[1] - (Hf - 1) / 2),
                        int(center[2] - (Wf - 1) / 2),
                    ],
                    device=self.device,
                )
                patch_logits_list.append(fine_logits[j].cpu())
                patch_starts_list.append(start.cpu())

        # Stitch patches
        stitched = stitch_patches_to_volume(
            patch_logits_list,
            patch_starts_list,
            full_shape,
            coarse_logits.shape[1],
            self.window.cpu(),
            device="cpu",
        )

        return {
            "fine_logits": stitched.to(self.device),
            "coarse_logits": coarse_logits[0],
            "proposals_zyx": proposals_full,
            "proposal_scores": scores,
        }

    @torch.no_grad()
    def predict_dense(
        self,
        image_full: Tensor,
        affine_full: Tensor,
        coarse_shape: tuple[int, int, int] = (128, 128, 128),
    ) -> dict[str, Tensor]:
        """Dense tiling inference for organs.

        Steps:
        1. Downsample to coarse resolution
        2. Run coarse network
        3. Generate overlapping tile positions
        4. Process fine patches at each tile
        5. Stitch results with weighted overlap-add

        Args:
            image_full: (1, 1, D, H, W) full resolution image.
            affine_full: (4, 4) NIfTI affine.
            coarse_shape: Target shape for coarse network.

        Returns:
            Dictionary with:
                - "fine_logits": (C, D, H, W) stitched fine predictions
                - "coarse_logits": (C, Dc, Hc, Wc) coarse predictions
        """
        full_shape = image_full.shape[2:]
        D, H, W = full_shape

        # Move inputs to device
        image_full = image_full.to(self.device)
        affine_full = affine_full.to(self.device)

        # Compute coarse image and affine
        image_coarse = self._downsample_to_coarse(image_full, coarse_shape)
        affine_coarse = self._compute_coarse_affine(affine_full, full_shape, coarse_shape)

        spacing_full_dhw, spacing_coarse_dhw = self._compute_spacings(
            affine_full, full_shape, coarse_shape
        )

        # Run coarse network
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            coarse_logits, coarse_feat = self.model.forward_coarse_only(image_coarse)

        # Generate tile positions
        tile_positions = generate_tile_positions(
            full_shape,
            self.config.fine_patch_shape,
            overlap_fraction=self.config.overlap_fraction,
        )

        # Process tiles in batches
        patch_logits_list = []
        patch_starts_list = []

        Df, Hf, Wf = self.config.fine_patch_shape

        for i in range(0, len(tile_positions), self.config.batch_size):
            batch_end = min(i + self.config.batch_size, len(tile_positions))
            batch_starts = tile_positions[i:batch_end]
            B = len(batch_starts)

            # Convert start positions to centers
            batch_centers = torch.tensor(
                [
                    [
                        start[0] + (Df - 1) / 2,
                        start[1] + (Hf - 1) / 2,
                        start[2] + (Wf - 1) / 2,
                    ]
                    for start in batch_starts
                ],
                device=self.device,
                dtype=torch.float32,
            )

            # Sample fine patches
            image_fine, _, valid_mask = sample_patch_from_full(
                image_full.expand(B, -1, -1, -1, -1),
                label_full=None,
                affine_full=affine_full[None].expand(B, -1, -1),
                center_full_index_zyx=batch_centers,
                out_shape=self.config.fine_patch_shape,
                spacing_fine_dhw_mm=spacing_full_dhw[None].expand(B, -1),
            )

            # Compute normalized centers in coarse space
            centers_coarse_norm = center_full_to_coarse_norm(
                batch_centers,
                affine_full[None].expand(B, -1, -1),
                affine_coarse[None].expand(B, -1, -1),
                coarse_shape,
            )

            # Run fine network with coarse context
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                fine_logits = self.model.forward_fine_with_context(
                    image_fine,
                    coarse_feat,
                    coarse_logits,
                    centers_coarse_norm,
                    self.config.fine_patch_shape,
                    spacing_full_dhw,
                    spacing_coarse_dhw,
                )

            # Store results
            for j in range(B):
                patch_logits_list.append(fine_logits[j].cpu())
                patch_starts_list.append(
                    torch.tensor(batch_starts[j], device="cpu")
                )

        # Stitch patches
        stitched = stitch_patches_to_volume(
            patch_logits_list,
            patch_starts_list,
            full_shape,
            coarse_logits.shape[1],
            self.window.cpu(),
            device="cpu",
        )

        return {
            "fine_logits": stitched.to(self.device),
            "coarse_logits": coarse_logits[0],
        }

    @torch.no_grad()
    def predict(
        self,
        image_full: Tensor,
        affine_full: Tensor,
        coarse_shape: tuple[int, int, int] = (128, 128, 128),
        **kwargs,
    ) -> dict[str, Tensor]:
        """Run inference using configured mode.

        Args:
            image_full: (1, 1, D, H, W) full resolution image.
            affine_full: (4, 4) NIfTI affine.
            coarse_shape: Target shape for coarse network.
            **kwargs: Additional arguments passed to mode-specific method.

        Returns:
            Dictionary with inference results (varies by mode).
        """
        if self.config.mode == "proposal":
            return self.predict_proposal(image_full, affine_full, coarse_shape, **kwargs)
        else:
            return self.predict_dense(image_full, affine_full, coarse_shape, **kwargs)


class BoundaryRefinementPredictor(Predictor):
    """Specialized predictor for boundary-focused organ refinement.

    This predictor runs coarse inference first, then identifies uncertain
    boundary regions and refines only those areas with fine patches.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: torch.device | str = "cuda",
        uncertainty_threshold: float = 0.3,
        dilation_vox: int = 5,
    ):
        """Initialize boundary refinement predictor.

        Args:
            model: Trained Swin3D-DNP model.
            config: Inference configuration.
            device: Device for inference.
            uncertainty_threshold: Threshold for identifying uncertain regions.
            dilation_vox: Dilation in voxels for boundary band.
        """
        super().__init__(model, config, device)
        self.uncertainty_threshold = uncertainty_threshold
        self.dilation_vox = dilation_vox

    def _find_boundary_regions(
        self,
        coarse_probs: Tensor,
        spacing_coarse_dhw: Tensor,
    ) -> list[tuple[int, int, int]]:
        """Find patch centers at uncertain boundary regions.

        Args:
            coarse_probs: (C, Dc, Hc, Wc) coarse probability map.
            spacing_coarse_dhw: (3,) coarse spacing in mm.

        Returns:
            List of (z, y, x) center positions in coarse volume.
        """
        # Compute uncertainty as max probability across classes
        max_prob = coarse_probs.max(dim=0)[0]  # (Dc, Hc, Wc)
        uncertain_mask = max_prob < (1.0 - self.uncertainty_threshold)

        # Get hard predictions
        hard_pred = coarse_probs.argmax(dim=0)  # (Dc, Hc, Wc)

        # Find boundaries via morphological operations
        boundary_mask = torch.zeros_like(uncertain_mask)

        for c in range(1, coarse_probs.shape[0]):  # Skip background
            class_mask = (hard_pred == c).float()
            if class_mask.sum() == 0:
                continue

            # Simple boundary detection via erosion
            class_mask_5d = class_mask[None, None]
            kernel = torch.ones((1, 1, 3, 3, 3), device=class_mask.device)
            eroded = (
                torch.nn.functional.conv3d(class_mask_5d, kernel, padding=1)
                >= kernel.sum()
            ).float()
            boundary = class_mask_5d - eroded
            boundary_mask = boundary_mask | (boundary[0, 0] > 0)

        # Combine uncertainty and boundary
        refine_mask = uncertain_mask & boundary_mask

        # Convert to center positions
        refine_coords = refine_mask.nonzero(as_tuple=False)

        if refine_coords.shape[0] == 0:
            return []

        # Subsample if too many
        max_centers = self.config.nms_topk * 2
        if refine_coords.shape[0] > max_centers:
            indices = torch.randperm(refine_coords.shape[0])[:max_centers]
            refine_coords = refine_coords[indices]

        return [(int(c[0]), int(c[1]), int(c[2])) for c in refine_coords]

    @torch.no_grad()
    def predict_with_boundary_refinement(
        self,
        image_full: Tensor,
        affine_full: Tensor,
        coarse_shape: tuple[int, int, int] = (128, 128, 128),
    ) -> dict[str, Tensor]:
        """Run inference with boundary-focused refinement.

        Args:
            image_full: (1, 1, D, H, W) full resolution image.
            affine_full: (4, 4) NIfTI affine.
            coarse_shape: Target shape for coarse network.

        Returns:
            Dictionary with inference results.
        """
        full_shape = image_full.shape[2:]
        D, H, W = full_shape

        # Move inputs to device
        image_full = image_full.to(self.device)
        affine_full = affine_full.to(self.device)

        # Compute coarse image and affine
        image_coarse = self._downsample_to_coarse(image_full, coarse_shape)
        affine_coarse = self._compute_coarse_affine(affine_full, full_shape, coarse_shape)

        spacing_full_dhw, spacing_coarse_dhw = self._compute_spacings(
            affine_full, full_shape, coarse_shape
        )

        # Run coarse network
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            coarse_logits, coarse_feat = self.model.forward_coarse_only(image_coarse)

        # Get coarse probabilities
        if coarse_logits.shape[1] > 1:
            coarse_probs = torch.softmax(coarse_logits[0], dim=0)
        else:
            p = torch.sigmoid(coarse_logits[0, 0])
            coarse_probs = torch.stack([1 - p, p], dim=0)

        # Find boundary regions to refine
        boundary_centers_coarse = self._find_boundary_regions(
            coarse_probs, spacing_coarse_dhw
        )

        if len(boundary_centers_coarse) == 0:
            # No refinement needed - upsample coarse
            upsampled = torch.nn.functional.interpolate(
                coarse_logits,
                size=full_shape,
                mode="trilinear",
                align_corners=False,
            )
            return {
                "fine_logits": upsampled[0],
                "coarse_logits": coarse_logits[0],
                "n_refined_patches": 0,
            }

        # Map coarse centers to full resolution
        centers_coarse = torch.tensor(
            boundary_centers_coarse, device=self.device, dtype=torch.float32
        )
        n_centers = centers_coarse.shape[0]

        centers_full = center_coarse_to_full_index(
            centers_coarse,
            affine_coarse[None].expand(n_centers, -1, -1),
            affine_full[None].expand(n_centers, -1, -1),
        )

        # Process fine patches
        patch_logits_list = []
        patch_starts_list = []

        Df, Hf, Wf = self.config.fine_patch_shape

        for i in range(0, n_centers, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, n_centers)
            batch_centers = centers_full[i:batch_end]
            B = batch_centers.shape[0]

            # Sample fine patches
            image_fine, _, valid_mask = sample_patch_from_full(
                image_full.expand(B, -1, -1, -1, -1),
                label_full=None,
                affine_full=affine_full[None].expand(B, -1, -1),
                center_full_index_zyx=batch_centers,
                out_shape=self.config.fine_patch_shape,
                spacing_fine_dhw_mm=spacing_full_dhw[None].expand(B, -1),
            )

            # Compute normalized centers
            centers_coarse_norm = center_full_to_coarse_norm(
                batch_centers,
                affine_full[None].expand(B, -1, -1),
                affine_coarse[None].expand(B, -1, -1),
                coarse_shape,
            )

            # Run fine network
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                fine_logits = self.model.forward_fine_with_context(
                    image_fine,
                    coarse_feat,
                    coarse_logits,
                    centers_coarse_norm,
                    self.config.fine_patch_shape,
                    spacing_full_dhw,
                    spacing_coarse_dhw,
                )

            for j in range(B):
                center = batch_centers[j]
                start = torch.tensor(
                    [
                        int(center[0] - (Df - 1) / 2),
                        int(center[1] - (Hf - 1) / 2),
                        int(center[2] - (Wf - 1) / 2),
                    ],
                    device="cpu",
                )
                patch_logits_list.append(fine_logits[j].cpu())
                patch_starts_list.append(start)

        # Stitch refined patches
        stitched = stitch_patches_to_volume(
            patch_logits_list,
            patch_starts_list,
            full_shape,
            coarse_logits.shape[1],
            self.window.cpu(),
            device="cpu",
        )

        # Blend with upsampled coarse predictions where not covered
        coarse_upsampled = torch.nn.functional.interpolate(
            coarse_logits.cpu(),
            size=full_shape,
            mode="trilinear",
            align_corners=False,
        )[0]

        # Create coverage mask from patch positions
        coverage = torch.zeros(full_shape, device="cpu", dtype=torch.float32)
        for start in patch_starts_list:
            sz, sy, sx = int(start[0]), int(start[1]), int(start[2])
            ez = min(sz + Df, D)
            ey = min(sy + Hf, H)
            ex = min(sx + Wf, W)
            sz = max(0, sz)
            sy = max(0, sy)
            sx = max(0, sx)
            coverage[sz:ez, sy:ey, sx:ex] = 1.0

        # Blend: use fine where covered, coarse elsewhere
        final = stitched * coverage[None] + coarse_upsampled * (1 - coverage[None])

        return {
            "fine_logits": final.to(self.device),
            "coarse_logits": coarse_logits[0],
            "n_refined_patches": n_centers,
        }
