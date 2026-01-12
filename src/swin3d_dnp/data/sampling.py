"""Patch sampling strategies for training.

This module implements various patch center sampling strategies:
- Boundary band sampling for organ refinement
- Positive center sampling from ground truth
- Hard negative sampling from false positives
- Unified PatchSampler class combining all strategies
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from swin3d_dnp.constants import (
    RATIO_BOUNDARY,
    RATIO_HARDNEG,
    RATIO_POSITIVE,
    RATIO_UNIFORM,
)


def sample_boundary_band_center(
    label_full: Tensor,
    organ_class: int,
    band_width_vox: int = 5,
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> Tensor | None:
    """Sample a center from the boundary band of an organ mask.

    The boundary band is defined as the region between the dilated and
    eroded versions of the organ mask. This focuses training on the
    most challenging boundary regions.

    Args:
        label_full: (D, H, W) label tensor (long).
        organ_class: Class index of the organ to sample from.
        band_width_vox: Width of the boundary band in voxels.
        patch_size: (Df, Hf, Wf) patch size for valid region calculation.

    Returns:
        (3,) float tensor with center coordinates (z, y, x), or None if
        no valid boundary candidates exist.
    """
    device = label_full.device

    # Create binary mask for the target organ
    mask = (label_full == organ_class).float()
    mask_5d = mask[None, None]  # (1, 1, D, H, W)

    # 3x3x3 kernel for morphological operations
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)

    # Iteratively dilate and erode
    dilated = mask_5d
    eroded = mask_5d

    for _ in range(band_width_vox):
        # Dilation: threshold at > 0 (any neighbor is positive)
        dilated = (F.conv3d(dilated, kernel, padding=1) > 0).float()
        # Erosion: threshold at >= kernel sum (all neighbors positive)
        eroded = (F.conv3d(eroded, kernel, padding=1) >= kernel.sum()).float()

    # Boundary band is dilated - eroded
    boundary_band = (dilated - eroded)[0, 0].clamp(0, 1)

    # Create valid region mask (patch must fit within volume)
    Df, Hf, Wf = patch_size
    D, H, W = label_full.shape

    valid_region = torch.zeros_like(boundary_band)

    # Compute valid ranges (centers that keep patch fully inside volume)
    z_start = Df // 2
    z_end = max(D - Df // 2, z_start + 1)
    y_start = Hf // 2
    y_end = max(H - Hf // 2, y_start + 1)
    x_start = Wf // 2
    x_end = max(W - Wf // 2, x_start + 1)

    valid_region[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    # Find candidates in both boundary band and valid region
    candidates = (boundary_band * valid_region).nonzero(as_tuple=False)

    if candidates.shape[0] == 0:
        return None

    # Sample random candidate
    idx = torch.randint(candidates.shape[0], (1,), device=device).item()
    return candidates[idx].float()  # (z, y, x)


def sample_positive_center(
    label_full: Tensor,
    target_classes: list[int] | None = None,
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> Tensor | None:
    """Sample a center from positive (foreground) regions.

    Args:
        label_full: (D, H, W) label tensor (long).
        target_classes: List of class indices to sample from.
            If None, samples from any non-background (> 0) class.
        patch_size: (Df, Hf, Wf) patch size for valid region calculation.

    Returns:
        (3,) float tensor with center coordinates (z, y, x), or None if
        no valid positive candidates exist.
    """
    device = label_full.device

    # Create positive mask
    if target_classes is None:
        positive_mask = label_full > 0
    else:
        positive_mask = torch.zeros_like(label_full, dtype=torch.bool)
        for cls in target_classes:
            positive_mask = positive_mask | (label_full == cls)

    positive_mask = positive_mask.float()

    # Create valid region mask
    Df, Hf, Wf = patch_size
    D, H, W = label_full.shape

    valid_region = torch.zeros_like(positive_mask)

    z_start = Df // 2
    z_end = max(D - Df // 2, z_start + 1)
    y_start = Hf // 2
    y_end = max(H - Hf // 2, y_start + 1)
    x_start = Wf // 2
    x_end = max(W - Wf // 2, x_start + 1)

    valid_region[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    # Find candidates
    candidates = (positive_mask * valid_region).nonzero(as_tuple=False)

    if candidates.shape[0] == 0:
        return None

    idx = torch.randint(candidates.shape[0], (1,), device=device).item()
    return candidates[idx].float()


def sample_uniform_center(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int] = (96, 96, 96),
    device: torch.device | None = None,
) -> Tensor:
    """Sample a uniformly random valid center.

    Args:
        volume_shape: (D, H, W) shape of the volume.
        patch_size: (Df, Hf, Wf) patch size for valid range.
        device: Device for output tensor.

    Returns:
        (3,) float tensor with center coordinates (z, y, x).
    """
    D, H, W = volume_shape
    Df, Hf, Wf = patch_size

    # Compute valid ranges
    z_min = Df // 2
    z_max = max(D - Df // 2, z_min + 1)
    y_min = Hf // 2
    y_max = max(H - Hf // 2, y_min + 1)
    x_min = Wf // 2
    x_max = max(W - Wf // 2, x_min + 1)

    # Sample uniformly in valid range
    z = torch.randint(z_min, z_max, (1,), device=device).float()
    y = torch.randint(y_min, y_max, (1,), device=device).float()
    x = torch.randint(x_min, x_max, (1,), device=device).float()

    return torch.cat([z, y, x])


def sample_hard_negative_centers(
    coarse_pred: Tensor,
    coarse_gt: Tensor,
    spacing_mm: tuple[float, float, float],
    min_dist_mm: float = 10.0,
    threshold: float = 0.5,
    topk: int = 16,
) -> tuple[Tensor, Tensor]:
    """Sample centers from false positive predictions.

    Used for hard negative mining during training.

    Args:
        coarse_pred: (D, H, W) predicted probability map (after sigmoid).
        coarse_gt: (D, H, W) ground truth binary mask.
        spacing_mm: (d, h, w) voxel spacing in mm.
        min_dist_mm: Minimum distance between proposals.
        threshold: Probability threshold for false positives.
        topk: Maximum number of proposals.

    Returns:
        coords: (N, 3) tensor of false positive centers (z, y, x).
        scores: (N,) tensor of corresponding prediction scores.
    """
    from swin3d_dnp.inference.nms import nms_3d_aniso_mm

    # False positive map: high prediction AND not ground truth
    fp_map = coarse_pred * (1.0 - coarse_gt.float())

    # Use NMS to get diverse false positive proposals
    return nms_3d_aniso_mm(
        prob=fp_map,
        spacing_mm=spacing_mm,
        min_dist_mm=min_dist_mm,
        threshold=threshold,
        topk=topk,
    )


class PatchSampler:
    """Unified patch center sampler with configurable strategy ratios.

    Combines uniform, positive, boundary, and hard negative sampling
    according to specified ratios. Falls back to uniform sampling
    when other strategies fail.

    Example:
        >>> sampler = PatchSampler(patch_size=(96, 96, 96))
        >>> center = sampler.sample(label_full, mode="auto")
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        ratio_uniform: float = RATIO_UNIFORM,
        ratio_positive: float = RATIO_POSITIVE,
        ratio_boundary: float = RATIO_BOUNDARY,
        ratio_hardneg: float = RATIO_HARDNEG,
        target_classes: list[int] | None = None,
        boundary_classes: list[int] | None = None,
        band_width_vox: int = 5,
    ):
        """Initialize patch sampler.

        Args:
            patch_size: (Df, Hf, Wf) fine patch size.
            ratio_uniform: Fraction of uniform samples.
            ratio_positive: Fraction of positive samples.
            ratio_boundary: Fraction of boundary samples.
            ratio_hardneg: Fraction for hard negatives (handled externally).
            target_classes: Classes for positive sampling.
            boundary_classes: Classes for boundary sampling.
            band_width_vox: Morphological band width for boundary.
        """
        self.patch_size = patch_size
        self.ratio_uniform = ratio_uniform
        self.ratio_positive = ratio_positive
        self.ratio_boundary = ratio_boundary
        self.ratio_hardneg = ratio_hardneg
        self.target_classes = target_classes
        self.boundary_classes = boundary_classes or [1]
        self.band_width_vox = band_width_vox

        # Normalize ratios (excluding hard neg which is handled separately)
        total = ratio_uniform + ratio_positive + ratio_boundary
        if total > 0:
            self._norm_uniform = ratio_uniform / total
            self._norm_positive = ratio_positive / total
            self._norm_boundary = ratio_boundary / total
        else:
            self._norm_uniform = 1.0
            self._norm_positive = 0.0
            self._norm_boundary = 0.0

    def _select_mode(self) -> str:
        """Probabilistically select sampling mode based on ratios."""
        r = torch.rand(1).item()

        if r < self._norm_uniform:
            return "uniform"
        elif r < self._norm_uniform + self._norm_positive:
            return "positive"
        else:
            return "boundary"

    def sample_uniform(
        self,
        volume_shape: tuple[int, int, int],
        device: torch.device | None = None,
    ) -> Tensor:
        """Sample uniform center."""
        return sample_uniform_center(volume_shape, self.patch_size, device)

    def sample_positive(self, label_full: Tensor) -> Tensor | None:
        """Sample positive center."""
        return sample_positive_center(
            label_full,
            target_classes=self.target_classes,
            patch_size=self.patch_size,
        )

    def sample_boundary(self, label_full: Tensor) -> Tensor | None:
        """Sample boundary center from any boundary class."""
        for cls in self.boundary_classes:
            center = sample_boundary_band_center(
                label_full,
                organ_class=cls,
                band_width_vox=self.band_width_vox,
                patch_size=self.patch_size,
            )
            if center is not None:
                return center
        return None

    def sample(
        self,
        label_full: Tensor,
        mode: str = "auto",
        volume_shape: tuple[int, int, int] | None = None,
    ) -> tuple[Tensor, str]:
        """Sample a patch center.

        Args:
            label_full: (D, H, W) label tensor (can be None for uniform only).
            mode: Sampling mode - "auto", "uniform", "positive", "boundary".
            volume_shape: Required if label_full is None and mode is "uniform".

        Returns:
            Tuple of (center, actual_mode) where center is (3,) tensor
            and actual_mode is the mode that succeeded.
        """
        if mode == "auto":
            mode = self._select_mode()

        device = label_full.device if label_full is not None else None
        shape = (
            tuple(label_full.shape)
            if label_full is not None
            else volume_shape
        )

        if shape is None:
            raise ValueError("volume_shape required when label_full is None")

        # Try requested mode
        if mode == "uniform":
            return self.sample_uniform(shape, device), "uniform"
        elif mode == "positive" and label_full is not None:
            center = self.sample_positive(label_full)
            if center is not None:
                return center, "positive"
        elif mode == "boundary" and label_full is not None:
            center = self.sample_boundary(label_full)
            if center is not None:
                return center, "boundary"

        # Fallback to uniform
        return self.sample_uniform(shape, device), "uniform"

    def sample_batch(
        self,
        label_full: Tensor,
        n_samples: int,
        modes: list[str] | None = None,
    ) -> tuple[Tensor, list[str]]:
        """Sample multiple patch centers.

        Args:
            label_full: (D, H, W) label tensor.
            n_samples: Number of centers to sample.
            modes: Optional list of modes (length n_samples). If None, uses auto.

        Returns:
            Tuple of (centers, actual_modes) where centers is (N, 3) tensor.
        """
        if modes is None:
            modes = ["auto"] * n_samples

        centers = []
        actual_modes = []

        for mode in modes:
            center, actual = self.sample(label_full, mode=mode)
            centers.append(center)
            actual_modes.append(actual)

        return torch.stack(centers), actual_modes


def sample_mixed_centers(
    label_full: Tensor,
    n_samples: int,
    patch_size: tuple[int, int, int],
    ratio_uniform: float = RATIO_UNIFORM,
    ratio_positive: float = RATIO_POSITIVE,
    ratio_boundary: float = RATIO_BOUNDARY,
    target_classes: list[int] | None = None,
    boundary_classes: list[int] | None = None,
) -> tuple[Tensor, list[str]]:
    """Convenience function to sample mixed patch centers.

    Args:
        label_full: (D, H, W) label tensor.
        n_samples: Number of samples.
        patch_size: (Df, Hf, Wf) patch size.
        ratio_uniform: Fraction of uniform samples.
        ratio_positive: Fraction of positive samples.
        ratio_boundary: Fraction of boundary samples.
        target_classes: Classes for positive sampling.
        boundary_classes: Classes for boundary sampling.

    Returns:
        Tuple of (centers, modes) where centers is (N, 3) tensor.
    """
    sampler = PatchSampler(
        patch_size=patch_size,
        ratio_uniform=ratio_uniform,
        ratio_positive=ratio_positive,
        ratio_boundary=ratio_boundary,
        target_classes=target_classes,
        boundary_classes=boundary_classes,
    )
    return sampler.sample_batch(label_full, n_samples)
