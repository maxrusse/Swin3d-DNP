"""Dataset classes for Swin3D-DNP.

This module provides dataset implementations that follow the data contracts
defined in the project specification:

Data contract per case:
- image_full: (1, D, H, W) float32
- label_full: (D, H, W) long
- affine_full: (4, 4) float64/float32 (NIfTI standard)
- spacing_full_dhw_mm: (3,) float32 in (d, h, w) order
- case_id: str
- modality: "ct" or "mr" (optional)

Derived coarse view:
- image_coarse: (1, Dc, Hc, Wc) float32
- label_coarse: (Dc, Hc, Wc) long (optional)
- affine_coarse: (4, 4)
- spacing_coarse_dhw_mm: (3,) float32
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from swin3d_dnp.data.transforms import downsample_image_coarse, downsample_label_coarse
from swin3d_dnp.constants import DEFAULT_COARSE_SHAPE


class Swin3DDNPDataset(Dataset):
    """Base dataset for Swin3D-DNP training.

    Loads full-resolution volumes and derives coarse views. Supports
    both NIfTI and numpy formats.
    """

    def __init__(
        self,
        case_list: list[dict[str, Any]],
        coarse_shape: tuple[int, int, int] = DEFAULT_COARSE_SHAPE,
        transform: Callable | None = None,
        is_binary_lesion: bool = False,
        cache_coarse: bool = True,
    ):
        """Initialize dataset.

        Args:
            case_list: List of case dictionaries. Each must contain:
                - "image_path": Path to image file (.nii.gz or .npy)
                - "label_path": Path to label file
                - "case_id": Unique case identifier
                Optional:
                - "affine": Pre-loaded affine (4, 4)
                - "spacing_dhw_mm": Pre-computed spacing
                - "modality": "ct" or "mr"
            coarse_shape: (Dc, Hc, Wc) target coarse resolution.
            transform: Optional transform applied to each sample.
            is_binary_lesion: If True, use maxpool for label downsampling.
            cache_coarse: Whether to cache coarse volumes in memory.
        """
        self.case_list = case_list
        self.coarse_shape = coarse_shape
        self.transform = transform
        self.is_binary_lesion = is_binary_lesion
        self.cache_coarse = cache_coarse

        # Cache for coarse volumes
        self._coarse_cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.case_list)

    def _load_nifti(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load NIfTI file and return data and affine."""
        import nibabel as nib

        img = nib.load(str(path))
        data = img.get_fdata()
        affine = img.affine.astype(np.float32)
        return data, affine

    def _load_numpy(self, path: str | Path) -> np.ndarray:
        """Load numpy file."""
        return np.load(str(path))

    def _compute_spacing_from_affine(self, affine: np.ndarray) -> np.ndarray:
        """Extract voxel spacing from NIfTI affine.

        NIfTI affine maps (i, j, k) to (x, y, z) world coordinates.
        Spacing is the magnitude of each column of the rotation matrix.

        Our internal ordering is (d, h, w) = (z, y, x), so we reorder.

        Args:
            affine: (4, 4) NIfTI affine matrix.

        Returns:
            (3,) spacing in (d, h, w) = (z, y, x) order.
        """
        # Spacing is magnitude of each column of rotation matrix
        # Columns correspond to (x, y, z) axes
        spacing_xyz = np.linalg.norm(affine[:3, :3], axis=0)
        # Reorder to (z, y, x) = (d, h, w)
        return np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)

    def _compute_coarse_affine(
        self,
        affine_full: np.ndarray,
        full_shape: tuple[int, int, int],
        coarse_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Compute coarse volume affine from full affine.

        The coarse volume covers the same physical extent but with
        different spacing. We scale the affine accordingly.

        Args:
            affine_full: (4, 4) full resolution affine.
            full_shape: (D, H, W) full resolution shape.
            coarse_shape: (Dc, Hc, Wc) coarse resolution shape.

        Returns:
            (4, 4) coarse resolution affine.
        """
        # Shape ratios
        D, H, W = full_shape
        Dc, Hc, Wc = coarse_shape

        # Scale factors (xyz order for affine)
        scale_x = W / Wc
        scale_y = H / Hc
        scale_z = D / Dc

        # Scale the rotation/spacing part of affine
        affine_coarse = affine_full.copy()
        affine_coarse[:3, 0] *= scale_x
        affine_coarse[:3, 1] *= scale_y
        affine_coarse[:3, 2] *= scale_z

        # Adjust origin to keep same physical center
        # The coarse voxel (0,0,0) should map to same world position
        # as the full voxel that corresponds to it
        # No adjustment needed if we scale uniformly from origin

        return affine_coarse

    def _load_case(self, case_info: dict) -> dict[str, Any]:
        """Load a single case from disk.

        Args:
            case_info: Case dictionary with paths and metadata.

        Returns:
            Dictionary with loaded data.
        """
        image_path = case_info["image_path"]
        label_path = case_info["label_path"]

        # Determine file format and load
        if str(image_path).endswith((".nii", ".nii.gz")):
            image_data, affine = self._load_nifti(image_path)
            label_data, _ = self._load_nifti(label_path)
        else:
            image_data = self._load_numpy(image_path)
            label_data = self._load_numpy(label_path)
            affine = case_info.get("affine", np.eye(4, dtype=np.float32))

        # Ensure correct dtypes
        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.int64)

        # Compute or use provided spacing
        if "spacing_dhw_mm" in case_info:
            spacing = np.array(case_info["spacing_dhw_mm"], dtype=np.float32)
        else:
            spacing = self._compute_spacing_from_affine(affine)

        return {
            "image_full": image_data,
            "label_full": label_data,
            "affine_full": affine,
            "spacing_full_dhw_mm": spacing,
            "case_id": case_info["case_id"],
            "modality": case_info.get("modality", "unknown"),
        }

    def _derive_coarse(self, sample: dict) -> dict:
        """Derive coarse resolution views from full resolution.

        Args:
            sample: Dictionary with full resolution data.

        Returns:
            Sample dictionary with added coarse data.
        """
        image_full = torch.from_numpy(sample["image_full"])
        label_full = torch.from_numpy(sample["label_full"])

        # Add channel dimension if needed
        if image_full.dim() == 3:
            image_full = image_full.unsqueeze(0)

        # Downsample
        image_coarse = downsample_image_coarse(image_full, self.coarse_shape)
        label_coarse = downsample_label_coarse(
            label_full, self.coarse_shape, is_binary_lesion=self.is_binary_lesion
        )

        # Compute coarse affine and spacing
        full_shape = tuple(label_full.shape)
        affine_coarse = self._compute_coarse_affine(
            sample["affine_full"], full_shape, self.coarse_shape
        )
        spacing_coarse = self._compute_spacing_from_affine(affine_coarse)

        sample["image_coarse"] = image_coarse
        sample["label_coarse"] = label_coarse
        sample["affine_coarse"] = affine_coarse
        sample["spacing_coarse_dhw_mm"] = spacing_coarse

        # Convert full to tensors too
        sample["image_full"] = image_full
        sample["label_full"] = label_full

        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
            - image_full: (1, D, H, W) full resolution image
            - label_full: (D, H, W) full resolution labels
            - affine_full: (4, 4) NIfTI affine
            - spacing_full_dhw_mm: (3,) voxel spacing
            - image_coarse: (1, Dc, Hc, Wc) coarse image
            - label_coarse: (Dc, Hc, Wc) coarse labels
            - affine_coarse: (4, 4) coarse affine
            - spacing_coarse_dhw_mm: (3,) coarse spacing
            - case_id: str
            - modality: str
        """
        case_info = self.case_list[idx]
        case_id = case_info["case_id"]

        # Check cache
        if self.cache_coarse and case_id in self._coarse_cache:
            sample = self._coarse_cache[case_id].copy()
        else:
            # Load and process
            sample = self._load_case(case_info)
            sample = self._derive_coarse(sample)

            if self.cache_coarse:
                self._coarse_cache[case_id] = sample.copy()

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TrainingPatchDataset(Dataset):
    """Dataset that provides training patches from full volumes.

    Wraps a base dataset and samples patches according to the
    training sampling strategy (uniform, positive, boundary, hard negative).
    """

    def __init__(
        self,
        base_dataset: Swin3DDNPDataset,
        fine_shape: tuple[int, int, int],
        patches_per_volume: int = 4,
        ratio_uniform: float = 0.30,
        ratio_positive: float = 0.30,
        ratio_boundary: float = 0.20,
        ratio_hardneg: float = 0.20,
        target_classes: list[int] | None = None,
        boundary_classes: list[int] | None = None,
        transform: Callable | None = None,
    ):
        """Initialize patch dataset.

        Args:
            base_dataset: Base dataset providing full volumes.
            fine_shape: (Df, Hf, Wf) fine patch shape.
            patches_per_volume: Number of patches per volume per epoch.
            ratio_uniform: Fraction of uniform random patches.
            ratio_positive: Fraction of positive-centered patches.
            ratio_boundary: Fraction of boundary band patches.
            ratio_hardneg: Fraction for hard negatives (placeholder).
            target_classes: Classes for positive sampling.
            boundary_classes: Classes for boundary band sampling.
            transform: Optional transform for patches.
        """
        self.base_dataset = base_dataset
        self.fine_shape = fine_shape
        self.patches_per_volume = patches_per_volume

        # Sampling ratios
        self.ratio_uniform = ratio_uniform
        self.ratio_positive = ratio_positive
        self.ratio_boundary = ratio_boundary
        self.ratio_hardneg = ratio_hardneg

        self.target_classes = target_classes
        self.boundary_classes = boundary_classes or [1]  # Default to class 1
        self.transform = transform

        # Compute total length
        self._total_patches = len(base_dataset) * patches_per_volume

    def __len__(self) -> int:
        return self._total_patches

    def _sample_patch_center(
        self, label_full: Tensor, sampling_mode: str
    ) -> Tensor | None:
        """Sample a patch center based on sampling mode.

        Args:
            label_full: (D, H, W) label tensor.
            sampling_mode: One of "uniform", "positive", "boundary".

        Returns:
            (3,) center tensor or None if sampling fails.
        """
        from swin3d_dnp.data.sampling import (
            sample_boundary_band_center,
            sample_positive_center,
            sample_uniform_center,
        )

        if sampling_mode == "uniform":
            return sample_uniform_center(
                tuple(label_full.shape),
                self.fine_shape,
                device=label_full.device,
            )
        elif sampling_mode == "positive":
            return sample_positive_center(
                label_full,
                target_classes=self.target_classes,
                patch_size=self.fine_shape,
            )
        elif sampling_mode == "boundary":
            # Try each boundary class
            for cls in self.boundary_classes:
                center = sample_boundary_band_center(
                    label_full,
                    organ_class=cls,
                    patch_size=self.fine_shape,
                )
                if center is not None:
                    return center
            return None
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    def _select_sampling_mode(self) -> str:
        """Select sampling mode based on ratios."""
        r = torch.rand(1).item()
        cum = 0.0

        cum += self.ratio_uniform
        if r < cum:
            return "uniform"

        cum += self.ratio_positive
        if r < cum:
            return "positive"

        cum += self.ratio_boundary
        if r < cum:
            return "boundary"

        # Hard negative handled separately during training
        return "uniform"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a patch sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with volume data and sampled patch center.
        """
        # Map to base dataset index
        volume_idx = idx // self.patches_per_volume

        # Get full volume
        sample = self.base_dataset[volume_idx]

        # Select and sample center
        sampling_mode = self._select_sampling_mode()
        center = self._sample_patch_center(sample["label_full"], sampling_mode)

        # Fallback to uniform if sampling fails
        if center is None:
            center = self._sample_patch_center(sample["label_full"], "uniform")

        sample["center_full_index_zyx"] = center
        sample["sampling_mode"] = sampling_mode

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def create_case_list_from_directory(
    image_dir: str | Path,
    label_dir: str | Path,
    pattern: str = "*.nii.gz",
    modality: str = "unknown",
) -> list[dict[str, Any]]:
    """Create case list from directory structure.

    Args:
        image_dir: Directory containing image files.
        label_dir: Directory containing label files.
        pattern: Glob pattern for finding files.
        modality: Imaging modality ("ct" or "mr").

    Returns:
        List of case dictionaries.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = sorted(image_dir.glob(pattern))
    case_list = []

    for img_path in image_files:
        # Find corresponding label
        label_path = label_dir / img_path.name

        if not label_path.exists():
            # Try alternative naming
            label_path = label_dir / img_path.name.replace("_0000", "")

        if not label_path.exists():
            continue

        case_id = img_path.stem.replace(".nii", "")

        case_list.append({
            "image_path": str(img_path),
            "label_path": str(label_path),
            "case_id": case_id,
            "modality": modality,
        })

    return case_list
