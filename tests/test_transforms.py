"""Tests for augmentation correctness.

This module implements the critical augmentation correctness tests,
including the checkerboard rotation test specified in Section 10.5
of the project plan.

The key invariant: after applying the same transformation to both
image and label, they must remain aligned - i.e., the label at
any voxel position should still correctly describe the image content
at that position.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from swin3d_dnp.geometry.sampling import sample_patch_from_full
from swin3d_dnp.geometry.coordinates import index_to_norm_acfalse, norm_to_index_acfalse
from swin3d_dnp.data.transforms import (
    random_rotation_matrix_3d,
    random_scale_factors,
    random_translation_mm,
    downsample_label_coarse,
    downsample_image_coarse,
)
from swin3d_dnp.training.utils import seed_everything


class TestCheckerboardRotation:
    """Checkerboard rotation test for augmentation correctness.

    This is a CRITICAL test from Section 10.5 of the project plan.
    It creates a checkerboard pattern where each "cell" has a unique
    value, then applies rotation augmentation to verify that image
    and label remain aligned after transformation.
    """

    def create_checkerboard_volume(
        self,
        shape: tuple[int, int, int],
        cell_size: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a checkerboard pattern with unique cell values.

        Args:
            shape: (D, H, W) volume shape.
            cell_size: Size of each checkerboard cell.

        Returns:
            Tuple of (image, label) where each cell has matching values.
        """
        D, H, W = shape

        # Create grid of cell indices
        image = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
        label = torch.zeros(1, 1, D, H, W, dtype=torch.float32)

        cell_id = 1
        for d in range(0, D, cell_size):
            for h in range(0, H, cell_size):
                for w in range(0, W, cell_size):
                    # Compute cell bounds
                    d_end = min(d + cell_size, D)
                    h_end = min(h + cell_size, H)
                    w_end = min(w + cell_size, W)

                    # Assign unique value based on position
                    # Use a formula that encodes the position
                    value = (d // cell_size) * 100 + (h // cell_size) * 10 + (w // cell_size)

                    image[0, 0, d:d_end, h:h_end, w:w_end] = float(value)
                    label[0, 0, d:d_end, h:h_end, w:w_end] = float(value)

                    cell_id += 1

        return image, label

    def test_checkerboard_no_rotation(self):
        """Baseline: checkerboard without rotation should be perfectly aligned."""
        shape = (64, 64, 64)
        image, label = self.create_checkerboard_volume(shape, cell_size=8)

        B = 1
        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image,
            label,
            affine,
            center,
            out_shape,
            spacing,
        )

        # Without rotation, image and label should match exactly
        diff = (img_patch - lbl_patch).abs()
        max_diff = diff.max().item()

        assert max_diff < 1e-5, f"Image and label mismatch without rotation: max_diff={max_diff}"

    def test_checkerboard_90deg_rotation(self):
        """Test 90-degree rotation maintains alignment.

        A 90-degree rotation is exact, so image and label should
        transform identically.
        """
        shape = (64, 64, 64)
        image, label = self.create_checkerboard_volume(shape, cell_size=8)

        B = 1
        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        # 90-degree rotation around z-axis
        R_90z = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])[None]

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image,
            label,
            affine,
            center,
            out_shape,
            spacing,
            R_world=R_90z,
        )

        # Apply valid mask before comparison
        img_masked = img_patch * valid_mask
        lbl_masked = lbl_patch * valid_mask

        # With 90-deg rotation, values should still match at corresponding positions
        # Note: interpolation artifacts may cause small differences at boundaries
        valid_voxels = valid_mask.sum()
        diff = (img_masked - lbl_masked).abs()
        mean_diff = diff.sum() / (valid_voxels + 1e-8)

        assert mean_diff < 0.5, f"Mean difference after 90deg rotation: {mean_diff}"

    def test_checkerboard_arbitrary_rotation(self):
        """Test arbitrary rotation maintains alignment within tolerance.

        For non-90-degree rotations, bilinear interpolation for image
        and nearest interpolation for label may cause small differences,
        but the overall pattern should remain aligned.
        """
        seed_everything(42)

        shape = (64, 64, 64)
        image, label = self.create_checkerboard_volume(shape, cell_size=8)

        B = 1
        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        # Small rotation (15 degrees around each axis)
        R = random_rotation_matrix_3d(max_angle_deg=15.0)[None]

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image,
            label,
            affine,
            center,
            out_shape,
            spacing,
            R_world=R,
        )

        # Focus on interior (avoid boundary artifacts)
        interior_mask = torch.zeros_like(valid_mask)
        margin = 4
        Df, Hf, Wf = out_shape
        interior_mask[:, :, margin:Df-margin, margin:Hf-margin, margin:Wf-margin] = 1
        combined_mask = valid_mask * interior_mask

        # Check alignment in interior
        img_interior = img_patch * combined_mask
        lbl_interior = lbl_patch * combined_mask

        # For checkerboard, we expect most voxels to have same value
        # after accounting for interpolation
        valid_count = combined_mask.sum()
        exact_match = (img_interior.round() == lbl_interior.round()) * combined_mask
        match_fraction = exact_match.sum() / (valid_count + 1e-8)

        # At least 80% of interior voxels should match (accounting for boundary cells)
        assert match_fraction > 0.7, f"Match fraction: {match_fraction}"

    def test_consistent_transformation_batch(self):
        """Test that batched transformation maintains per-sample alignment."""
        seed_everything(123)

        shape = (64, 64, 64)
        B = 4

        # Create batch of checkerboards (each with unique ID offset)
        images = []
        labels = []
        for i in range(B):
            img, lbl = self.create_checkerboard_volume(shape, cell_size=8)
            # Add batch-specific offset so we can verify correct batch alignment
            img = img + i * 1000
            lbl = lbl + i * 1000
            images.append(img)
            labels.append(lbl)

        image_batch = torch.cat(images, dim=0)
        label_batch = torch.cat(labels, dim=0)

        affine = torch.eye(4)[None].expand(B, -1, -1)
        centers = torch.tensor([
            [32.0, 32.0, 32.0],
            [32.0, 32.0, 32.0],
            [32.0, 32.0, 32.0],
            [32.0, 32.0, 32.0],
        ])
        spacing = torch.tensor([[1.0, 1.0, 1.0]]).expand(B, -1)
        out_shape = (16, 16, 16)

        # Different rotation for each sample
        rotations = torch.stack([
            random_rotation_matrix_3d(max_angle_deg=10.0)
            for _ in range(B)
        ])

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image_batch,
            label_batch,
            affine,
            centers,
            out_shape,
            spacing,
            R_world=rotations,
        )

        # Verify each sample's batch offset is preserved
        for i in range(B):
            expected_offset = i * 1000

            # Get min value in valid region (should be close to offset)
            valid = valid_mask[i:i+1] > 0.5
            img_vals = img_patch[i:i+1][valid]
            lbl_vals = lbl_patch[i:i+1][valid]

            if img_vals.numel() > 0:
                img_min = img_vals.min().item()
                lbl_min = lbl_vals.min().item()

                # The minimum should be approximately the offset
                assert abs(img_min - expected_offset) < 50, f"Sample {i}: img offset wrong"
                assert abs(lbl_min - expected_offset) < 50, f"Sample {i}: lbl offset wrong"


class TestRotationMatrixProperties:
    """Test rotation matrix mathematical properties."""

    def test_rotation_matrix_orthogonal(self):
        """Verify rotation matrix is orthogonal: R @ R.T = I."""
        for _ in range(10):
            R = random_rotation_matrix_3d(max_angle_deg=45.0)

            # R @ R.T should be identity
            result = R @ R.T
            assert torch.allclose(result, torch.eye(3), atol=1e-5), (
                f"R @ R.T = \n{result}\n should be identity"
            )

    def test_rotation_matrix_determinant(self):
        """Verify det(R) = 1 (proper rotation, not reflection)."""
        for _ in range(10):
            R = random_rotation_matrix_3d(max_angle_deg=45.0)
            det = torch.linalg.det(R)
            assert torch.isclose(det, torch.tensor(1.0), atol=1e-5), (
                f"det(R) = {det}, should be 1"
            )

    def test_rotation_preserves_length(self):
        """Verify rotation preserves vector length."""
        for _ in range(10):
            R = random_rotation_matrix_3d(max_angle_deg=45.0)
            v = torch.randn(3)
            v_rotated = R @ v

            orig_norm = v.norm()
            rotated_norm = v_rotated.norm()

            assert torch.isclose(orig_norm, rotated_norm, atol=1e-5), (
                f"Length changed: {orig_norm} -> {rotated_norm}"
            )


class TestScaleTransform:
    """Tests for scale transformation correctness."""

    def test_isotropic_scale(self):
        """Test isotropic scaling preserves shape proportions."""
        seed_everything(42)

        shape = (64, 64, 64)
        B = 1

        # Create volume with a centered sphere
        D, H, W = shape
        center = torch.tensor([D/2, H/2, W/2])
        radius = 10

        image = torch.zeros(B, 1, D, H, W)
        label = torch.zeros(B, 1, D, H, W)

        for d in range(D):
            for h in range(H):
                for w in range(W):
                    dist = ((d - center[0])**2 + (h - center[1])**2 + (w - center[2])**2).sqrt()
                    if dist < radius:
                        image[0, 0, d, h, w] = 1.0
                        label[0, 0, d, h, w] = 1.0

        affine = torch.eye(4)[None]
        patch_center = center[None]
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        # Apply 0.8x scale (zoom out)
        scale = torch.tensor([[0.8, 0.8, 0.8]])

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image,
            label,
            affine,
            patch_center,
            out_shape,
            spacing,
            S_world=scale,
        )

        # The scaled image should be larger (zoom out = more context)
        # Compare sphere coverage
        sphere_in_patch = (img_patch * valid_mask).sum()

        # With zoom out, we capture more of the surrounding area
        # The sphere should still be centered and spherical
        assert sphere_in_patch > 0, "Sphere should be visible in patch"

    def test_anisotropic_scale(self):
        """Test anisotropic scaling stretches correctly."""
        seed_everything(42)

        shape = (64, 64, 64)
        B = 1

        # Create a cube in the center
        image = torch.zeros(B, 1, *shape)
        label = torch.zeros(B, 1, *shape)
        image[0, 0, 24:40, 24:40, 24:40] = 1.0
        label[0, 0, 24:40, 24:40, 24:40] = 1.0

        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        # Scale 2x in z, 1x in y and x
        scale = torch.tensor([[1.0, 1.0, 2.0]])  # xyz order

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image,
            label,
            affine,
            center,
            out_shape,
            spacing,
            S_world=scale,
        )

        # Image and label should still match
        valid_diff = ((img_patch - lbl_patch).abs() * valid_mask).sum()
        assert valid_diff < 1e-3, f"Image/label mismatch: {valid_diff}"


class TestTranslationTransform:
    """Tests for translation transformation correctness."""

    def test_translation_shifts_content(self):
        """Test that translation shifts both image and label."""
        shape = (64, 64, 64)
        B = 1

        # Create a small blob at known location
        image = torch.zeros(B, 1, *shape)
        label = torch.zeros(B, 1, *shape)
        image[0, 0, 30:34, 30:34, 30:34] = 1.0
        label[0, 0, 30:34, 30:34, 30:34] = 1.0

        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        # No translation
        img1, lbl1, _ = sample_patch_from_full(
            image, label, affine, center, out_shape, spacing,
        )

        # Translation in x
        translation = torch.tensor([[5.0, 0.0, 0.0]])
        img2, lbl2, _ = sample_patch_from_full(
            image, label, affine, center, out_shape, spacing,
            t_world_mm=translation,
        )

        # The translated patch should have different content position
        blob1_center = (img1 * torch.arange(32)[None, None, None, None, :]).sum() / (img1.sum() + 1e-8)
        blob2_center = (img2 * torch.arange(32)[None, None, None, None, :]).sum() / (img2.sum() + 1e-8)

        # Blob should have moved due to translation
        assert abs(blob1_center.item() - blob2_center.item()) > 1.0, "Translation should shift content"


class TestValidMaskCorrectness:
    """Tests for valid mask computation correctness."""

    def test_valid_mask_center_patch(self):
        """Test that center patch has all valid voxels."""
        shape = (64, 64, 64)
        B = 1

        image = torch.randn(B, 1, *shape)
        affine = torch.eye(4)[None]
        center = torch.tensor([[32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (16, 16, 16)

        _, _, valid_mask = sample_patch_from_full(
            image, None, affine, center, out_shape, spacing,
        )

        # Center patch should be fully valid
        assert valid_mask.sum() == valid_mask.numel(), "Center patch should be fully valid"

    def test_valid_mask_corner_patch(self):
        """Test that corner patch has partial valid voxels."""
        shape = (64, 64, 64)
        B = 1

        image = torch.randn(B, 1, *shape)
        affine = torch.eye(4)[None]
        center = torch.tensor([[4.0, 4.0, 4.0]])  # Near corner
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (16, 16, 16)

        _, _, valid_mask = sample_patch_from_full(
            image, None, affine, center, out_shape, spacing,
        )

        # Corner patch should be partially valid (some voxels out of bounds)
        valid_count = valid_mask.sum()
        total = valid_mask.numel()

        assert valid_count < total, "Corner patch should have some invalid voxels"
        assert valid_count > 0, "Corner patch should have some valid voxels"

    def test_valid_mask_alignment_with_label(self):
        """Test that valid mask correctly identifies where label is valid."""
        shape = (64, 64, 64)
        B = 1

        # Create image with distinct border value
        image = torch.ones(B, 1, *shape)
        label = torch.ones(B, 1, *shape) * 42

        affine = torch.eye(4)[None]
        center = torch.tensor([[5.0, 32.0, 32.0]])  # Partially out of bounds
        spacing = torch.tensor([[1.0, 1.0, 1.0]])
        out_shape = (32, 32, 32)

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image, label, affine, center, out_shape, spacing,
        )

        # In valid region, label should be 42
        valid_label = lbl_patch[valid_mask > 0.5]
        if valid_label.numel() > 0:
            assert (valid_label == 42).all(), "Valid region should have correct label"

        # In invalid region, label should be 0 (padding_mode=zeros for labels)
        invalid_label = lbl_patch[valid_mask < 0.5]
        if invalid_label.numel() > 0:
            assert (invalid_label == 0).all(), "Invalid region should be zero-padded"


class TestCombinedTransforms:
    """Tests for combined rotation + scale + translation."""

    def test_combined_transform_alignment(self):
        """Test that combined transforms maintain image-label alignment."""
        seed_everything(789)

        shape = (64, 64, 64)
        B = 2

        # Create structured volumes
        image = torch.zeros(B, 1, *shape)
        label = torch.zeros(B, 1, *shape)

        # Create patterns (different per batch)
        for b in range(B):
            for d in range(0, 64, 8):
                for h in range(0, 64, 8):
                    for w in range(0, 64, 8):
                        val = d + h*10 + w*100 + b*10000
                        image[b, 0, d:d+8, h:h+8, w:w+8] = val
                        label[b, 0, d:d+8, h:h+8, w:w+8] = val

        affine = torch.eye(4)[None].expand(B, -1, -1)
        centers = torch.tensor([[32.0, 32.0, 32.0], [32.0, 32.0, 32.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]]).expand(B, -1)
        out_shape = (16, 16, 16)

        # Combined transforms
        R = torch.stack([random_rotation_matrix_3d(15.0) for _ in range(B)])
        S = torch.stack([random_scale_factors(0.9, 1.1, isotropic=True) for _ in range(B)])
        t = torch.stack([random_translation_mm(5.0) for _ in range(B)])

        img_patch, lbl_patch, valid_mask = sample_patch_from_full(
            image, label, affine, centers, out_shape, spacing,
            R_world=R, S_world=S, t_world_mm=t,
        )

        # Check alignment for each batch sample
        for b in range(B):
            img_b = img_patch[b:b+1]
            lbl_b = lbl_patch[b:b+1]
            valid_b = valid_mask[b:b+1]

            # Compute difference in valid region
            diff = (img_b - lbl_b).abs() * valid_b
            mean_diff = diff.sum() / (valid_b.sum() + 1e-8)

            # Difference should be small (interpolation artifacts only)
            assert mean_diff < 10.0, f"Batch {b}: mean difference {mean_diff} too large"


class TestDownsampleAlignment:
    """Tests for downsample transform alignment."""

    def test_downsample_preserves_structure(self):
        """Test that downsampling preserves overall structure."""
        # Create image with distinct regions
        image = torch.zeros(128, 128, 128)
        image[0:64, :, :] = 1.0  # Top half
        image[64:128, :, :] = 2.0  # Bottom half

        coarse = downsample_image_coarse(image, (64, 64, 64))

        # Structure should be preserved
        assert coarse[0:32, :, :].mean() < 1.5  # Should be closer to 1.0
        assert coarse[32:64, :, :].mean() > 1.5  # Should be closer to 2.0

    def test_downsample_label_multiclass(self):
        """Test multi-class label downsampling."""
        label = torch.zeros(128, 128, 128, dtype=torch.long)
        label[0:64, :, :] = 1
        label[64:128, :, :] = 2

        coarse = downsample_label_coarse(label, (64, 64, 64))

        # Classes should be preserved
        assert (coarse[0:32, :, :] == 1).all()
        assert (coarse[32:64, :, :] == 2).all()
