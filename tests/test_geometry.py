"""Tests for geometry utilities.

These tests verify:
1. Coordinate mapping identity (n(u) and u(n) are inverse)
2. Round-trip sampling on synthetic volumes
3. Context sampler alignment
4. Masking correctness for out-of-bounds regions
"""

import pytest
import torch

from swin3d_dnp.geometry.coordinates import (
    index_to_norm_acfalse,
    norm_to_index_acfalse,
    patch_center_index,
    patch_center_norm,
)
from swin3d_dnp.geometry.mapping import (
    center_coarse_to_full_index,
    center_full_to_coarse_norm,
)
from swin3d_dnp.geometry.sampling import (
    DifferentiableContextSampler,
    extent_vox_in_src_from_spacings,
    sample_patch_from_full,
)


class TestCoordinates:
    """Tests for coordinate transformation functions."""

    def test_index_to_norm_identity(self):
        """Verify n(u) and u(n) are inverse functions."""
        N = 64
        u = torch.linspace(0, N - 1, 100)
        n = index_to_norm_acfalse(u, N)
        u_recovered = norm_to_index_acfalse(n, N)

        assert torch.allclose(u, u_recovered, atol=1e-6)

    def test_norm_to_index_identity(self):
        """Verify u(n) and n(u) are inverse functions."""
        N = 128
        n = torch.linspace(-1, 1, 100)
        u = norm_to_index_acfalse(n, N)
        n_recovered = index_to_norm_acfalse(u, N)

        assert torch.allclose(n, n_recovered, atol=1e-6)

    def test_boundary_values(self):
        """Test coordinate transforms at boundaries."""
        N = 10

        # First voxel center (index 0) should map to near -1
        n0 = index_to_norm_acfalse(torch.tensor(0.0), N)
        assert n0 < -0.8  # Should be -0.9 for N=10

        # Last voxel center (index N-1) should map to near 1
        n_last = index_to_norm_acfalse(torch.tensor(float(N - 1)), N)
        assert n_last > 0.8  # Should be 0.9 for N=10

        # Center of volume should map to 0
        n_center = index_to_norm_acfalse(torch.tensor((N - 1) / 2.0), N)
        assert abs(n_center) < 0.1

    def test_patch_center_index(self):
        """Test patch center calculation in index space."""
        start = torch.tensor([10, 20, 30])
        size = torch.tensor([32, 32, 32])
        center = patch_center_index(start, size)

        # Center should be start + (size - 1) / 2
        expected = torch.tensor([10 + 15.5, 20 + 15.5, 30 + 15.5])
        assert torch.allclose(center, expected)

    def test_patch_center_norm(self):
        """Test patch center calculation in normalized space."""
        start = torch.tensor([0, 0, 0])
        size = torch.tensor([64, 64, 64])
        volume_size = torch.tensor([128, 128, 128])
        center_norm = patch_center_norm(start, size, volume_size)

        # Patch at start covering half the volume should have center at -0.5
        # center = 31.5, normalized = 2*(31.5+0.5)/128 - 1 = 2*32/128 - 1 = -0.5
        assert torch.allclose(center_norm, torch.tensor([-0.5, -0.5, -0.5]))


class TestMapping:
    """Tests for coordinate mapping between full and coarse volumes."""

    def test_full_to_coarse_at_center(self, sample_affine):
        """Test mapping at volume center.

        When downsampling 128³ (1mm) to 64³ (2mm) covering the same FOV,
        the physical center should map to normalized (0, 0, 0) in coarse.
        This requires the coarse affine to have 2x spacing.
        """
        affine_full = sample_affine[None]  # (1, 4, 4), 1mm spacing

        # Coarse affine with 2x spacing to maintain same physical FOV
        # Full: 128 voxels * 1mm = 128mm FOV
        # Coarse: 64 voxels * 2mm = 128mm FOV
        affine_coarse = torch.tensor(
            [
                [2.0, 0.0, 0.0, -64.0],  # 2mm spacing in x
                [0.0, 2.0, 0.0, -64.0],  # 2mm spacing in y
                [0.0, 0.0, 2.0, -64.0],  # 2mm spacing in z
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )[None]  # (1, 4, 4)

        shape_coarse = (64, 64, 64)

        # Center of a 128x128x128 volume at index (63.5, 63.5, 63.5)
        # World: 63.5 * 1 - 64 = -0.5 mm
        center_full = torch.tensor([[63.5, 63.5, 63.5]])

        coarse_norm = center_full_to_coarse_norm(
            center_full, affine_full, affine_coarse, shape_coarse
        )

        # Center of coarse 64³ volume is at index (31.5, 31.5, 31.5)
        # World: 31.5 * 2 - 64 = 63 - 64 = -1 mm (close to -0.5)
        # The normalized value should be close to 0
        # n = 2*(31.75 + 0.5)/64 - 1 ≈ 0.0078 ≈ 0
        assert torch.allclose(coarse_norm, torch.zeros(1, 3), atol=0.1)

    def test_round_trip_mapping(self, sample_affine):
        """Test that full->coarse->full mapping is identity."""
        B = 4
        affine_full = sample_affine[None].repeat(B, 1, 1)
        affine_coarse = sample_affine[None].repeat(B, 1, 1)

        # Random centers
        center_full = torch.rand(B, 3) * 100 + 10  # Random points in volume

        # Full -> coarse norm -> coarse index -> full
        coarse_norm = center_full_to_coarse_norm(
            center_full, affine_full, affine_coarse, (128, 128, 128)
        )

        # Convert norm to index
        shape = torch.tensor([128, 128, 128], dtype=torch.float32)
        coarse_index = ((coarse_norm + 1.0) * shape) / 2.0 - 0.5

        center_recovered = center_coarse_to_full_index(
            coarse_index, affine_coarse, affine_full
        )

        assert torch.allclose(center_full, center_recovered, atol=1e-4)


class TestSampling:
    """Tests for patch and context sampling."""

    def test_extent_calculation(self):
        """Test extent calculation from spacings."""
        out_shape = (96, 96, 96)
        fine_spacing = (1.0, 1.0, 1.0)  # 1mm isotropic
        coarse_spacing = (2.0, 2.0, 2.0)  # 2mm isotropic

        extent = extent_vox_in_src_from_spacings(out_shape, fine_spacing, coarse_spacing)

        # 96mm fine FOV / 2mm coarse spacing = 48 coarse voxels
        assert extent == (48.0, 48.0, 48.0)

    def test_extent_anisotropic(self):
        """Test extent with anisotropic spacing."""
        out_shape = (64, 128, 128)
        fine_spacing = (2.0, 1.0, 1.0)  # 2mm z, 1mm x/y
        coarse_spacing = (4.0, 2.0, 2.0)

        extent = extent_vox_in_src_from_spacings(out_shape, fine_spacing, coarse_spacing)

        # D: 64*2/4 = 32, H: 128*1/2 = 64, W: 128*1/2 = 64
        assert extent == (32.0, 64.0, 64.0)

    def test_sample_patch_center_aligned(self, sample_affine, seed):
        """Test that sampling at center returns center of volume."""
        B = 1
        D, H, W = 64, 64, 64
        image_full = torch.zeros(B, 1, D, H, W)
        # Put a marker at center
        image_full[0, 0, D // 2, H // 2, W // 2] = 1.0

        affine = sample_affine[None]
        center = torch.tensor([[D / 2 - 0.5, H / 2 - 0.5, W / 2 - 0.5]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])

        img, lbl, mask = sample_patch_from_full(
            image_full,
            None,
            affine,
            center,
            out_shape=(32, 32, 32),
            spacing_fine_dhw_mm=spacing,
        )

        # Patch center should contain the marker (approximately)
        assert img.shape == (1, 1, 32, 32, 32)
        center_val = img[0, 0, 16, 16, 16]
        assert center_val > 0.5  # Should capture marker

    def test_valid_mask_full_inbounds(self, sample_affine):
        """Test that centered patch has all-ones valid mask."""
        B = 1
        D, H, W = 128, 128, 128
        image_full = torch.randn(B, 1, D, H, W)
        affine = sample_affine[None]
        center = torch.tensor([[D / 2 - 0.5, H / 2 - 0.5, W / 2 - 0.5]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])

        _, _, mask = sample_patch_from_full(
            image_full,
            None,
            affine,
            center,
            out_shape=(32, 32, 32),
            spacing_fine_dhw_mm=spacing,
        )

        # All voxels should be valid for centered patch
        assert mask.sum() == mask.numel()

    def test_valid_mask_partial_oob(self, sample_affine):
        """Test that corner patch has partial valid mask."""
        B = 1
        D, H, W = 64, 64, 64
        image_full = torch.randn(B, 1, D, H, W)
        affine = sample_affine[None]
        # Place center near corner
        center = torch.tensor([[5.0, 5.0, 5.0]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])

        _, _, mask = sample_patch_from_full(
            image_full,
            None,
            affine,
            center,
            out_shape=(32, 32, 32),
            spacing_fine_dhw_mm=spacing,
        )

        # Some voxels should be invalid (out of bounds)
        assert mask.sum() < mask.numel()
        # But some should be valid
        assert mask.sum() > 0


class TestContextSampler:
    """Tests for DifferentiableContextSampler."""

    def test_context_sampler_shape(self):
        """Test that context sampler outputs correct shape."""
        sampler = DifferentiableContextSampler()
        B, C = 2, 16
        src = torch.randn(B, C, 32, 32, 32)
        centers = torch.zeros(B, 3)  # Center of volume
        out_shape = (16, 16, 16)
        extent = (16.0, 16.0, 16.0)

        output = sampler(src, centers, out_shape, extent)

        assert output.shape == (B, C, 16, 16, 16)

    def test_context_sampler_center_alignment(self):
        """Test that sampling at center returns center features."""
        sampler = DifferentiableContextSampler()
        B = 1
        src = torch.zeros(B, 1, 32, 32, 32)
        src[0, 0, 16, 16, 16] = 1.0  # Marker at center

        centers = torch.zeros(B, 3)  # Center in normalized coords
        out_shape = (8, 8, 8)
        extent = (8.0, 8.0, 8.0)

        output = sampler(src, centers, out_shape, extent)

        # Center of output should have high value (marker)
        center_val = output[0, 0, 4, 4, 4]
        assert center_val > 0.5

    def test_context_sampler_gradient_flow(self):
        """Test that gradients flow through context sampler."""
        sampler = DifferentiableContextSampler()
        B = 2
        src = torch.randn(B, 8, 16, 16, 16, requires_grad=True)
        centers = torch.zeros(B, 3)

        output = sampler(src, centers, (8, 8, 8), (8.0, 8.0, 8.0))
        loss = output.mean()
        loss.backward()

        assert src.grad is not None
        assert src.grad.abs().sum() > 0

    def test_ramp_volume_alignment(self):
        """Test context sampler with ramp volumes encoding coordinates."""
        sampler = DifferentiableContextSampler()
        B = 1
        D, H, W = 32, 32, 32

        # Create ramp volumes encoding z, y, x coordinates
        z_ramp = torch.linspace(0, 1, D).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        y_ramp = torch.linspace(0, 1, H).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        x_ramp = torch.linspace(0, 1, W).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
        src = torch.cat([z_ramp, y_ramp, x_ramp], dim=1)  # (B, 3, D, H, W)

        # Sample at center with full extent
        centers = torch.zeros(B, 3)
        out_shape = (16, 16, 16)
        extent = (16.0, 16.0, 16.0)

        output = sampler(src, centers, out_shape, extent)

        # Output should also be ramps centered around 0.5
        z_out = output[0, 0, :, 8, 8]  # z ramp at center y, x
        assert z_out[0] < z_out[-1]  # Should be increasing
        assert abs(z_out[8] - 0.5) < 0.1  # Center should be near 0.5


class TestRoundTripSampling:
    """Test round-trip sampling accuracy."""

    def test_identity_sampling(self, sample_affine):
        """Test that sampling without transform returns original values."""
        B = 1
        D, H, W = 32, 32, 32

        # Create a structured test volume
        image_full = torch.zeros(B, 1, D, H, W)
        for i in range(D):
            image_full[0, 0, i, :, :] = float(i)  # z-ramp

        affine = sample_affine[None]
        center = torch.tensor([[D / 2 - 0.5, H / 2 - 0.5, W / 2 - 0.5]])
        spacing = torch.tensor([[1.0, 1.0, 1.0]])

        img, _, _ = sample_patch_from_full(
            image_full,
            None,
            affine,
            center,
            out_shape=(16, 16, 16),
            spacing_fine_dhw_mm=spacing,
        )

        # Check that z-ramp is preserved
        z_slice = img[0, 0, :, 8, 8]
        assert z_slice[0] < z_slice[-1]  # Should be increasing
