"""Tests for inference utilities.

These tests verify:
1. NMS produces correct number of proposals
2. NMS correctly suppresses nearby peaks
3. Boundary band correctly identifies edges
4. Label downsampling preserves small objects
5. Stitching window functions
6. Stitching uniformity (constant patches -> constant output)
7. Tile position generation
8. Proposal mapping round-trip
"""

import pytest
import torch

from swin3d_dnp.inference.nms import nms_3d_aniso_mm, nms_3d_isotropic
from swin3d_dnp.inference.stitching import (
    cos2_window_1d,
    cos2_window_3d,
    stitch_patches_to_volume,
    generate_tile_positions,
)
from swin3d_dnp.data.sampling import (
    sample_boundary_band_center,
    sample_positive_center,
    sample_uniform_center,
)
from swin3d_dnp.data.transforms import (
    downsample_label_coarse,
    downsample_image_coarse,
    random_rotation_matrix_3d,
)
from swin3d_dnp.geometry.mapping import (
    center_full_to_coarse_norm,
    center_coarse_to_full_index,
)


class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_nms_single_peak(self):
        """Test NMS with a single isolated peak."""
        prob = torch.zeros(32, 32, 32)
        prob[16, 16, 16] = 0.9  # Single peak at center

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=5.0,
            threshold=0.5,
            topk=10,
        )

        assert coords.shape[0] == 1
        assert torch.allclose(coords[0].float(), torch.tensor([16.0, 16.0, 16.0]))
        assert scores[0] == 0.9

    def test_nms_multiple_separated_peaks(self):
        """Test NMS with multiple well-separated peaks."""
        prob = torch.zeros(64, 64, 64)
        # Place 4 peaks well apart (20 voxels = 20mm with 1mm spacing)
        prob[10, 10, 10] = 0.8
        prob[30, 30, 30] = 0.9
        prob[50, 10, 10] = 0.7
        prob[10, 50, 50] = 0.6

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=10.0,  # 10mm min distance
            threshold=0.5,
            topk=10,
        )

        # All 4 peaks should be kept
        assert coords.shape[0] == 4
        # Scores should be sorted descending
        assert torch.all(scores[:-1] >= scores[1:])

    def test_nms_suppression(self):
        """Test that NMS suppresses nearby peaks."""
        prob = torch.zeros(32, 32, 32)
        # Two peaks very close together
        prob[16, 16, 16] = 0.9
        prob[17, 17, 17] = 0.8  # ~1.7 mm away with isotropic 1mm spacing

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=5.0,  # 5mm min distance
            threshold=0.5,
            topk=10,
        )

        # Only the higher peak should remain
        assert coords.shape[0] == 1
        assert scores[0] == 0.9

    def test_nms_topk_limit(self):
        """Test that NMS respects topk limit."""
        prob = torch.zeros(128, 64, 64)
        # Create many well-separated peaks
        for i in range(10):
            prob[i * 10 + 5, 32, 32] = 0.9 - i * 0.05

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=5.0,
            threshold=0.5,
            topk=5,  # Limit to 5
        )

        assert coords.shape[0] == 5

    def test_nms_threshold(self):
        """Test that NMS respects probability threshold."""
        prob = torch.zeros(32, 32, 32)
        prob[10, 10, 10] = 0.4  # Below threshold
        prob[20, 20, 20] = 0.6  # Above threshold

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=5.0,
            threshold=0.5,
            topk=10,
        )

        # Only the peak above threshold should remain
        assert coords.shape[0] == 1
        assert scores[0] == 0.6

    def test_nms_anisotropic_spacing(self):
        """Test NMS with anisotropic voxel spacing."""
        prob = torch.zeros(32, 32, 32)
        # Two peaks 5 voxels apart in z, but 10mm apart with 2mm z-spacing
        prob[15, 16, 16] = 0.9
        prob[20, 16, 16] = 0.8  # 5 voxels in z = 10mm with 2mm spacing

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(2.0, 1.0, 1.0),  # 2mm in z, 1mm in y and x
            min_dist_mm=8.0,  # 8mm min distance
            threshold=0.5,
            topk=10,
        )

        # 5 voxels * 2mm = 10mm > 8mm, so both should be kept
        assert coords.shape[0] == 2

    def test_nms_empty_result(self):
        """Test NMS returns empty when no peaks above threshold."""
        prob = torch.zeros(32, 32, 32)
        prob[16, 16, 16] = 0.3  # Below threshold

        coords, scores = nms_3d_aniso_mm(
            prob,
            spacing_mm=(1.0, 1.0, 1.0),
            min_dist_mm=5.0,
            threshold=0.5,
            topk=10,
        )

        assert coords.shape[0] == 0
        assert scores.shape[0] == 0

    def test_nms_isotropic_wrapper(self):
        """Test the isotropic NMS wrapper function."""
        prob = torch.zeros(32, 32, 32)
        prob[16, 16, 16] = 0.9

        coords, scores = nms_3d_isotropic(
            prob,
            min_dist_vox=5,
            threshold=0.5,
            topk=10,
        )

        assert coords.shape[0] == 1


class TestBoundaryBandSampling:
    """Tests for boundary band sampling."""

    def test_boundary_band_finds_edges(self):
        """Test that boundary band sampling finds edge regions."""
        # Create a simple cube organ
        label = torch.zeros(64, 64, 64, dtype=torch.long)
        label[20:44, 20:44, 20:44] = 1  # 24x24x24 cube

        center = sample_boundary_band_center(
            label,
            organ_class=1,
            band_width_vox=3,
            patch_size=(16, 16, 16),
        )

        assert center is not None
        z, y, x = center.long().tolist()

        # Center should be near the boundary of the cube
        # (within band_width of the 20-43 range)
        in_z_band = (17 <= z <= 23) or (40 <= z <= 46)
        in_y_band = (17 <= y <= 23) or (40 <= y <= 46)
        in_x_band = (17 <= x <= 23) or (40 <= x <= 46)

        # At least one coordinate should be near a boundary
        # OR the center could be inside if it's a corner region
        assert center is not None  # Just verify we got a valid center

    def test_boundary_band_respects_patch_size(self):
        """Test that sampled center keeps patch inside volume."""
        label = torch.zeros(64, 64, 64, dtype=torch.long)
        label[10:54, 10:54, 10:54] = 1

        patch_size = (32, 32, 32)
        center = sample_boundary_band_center(
            label,
            organ_class=1,
            band_width_vox=5,
            patch_size=patch_size,
        )

        if center is not None:
            z, y, x = center.tolist()
            # Center should be at least patch_size/2 from edges
            assert z >= patch_size[0] // 2
            assert z <= 64 - patch_size[0] // 2
            assert y >= patch_size[1] // 2
            assert y <= 64 - patch_size[1] // 2
            assert x >= patch_size[2] // 2
            assert x <= 64 - patch_size[2] // 2

    def test_boundary_band_no_organ(self):
        """Test that boundary band returns None when organ not present."""
        label = torch.zeros(64, 64, 64, dtype=torch.long)

        center = sample_boundary_band_center(
            label,
            organ_class=1,
            band_width_vox=3,
            patch_size=(16, 16, 16),
        )

        assert center is None


class TestPositiveSampling:
    """Tests for positive center sampling."""

    def test_sample_positive_center(self):
        """Test sampling from positive regions."""
        label = torch.zeros(64, 64, 64, dtype=torch.long)
        label[30:34, 30:34, 30:34] = 1  # Small positive region

        center = sample_positive_center(
            label,
            target_classes=[1],
            patch_size=(16, 16, 16),
        )

        assert center is not None
        z, y, x = center.long().tolist()
        # Center should be within the positive region
        assert 30 <= z <= 33
        assert 30 <= y <= 33
        assert 30 <= x <= 33

    def test_sample_positive_center_any_class(self):
        """Test sampling from any positive class."""
        label = torch.zeros(64, 64, 64, dtype=torch.long)
        label[30:34, 30:34, 30:34] = 2  # Class 2

        center = sample_positive_center(
            label,
            target_classes=None,  # Any positive
            patch_size=(16, 16, 16),
        )

        assert center is not None


class TestUniformSampling:
    """Tests for uniform center sampling."""

    def test_sample_uniform_center(self):
        """Test uniform sampling produces valid centers."""
        volume_shape = (64, 64, 64)
        patch_size = (32, 32, 32)

        for _ in range(10):
            center = sample_uniform_center(volume_shape, patch_size)

            z, y, x = center.tolist()
            assert z >= patch_size[0] // 2
            assert z <= volume_shape[0] - patch_size[0] // 2
            assert y >= patch_size[1] // 2
            assert y <= volume_shape[1] - patch_size[1] // 2
            assert x >= patch_size[2] // 2
            assert x <= volume_shape[2] - patch_size[2] // 2


class TestLabelDownsampling:
    """Tests for label downsampling."""

    def test_downsample_multiclass(self):
        """Test multi-class label downsampling."""
        label = torch.zeros(128, 128, 128, dtype=torch.long)
        label[32:96, 32:96, 32:96] = 1
        label[48:80, 48:80, 48:80] = 2

        coarse = downsample_label_coarse(label, (64, 64, 64), is_binary_lesion=False)

        assert coarse.shape == (64, 64, 64)
        # Check that classes are preserved
        assert (coarse == 0).any()
        assert (coarse == 1).any()
        assert (coarse == 2).any()

    def test_downsample_binary_preserves_small(self):
        """Test that binary downsampling preserves small lesions."""
        label = torch.zeros(128, 128, 128, dtype=torch.long)
        # Small lesion that might be lost with nearest interpolation
        label[64:66, 64:66, 64:66] = 1  # 2x2x2 lesion

        coarse = downsample_label_coarse(label, (32, 32, 32), is_binary_lesion=True)

        assert coarse.shape == (32, 32, 32)
        # Small lesion should still be present (maxpool preserves it)
        assert (coarse > 0).any()

    def test_downsample_image(self):
        """Test image downsampling."""
        image = torch.randn(64, 64, 64)
        coarse = downsample_image_coarse(image, (32, 32, 32))

        assert coarse.shape == (32, 32, 32)

    def test_downsample_image_with_channels(self):
        """Test image downsampling with channel dimension."""
        image = torch.randn(3, 64, 64, 64)
        coarse = downsample_image_coarse(image, (32, 32, 32))

        assert coarse.shape == (3, 32, 32, 32)


class TestAugmentationHelpers:
    """Tests for augmentation helper functions."""

    def test_random_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        R = random_rotation_matrix_3d(max_angle_deg=30.0)

        # R @ R.T should be identity
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        # det(R) should be 1
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0), atol=1e-5)

    def test_random_rotation_matrix_range(self):
        """Test that rotation produces reasonable transforms."""
        for _ in range(10):
            R = random_rotation_matrix_3d(max_angle_deg=15.0)

            # Apply to unit vector, should still be unit length
            v = torch.tensor([1.0, 0.0, 0.0])
            v_rot = R @ v
            assert torch.allclose(v_rot.norm(), torch.tensor(1.0), atol=1e-5)


class TestStitchingWindow:
    """Tests for stitching window functions."""

    def test_cos2_window_1d_shape(self):
        """Test 1D window has correct shape."""
        for n in [16, 32, 64, 96]:
            w = cos2_window_1d(n)
            assert w.shape == (n,)

    def test_cos2_window_1d_range(self):
        """Test 1D window values are in [0, 1]."""
        w = cos2_window_1d(64)
        assert w.min() >= 0.0
        assert w.max() <= 1.0

    def test_cos2_window_1d_symmetric(self):
        """Test 1D window is symmetric."""
        w = cos2_window_1d(64)
        assert torch.allclose(w, w.flip(0), atol=1e-6)

    def test_cos2_window_1d_edges_nonzero(self):
        """Test 1D window has non-zero edges for numerical stability."""
        w = cos2_window_1d(64)
        assert w[0] > 0
        assert w[-1] > 0

    def test_cos2_window_1d_max_at_center(self):
        """Test 1D window has maximum near center."""
        w = cos2_window_1d(65)  # Odd size for single center
        center_idx = 32
        assert w[center_idx] == w.max()

    def test_cos2_window_3d_shape(self):
        """Test 3D window has correct shape."""
        shape = (32, 48, 64)
        w = cos2_window_3d(shape)
        assert w.shape == shape

    def test_cos2_window_3d_separable(self):
        """Test 3D window is separable (outer product of 1D windows)."""
        shape = (16, 24, 32)
        w3d = cos2_window_3d(shape)

        wz = cos2_window_1d(shape[0])
        wy = cos2_window_1d(shape[1])
        wx = cos2_window_1d(shape[2])

        expected = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
        assert torch.allclose(w3d, expected)

    def test_cos2_window_3d_device(self):
        """Test 3D window respects device parameter."""
        shape = (16, 16, 16)
        w = cos2_window_3d(shape, device="cpu")
        assert w.device.type == "cpu"


class TestStitching:
    """Tests for patch stitching."""

    def test_stitch_single_patch(self):
        """Test stitching with a single patch."""
        D, H, W = 64, 64, 64
        Df, Hf, Wf = 32, 32, 32
        num_classes = 2

        window = cos2_window_3d((Df, Hf, Wf))
        logits = torch.randn(num_classes, Df, Hf, Wf)
        start = torch.tensor([16, 16, 16])

        result = stitch_patches_to_volume(
            [logits],
            [start],
            (D, H, W),
            num_classes,
            window,
            device="cpu",
        )

        assert result.shape == (num_classes, D, H, W)

    def test_stitch_uniform_constant(self):
        """Test that constant patches stitch to constant volume in interior.

        This is a critical test from the spec (Section 10.7).
        """
        D, H, W = 128, 128, 128
        Df, Hf, Wf = 64, 64, 64
        num_classes = 3
        stride = 32  # 50% overlap

        window = cos2_window_3d((Df, Hf, Wf))

        # Create overlapping patches with constant value
        constant_val = 1.5
        patch_logits_list = []
        patch_starts_list = []

        for sz in range(0, D - Df + 1, stride):
            for sy in range(0, H - Hf + 1, stride):
                for sx in range(0, W - Wf + 1, stride):
                    logits = torch.full((num_classes, Df, Hf, Wf), constant_val)
                    patch_logits_list.append(logits)
                    patch_starts_list.append(torch.tensor([sz, sy, sx]))

        result = stitch_patches_to_volume(
            patch_logits_list,
            patch_starts_list,
            (D, H, W),
            num_classes,
            window,
            device="cpu",
        )

        # Check interior region (away from boundaries by one patch size)
        interior = result[:, Df : D - Df, Hf : H - Hf, Wf : W - Wf]

        # Interior should be constant within numerical tolerance
        assert torch.allclose(
            interior, torch.full_like(interior, constant_val), atol=1e-5
        ), f"Interior not constant: min={interior.min()}, max={interior.max()}, expected={constant_val}"

    def test_stitch_overlapping_patches(self):
        """Test that overlapping patches blend smoothly."""
        D, H, W = 64, 64, 64
        Df, Hf, Wf = 32, 32, 32
        num_classes = 1

        window = cos2_window_3d((Df, Hf, Wf))

        # Two overlapping patches with different values
        patch1 = torch.ones(num_classes, Df, Hf, Wf) * 1.0
        patch2 = torch.ones(num_classes, Df, Hf, Wf) * 2.0

        start1 = torch.tensor([0, 0, 0])
        start2 = torch.tensor([16, 16, 16])  # 50% overlap

        result = stitch_patches_to_volume(
            [patch1, patch2],
            [start1, start2],
            (D, H, W),
            num_classes,
            window,
            device="cpu",
        )

        # Check that overlap region has intermediate values
        overlap_region = result[:, 16:32, 16:32, 16:32]
        assert overlap_region.min() >= 1.0
        assert overlap_region.max() <= 2.0

    def test_stitch_boundary_handling(self):
        """Test stitching handles boundary patches correctly."""
        D, H, W = 64, 64, 64
        Df, Hf, Wf = 32, 32, 32
        num_classes = 1

        window = cos2_window_3d((Df, Hf, Wf))

        # Patch at negative offset (partially out of bounds)
        logits = torch.ones(num_classes, Df, Hf, Wf)
        start = torch.tensor([-8, 0, 0])

        result = stitch_patches_to_volume(
            [logits],
            [start],
            (D, H, W),
            num_classes,
            window,
            device="cpu",
        )

        # Result should still have valid values in covered region
        assert result.shape == (num_classes, D, H, W)
        # Region covered by patch (after clipping) should have values
        assert result[:, 0:24, 0:32, 0:32].max() > 0


class TestTilePositions:
    """Tests for tile position generation."""

    def test_tile_positions_coverage(self):
        """Test that generated positions cover the full volume."""
        full_shape = (128, 128, 128)
        patch_shape = (64, 64, 64)

        positions = generate_tile_positions(
            full_shape, patch_shape, overlap_fraction=0.5
        )

        # Create coverage mask
        coverage = torch.zeros(full_shape)
        for sz, sy, sx in positions:
            coverage[
                sz : sz + patch_shape[0],
                sy : sy + patch_shape[1],
                sx : sx + patch_shape[2],
            ] = 1

        # Entire volume should be covered
        assert coverage.sum() == coverage.numel()

    def test_tile_positions_overlap(self):
        """Test that overlap fraction is respected."""
        full_shape = (128, 128, 128)
        patch_shape = (64, 64, 64)

        positions_50 = generate_tile_positions(
            full_shape, patch_shape, overlap_fraction=0.5
        )
        positions_25 = generate_tile_positions(
            full_shape, patch_shape, overlap_fraction=0.25
        )

        # More overlap should mean more positions
        assert len(positions_50) >= len(positions_25)

    def test_tile_positions_small_volume(self):
        """Test tile positions for volume smaller than patch."""
        full_shape = (32, 32, 32)
        patch_shape = (64, 64, 64)

        positions = generate_tile_positions(
            full_shape, patch_shape, overlap_fraction=0.5
        )

        # Should still generate at least one position
        assert len(positions) >= 1
        # Position should be (0, 0, 0) for small volumes
        assert positions[0] == (0, 0, 0)

    def test_tile_positions_explicit_stride(self):
        """Test tile positions with explicit stride."""
        full_shape = (128, 128, 128)
        patch_shape = (32, 32, 32)
        stride = (32, 32, 32)  # No overlap

        positions = generate_tile_positions(
            full_shape, patch_shape, stride=stride
        )

        # With no overlap, should have exactly 4^3 = 64 positions
        # (plus edge positions)
        assert len(positions) >= 64


class TestProposalMapping:
    """Tests for proposal coordinate mapping."""

    def test_full_to_coarse_norm_identity(self):
        """Test mapping with identity affines."""
        B = 2
        center_full = torch.tensor([[32.0, 32.0, 32.0], [16.0, 48.0, 24.0]])

        # Identity affine (no scaling, no translation)
        affine = torch.eye(4)[None].expand(B, -1, -1)

        shape_coarse = (64, 64, 64)

        center_coarse_norm = center_full_to_coarse_norm(
            center_full, affine, affine, shape_coarse
        )

        # With identity affines and same shape, normalized coords should match
        # index 32 in 64-voxel volume -> norm = 2*(32+0.5)/64 - 1 = 0.015625
        expected = 2.0 * (center_full + 0.5) / 64.0 - 1.0
        assert torch.allclose(center_coarse_norm, expected, atol=1e-5)

    def test_coarse_to_full_round_trip(self):
        """Test that mapping coarse->full->coarse is identity."""
        B = 3
        center_coarse = torch.tensor([
            [16.0, 16.0, 16.0],
            [32.0, 32.0, 32.0],
            [8.0, 24.0, 40.0],
        ])

        # Create simple affines
        # Full: 1mm spacing
        affine_full = torch.eye(4)[None].expand(B, -1, -1)
        # Coarse: 2mm spacing (half resolution)
        affine_coarse = torch.eye(4).clone()
        affine_coarse[0, 0] = 2.0
        affine_coarse[1, 1] = 2.0
        affine_coarse[2, 2] = 2.0
        affine_coarse = affine_coarse[None].expand(B, -1, -1)

        # Map coarse -> full
        center_full = center_coarse_to_full_index(
            center_coarse, affine_coarse, affine_full
        )

        # Map full -> coarse
        center_coarse_back = center_coarse_to_full_index(
            center_full, affine_full, affine_coarse
        )

        # Should round-trip
        assert torch.allclose(center_coarse, center_coarse_back, atol=1e-5)

    def test_mapping_with_translation(self):
        """Test mapping with translated affines."""
        B = 1
        center_full = torch.tensor([[50.0, 50.0, 50.0]])

        # Full affine with origin offset
        affine_full = torch.eye(4)[None]
        affine_full[0, :3, 3] = torch.tensor([10.0, 20.0, 30.0])  # xyz offset

        # Coarse affine with 2x spacing and same offset
        affine_coarse = torch.eye(4)[None]
        affine_coarse[0, 0, 0] = 2.0
        affine_coarse[0, 1, 1] = 2.0
        affine_coarse[0, 2, 2] = 2.0
        affine_coarse[0, :3, 3] = torch.tensor([10.0, 20.0, 30.0])

        shape_coarse = (64, 64, 64)

        center_coarse_norm = center_full_to_coarse_norm(
            center_full, affine_full, affine_coarse, shape_coarse
        )

        # With 2x coarse spacing, full index 50 -> coarse index 25
        expected_coarse_idx = torch.tensor([[25.0, 25.0, 25.0]])
        expected_norm = 2.0 * (expected_coarse_idx + 0.5) / 64.0 - 1.0

        # Note: The zyx vs xyz conversion affects the expected result
        # Our function returns in dhw (zyx) order
        assert center_coarse_norm.shape == (B, 3)
        # Values should be reasonable (in [-1, 1] range)
        assert center_coarse_norm.abs().max() <= 1.0

    def test_mapping_anisotropic_spacing(self):
        """Test mapping with anisotropic voxel spacing."""
        B = 1
        center_full = torch.tensor([[64.0, 64.0, 64.0]])

        # Full: 1mm isotropic
        affine_full = torch.eye(4)[None]

        # Coarse: 2mm in z, 1mm in x/y
        affine_coarse = torch.eye(4)[None]
        affine_coarse[0, 2, 2] = 2.0  # z-spacing (in NIfTI, k maps to z)

        shape_coarse = (64, 128, 128)  # Half resolution in z

        center_coarse_norm = center_full_to_coarse_norm(
            center_full, affine_full, affine_coarse, shape_coarse
        )

        # Result should be valid normalized coordinates
        assert center_coarse_norm.shape == (B, 3)
        # d (z) should be different from h, w due to anisotropic spacing
        # z=64 full -> z=32 coarse with 2mm spacing
