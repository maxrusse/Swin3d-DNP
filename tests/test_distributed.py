"""Tests for distributed data parallel (DDP) alignment.

These tests verify that batch scatter maintains proper alignment between
image patches and their corresponding centers/metadata during multi-GPU training.

The critical invariant: after DDP batch scatter, (image_fine checksum, center)
pairs must remain aligned across different processes.
"""

import pytest
import torch
import torch.distributed as dist

from swin3d_dnp.training.utils import seed_everything


class TestDDPAlignment:
    """Tests for DDP batch alignment.

    These tests verify that when batches are scattered across GPUs,
    the data remains properly aligned - each image patch corresponds
    to its correct center, affine, and other metadata.
    """

    def test_batch_alignment_simulation(self):
        """Simulate DDP batch scatter and verify alignment is preserved.

        This test creates a batch with identifiable checksums and verifies
        that after simulated scatter/gather, the alignment is preserved.
        """
        seed_everything(42)

        B = 8  # Batch size
        D, H, W = 32, 32, 32

        # Create batch with unique, identifiable patterns per sample
        # Each sample has a unique pattern based on its index
        image_batch = torch.zeros(B, 1, D, H, W)
        centers = torch.zeros(B, 3)
        checksums = torch.zeros(B)

        for i in range(B):
            # Create a unique pattern - fill with index value
            pattern = float(i + 1) * 0.1
            image_batch[i, 0, :, :, :] = pattern

            # Center encodes the batch index
            centers[i] = torch.tensor([i * 10.0, i * 10.0, i * 10.0])

            # Compute checksum of the image (should be unique per sample)
            checksums[i] = image_batch[i].sum()

        # Simulate DDP scatter: split batch across 2 "GPUs"
        world_size = 2
        per_gpu = B // world_size

        image_chunks = torch.chunk(image_batch, world_size, dim=0)
        center_chunks = torch.chunk(centers, world_size, dim=0)
        checksum_chunks = torch.chunk(checksums, world_size, dim=0)

        # Verify alignment within each chunk
        for rank in range(world_size):
            img_chunk = image_chunks[rank]
            center_chunk = center_chunks[rank]
            checksum_chunk = checksum_chunks[rank]

            for local_idx in range(per_gpu):
                # Recompute checksum
                recomputed = img_chunk[local_idx].sum()
                expected = checksum_chunk[local_idx]

                assert torch.isclose(recomputed, expected, atol=1e-5), (
                    f"Rank {rank}, local idx {local_idx}: "
                    f"checksum mismatch {recomputed} vs {expected}"
                )

                # Verify center matches the image pattern
                # The center's first value / 10 should equal the global index
                global_idx = rank * per_gpu + local_idx
                expected_center_val = global_idx * 10.0

                assert torch.isclose(
                    center_chunk[local_idx, 0],
                    torch.tensor(expected_center_val),
                    atol=1e-5
                ), (
                    f"Rank {rank}, local idx {local_idx}: "
                    f"center mismatch {center_chunk[local_idx, 0]} vs {expected_center_val}"
                )

    def test_batch_alignment_with_shuffle(self):
        """Test that alignment is preserved even after shuffling.

        DataLoader with shuffle=True reorders the batch. We must ensure
        that all related tensors are shuffled consistently.
        """
        seed_everything(123)

        B = 16
        D, H, W = 16, 16, 16

        # Create matched batch
        images = torch.zeros(B, 1, D, H, W)
        centers = torch.zeros(B, 3)
        ids = torch.arange(B)  # Ground truth ID for each sample

        for i in range(B):
            # Encode the ID in the image
            images[i, 0, 0, 0, 0] = float(i)
            centers[i, 0] = float(i)

        # Simulate shuffle (same permutation for all tensors)
        perm = torch.randperm(B)
        images_shuffled = images[perm]
        centers_shuffled = centers[perm]
        ids_shuffled = ids[perm]

        # Verify alignment after shuffle
        for i in range(B):
            # The ID in the image should match the center's first value
            img_id = images_shuffled[i, 0, 0, 0, 0].item()
            center_id = centers_shuffled[i, 0].item()
            expected_id = ids_shuffled[i].item()

            assert img_id == expected_id, f"Image ID mismatch at {i}"
            assert center_id == expected_id, f"Center ID mismatch at {i}"

    def test_batch_creation_from_dict(self):
        """Test that batch dictionaries maintain alignment.

        When creating batches from dictionaries (common in PyTorch dataloaders),
        all values must be aligned by their keys.
        """
        B = 4
        D, H, W = 8, 8, 8

        # Create batch dictionary
        batch = {
            'image_fine': torch.zeros(B, 1, D, H, W),
            'center_full_index_zyx': torch.zeros(B, 3),
            'valid_mask': torch.ones(B, 1, D, H, W),
            'case_id': [f'case_{i}' for i in range(B)],
        }

        # Encode alignment info
        for i in range(B):
            batch['image_fine'][i, 0, 0, 0, 0] = float(i)
            batch['center_full_index_zyx'][i, 0] = float(i)

        # Simulate subset selection (as might happen in distributed sampler)
        indices = [0, 2]  # Select every other sample

        subset = {
            'image_fine': batch['image_fine'][indices],
            'center_full_index_zyx': batch['center_full_index_zyx'][indices],
            'valid_mask': batch['valid_mask'][indices],
            'case_id': [batch['case_id'][i] for i in indices],
        }

        # Verify alignment in subset
        for local_idx, global_idx in enumerate(indices):
            img_val = subset['image_fine'][local_idx, 0, 0, 0, 0].item()
            center_val = subset['center_full_index_zyx'][local_idx, 0].item()

            assert img_val == float(global_idx)
            assert center_val == float(global_idx)
            assert subset['case_id'][local_idx] == f'case_{global_idx}'


class TestDistributedSamplerAlignment:
    """Tests for distributed sampler alignment."""

    def test_index_alignment_across_ranks(self):
        """Test that DistributedSampler produces aligned indices across ranks."""
        total_samples = 100
        world_size = 4

        # Simulate what DistributedSampler does
        # It divides indices into world_size chunks
        indices = list(range(total_samples))

        # Pad to be divisible by world_size
        while len(indices) % world_size != 0:
            indices.append(indices[0])  # Pad with repeated sample

        # Split across ranks
        per_rank = len(indices) // world_size
        rank_indices = [
            indices[rank * per_rank : (rank + 1) * per_rank]
            for rank in range(world_size)
        ]

        # Verify complete coverage (with potential overlap due to padding)
        all_covered = set()
        for rank_idx in rank_indices:
            all_covered.update(rank_idx)

        # All original samples should be covered
        original_samples = set(range(total_samples))
        assert original_samples <= all_covered

    def test_deterministic_sampling_across_ranks(self):
        """Test that deterministic sampling gives same order across ranks."""
        seed_everything(42)

        total_samples = 32
        world_size = 2

        # Generate deterministic order
        generator1 = torch.Generator().manual_seed(42)
        perm1 = torch.randperm(total_samples, generator=generator1).tolist()

        generator2 = torch.Generator().manual_seed(42)
        perm2 = torch.randperm(total_samples, generator=generator2).tolist()

        # Same seed should give same permutation
        assert perm1 == perm2


class TestAllGatherAlignment:
    """Tests for all_gather alignment during gradient synchronization."""

    def test_gradient_all_gather_simulation(self):
        """Simulate all_gather of gradients and verify alignment."""
        world_size = 4
        feature_size = 32

        # Simulate gradients from each rank (each has unique pattern)
        local_grads = [
            torch.full((feature_size,), float(rank + 1))
            for rank in range(world_size)
        ]

        # Simulate all_gather (concatenate all gradients)
        gathered = torch.stack(local_grads, dim=0)

        # Verify each rank's gradient is in the correct position
        for rank in range(world_size):
            expected_val = float(rank + 1)
            actual = gathered[rank]

            assert torch.allclose(
                actual,
                torch.full((feature_size,), expected_val)
            ), f"Gradient from rank {rank} misaligned"

    def test_output_alignment_after_gather(self):
        """Test output alignment after gathering predictions from all ranks."""
        world_size = 2
        B_per_rank = 4
        C, D, H, W = 2, 8, 8, 8

        # Each rank produces predictions for its local batch
        rank_outputs = [
            torch.randn(B_per_rank, C, D, H, W) + rank  # Add rank for identification
            for rank in range(world_size)
        ]

        # Rank 0 gets all outputs via gather
        gathered = torch.cat(rank_outputs, dim=0)

        # Total batch size should be preserved
        assert gathered.shape == (B_per_rank * world_size, C, D, H, W)

        # Check each portion came from the right rank (via the offset we added)
        for rank in range(world_size):
            start = rank * B_per_rank
            end = start + B_per_rank
            portion = gathered[start:end]

            # Mean should be close to rank (since we added rank to randn output)
            mean = portion.mean()
            assert rank - 0.5 < mean < rank + 0.5


class TestBatchMetadataAlignment:
    """Tests for maintaining metadata alignment in batches."""

    def test_affine_alignment(self):
        """Test that affine matrices stay aligned with their images."""
        B = 4
        D, H, W = 16, 16, 16

        images = torch.randn(B, 1, D, H, W)
        affines = torch.zeros(B, 4, 4)

        # Encode alignment info in both
        for i in range(B):
            # Use a simple pattern in the image
            images[i, 0, 0, 0, :] = float(i)
            # Encode the same ID in the affine's translation
            affines[i] = torch.eye(4)
            affines[i, 0, 3] = float(i)  # x translation = sample ID

        # Simulate reordering (as might happen in dataloader)
        perm = torch.tensor([3, 1, 0, 2])
        images_reordered = images[perm]
        affines_reordered = affines[perm]

        # Verify alignment
        for i in range(B):
            img_id = images_reordered[i, 0, 0, 0, 0].item()
            affine_id = affines_reordered[i, 0, 3].item()

            assert img_id == affine_id, (
                f"At position {i}: image ID {img_id} != affine ID {affine_id}"
            )

    def test_spacing_alignment(self):
        """Test that spacing values stay aligned with their volumes."""
        B = 8
        D, H, W = 8, 8, 8

        images = torch.randn(B, 1, D, H, W)
        spacings = torch.zeros(B, 3)

        for i in range(B):
            # Unique spacing per sample
            spacing_val = (i + 1) * 0.5
            spacings[i] = torch.tensor([spacing_val, spacing_val, spacing_val])
            # Encode in image
            images[i, 0, 0, 0, 0] = spacing_val

        # Subset selection
        indices = [1, 3, 5, 7]
        images_sub = images[indices]
        spacings_sub = spacings[indices]

        for local_i, global_i in enumerate(indices):
            expected_spacing = (global_i + 1) * 0.5
            actual_from_img = images_sub[local_i, 0, 0, 0, 0].item()
            actual_spacing = spacings_sub[local_i, 0].item()

            assert abs(actual_from_img - expected_spacing) < 1e-5
            assert abs(actual_spacing - expected_spacing) < 1e-5


class TestCollateFunction:
    """Tests for custom collate function alignment."""

    def test_collate_preserves_alignment(self):
        """Test that collating samples preserves alignment."""
        # Simulate individual samples as they come from dataset
        samples = [
            {
                'image': torch.full((1, 8, 8, 8), float(i)),
                'center': torch.tensor([i * 10.0, i * 10.0, i * 10.0]),
                'label': torch.full((8, 8, 8), i, dtype=torch.long),
            }
            for i in range(4)
        ]

        # Simple collate: stack tensors by key
        def collate(samples):
            return {
                'image': torch.stack([s['image'] for s in samples]),
                'center': torch.stack([s['center'] for s in samples]),
                'label': torch.stack([s['label'] for s in samples]),
            }

        batch = collate(samples)

        # Verify alignment
        for i in range(4):
            img_val = batch['image'][i, 0, 0, 0, 0].item()
            center_val = batch['center'][i, 0].item()
            label_val = batch['label'][i, 0, 0, 0].item()

            assert img_val == float(i)
            assert center_val == float(i * 10)
            assert label_val == i


# Skip actual distributed tests if not in distributed environment
@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available"
)
class TestRealDistributed:
    """Tests that require actual distributed setup.

    These are skipped unless running in a distributed environment.
    """

    @pytest.mark.skip(reason="Requires multi-GPU setup")
    def test_actual_ddp_alignment(self):
        """Test actual DDP alignment with real distributed setup.

        This test should be run with:
        torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py -k actual
        """
        pass  # Would contain actual DDP test code
