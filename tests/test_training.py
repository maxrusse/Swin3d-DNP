"""Tests for training pipeline components.

Tests cover:
- PhaseScheduler behavior
- Training utilities (seeding, memory estimation)
- PatchSampler strategies
- Trainer basic functionality
"""

import pytest
import torch
import numpy as np

from swin3d_dnp.training import (
    PhaseScheduler,
    seed_everything,
    get_worker_init_fn,
    estimate_memory_gb,
    get_memory_mitigation_tips,
)
from swin3d_dnp.data.sampling import (
    PatchSampler,
    sample_uniform_center,
    sample_positive_center,
    sample_boundary_band_center,
    sample_mixed_centers,
)


class TestPhaseScheduler:
    """Test PhaseScheduler phase transitions and parameter scheduling."""

    def test_phase_transitions(self):
        """Test that phases transition at correct step boundaries."""
        scheduler = PhaseScheduler(
            total_steps=100,
            phase1_end=0.10,  # Step 10
            phase2_end=0.60,  # Step 60
        )

        # Phase 1: steps 0-9
        for step in range(10):
            assert scheduler.get_phase(step) == 1, f"Step {step} should be phase 1"

        # Phase 2: steps 10-59
        for step in range(10, 60):
            assert scheduler.get_phase(step) == 2, f"Step {step} should be phase 2"

        # Phase 3: steps 60-99
        for step in range(60, 100):
            assert scheduler.get_phase(step) == 3, f"Step {step} should be phase 3"

    def test_lambda_values(self):
        """Test that lambda values change correctly per phase."""
        scheduler = PhaseScheduler(
            total_steps=100,
            lambda0_phase1=1.0,
            lambda1_phase1=0.5,
            lambda0_phase2=0.5,
            lambda1_phase2=1.0,
            lambda0_phase3=0.3,
            lambda1_phase3=1.0,
        )

        # Phase 1
        l0, l1 = scheduler.get_lambdas(5)
        assert l0 == 1.0
        assert l1 == 0.5

        # Phase 2
        l0, l1 = scheduler.get_lambdas(30)
        assert l0 == 0.5
        assert l1 == 1.0

        # Phase 3
        l0, l1 = scheduler.get_lambdas(80)
        assert l0 == 0.3
        assert l1 == 1.0

    def test_context_detachment(self):
        """Test context detachment flag per phase."""
        scheduler = PhaseScheduler(total_steps=100)

        # Phase 1: detach
        assert scheduler.should_detach_context(5) is True

        # Phase 2/3: no detach
        assert scheduler.should_detach_context(30) is False
        assert scheduler.should_detach_context(80) is False

    def test_hard_negative_warmup(self):
        """Test hard negative mining activation."""
        scheduler = PhaseScheduler(
            total_steps=10000,
            hardneg_warmup_steps=500,
        )

        # Before warmup (still phase 1)
        assert scheduler.should_use_hard_negatives(100) is False

        # After warmup but still phase 1
        assert scheduler.should_use_hard_negatives(800) is False

        # Phase 2 after warmup: enabled
        assert scheduler.should_use_hard_negatives(1500) is True

    def test_lr_scale_phase3(self):
        """Test LR scaling in phase 3."""
        scheduler = PhaseScheduler(
            total_steps=100,
            lr_scale_phase3=0.1,
        )

        # Phases 1-2: scale = 1.0
        assert scheduler.get_lr_scale(5) == 1.0
        assert scheduler.get_lr_scale(30) == 1.0

        # Phase 3: scale = 0.1
        assert scheduler.get_lr_scale(80) == 0.1

    def test_step_returns_complete_info(self):
        """Test that step() returns all required information."""
        scheduler = PhaseScheduler(total_steps=100)
        info = scheduler.step(50)

        required_keys = [
            "phase",
            "lambda0",
            "lambda1",
            "detach_context",
            "use_hard_neg",
            "lr_scale",
            "progress",
        ]

        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        assert info["progress"] == 0.5

    def test_state_dict_roundtrip(self):
        """Test saving and loading scheduler state."""
        scheduler = PhaseScheduler(total_steps=1000)
        scheduler.step(500)

        state = scheduler.state_dict()

        new_scheduler = PhaseScheduler(total_steps=100)  # Different initial value
        new_scheduler.load_state_dict(state)

        assert new_scheduler.total_steps == 1000
        assert new_scheduler._current_step == 500


class TestSeedingUtilities:
    """Test reproducibility utilities."""

    def test_seed_everything_pytorch(self):
        """Test that seed_everything makes PyTorch deterministic."""
        seed_everything(42)
        t1 = torch.randn(10)

        seed_everything(42)
        t2 = torch.randn(10)

        assert torch.allclose(t1, t2)

    def test_seed_everything_numpy(self):
        """Test that seed_everything makes NumPy deterministic."""
        seed_everything(42)
        a1 = np.random.randn(10)

        seed_everything(42)
        a2 = np.random.randn(10)

        assert np.allclose(a1, a2)

    def test_worker_init_fn_different_seeds(self):
        """Test that different workers get different seeds."""
        init_fn = get_worker_init_fn(42)

        # Simulate two workers
        init_fn(0)
        vals_worker0 = torch.randn(5).tolist()

        init_fn(1)
        vals_worker1 = torch.randn(5).tolist()

        # Workers should have different random states
        assert vals_worker0 != vals_worker1


class TestMemoryEstimation:
    """Test memory estimation utilities."""

    def test_estimate_returns_all_components(self):
        """Test that estimate_memory_gb returns all required fields."""
        result = estimate_memory_gb()

        required_keys = [
            "activations_gb",
            "gradients_gb",
            "parameters_gb",
            "optimizer_gb",
            "total_estimated_gb",
        ]

        for key in required_keys:
            assert key in result
            assert result[key] > 0

    def test_estimate_scales_with_batch_size(self):
        """Test that memory estimate scales with batch size."""
        mem_b1 = estimate_memory_gb(batch_size=1)
        mem_b2 = estimate_memory_gb(batch_size=2)

        # Activations should roughly double
        assert mem_b2["activations_gb"] > mem_b1["activations_gb"]

    def test_mitigation_tips_when_insufficient(self):
        """Test mitigation tips for insufficient memory."""
        tips = get_memory_mitigation_tips(available_gb=8.0, required_gb=24.0)

        assert len(tips) > 1
        assert any("AMP" in tip for tip in tips)

    def test_mitigation_tips_when_sufficient(self):
        """Test mitigation tips when memory is sufficient."""
        tips = get_memory_mitigation_tips(available_gb=48.0, required_gb=24.0)

        assert "Memory requirements satisfied" in tips[0]


class TestPatchSampler:
    """Test unified patch sampling strategies."""

    @pytest.fixture
    def sample_label(self):
        """Create a sample label tensor with organs."""
        label = torch.zeros((64, 64, 64), dtype=torch.long)
        # Add organ (class 1) in center region
        label[20:40, 20:40, 20:40] = 1
        # Add second organ (class 2) in corner
        label[5:15, 5:15, 5:15] = 2
        return label

    def test_uniform_sampling_in_valid_range(self, sample_label):
        """Test uniform sampling stays in valid range."""
        patch_size = (16, 16, 16)
        sampler = PatchSampler(patch_size=patch_size)

        for _ in range(20):
            center, mode = sampler.sample(sample_label, mode="uniform")
            assert mode == "uniform"

            z, y, x = center.tolist()
            D, H, W = sample_label.shape

            # Center should allow full patch to fit
            assert z >= patch_size[0] // 2
            assert z < D - patch_size[0] // 2
            assert y >= patch_size[1] // 2
            assert y < H - patch_size[1] // 2
            assert x >= patch_size[2] // 2
            assert x < W - patch_size[2] // 2

    def test_positive_sampling_on_foreground(self, sample_label):
        """Test positive sampling returns foreground voxels."""
        patch_size = (16, 16, 16)
        sampler = PatchSampler(patch_size=patch_size)

        for _ in range(20):
            center, mode = sampler.sample(sample_label, mode="positive")
            if mode == "positive":  # May fallback to uniform
                z, y, x = center.long().tolist()
                assert sample_label[z, y, x] > 0

    def test_positive_sampling_with_target_classes(self, sample_label):
        """Test positive sampling respects target_classes."""
        patch_size = (16, 16, 16)
        sampler = PatchSampler(patch_size=patch_size, target_classes=[2])

        found_class_2 = False
        for _ in range(50):
            center, mode = sampler.sample(sample_label, mode="positive")
            if mode == "positive":
                z, y, x = center.long().tolist()
                if sample_label[z, y, x] == 2:
                    found_class_2 = True
                    break

        assert found_class_2, "Should find class 2 voxels"

    def test_boundary_sampling_near_edges(self, sample_label):
        """Test boundary sampling finds edge regions."""
        patch_size = (8, 8, 8)
        sampler = PatchSampler(patch_size=patch_size, boundary_classes=[1])

        found_boundary = False
        for _ in range(50):
            center, mode = sampler.sample(sample_label, mode="boundary")
            if mode == "boundary":
                found_boundary = True
                break

        assert found_boundary, "Should find boundary voxels"

    def test_auto_mode_uses_ratios(self, sample_label):
        """Test auto mode samples according to ratios."""
        patch_size = (8, 8, 8)
        sampler = PatchSampler(
            patch_size=patch_size,
            ratio_uniform=0.5,
            ratio_positive=0.3,
            ratio_boundary=0.2,
        )

        mode_counts = {"uniform": 0, "positive": 0, "boundary": 0}
        n_samples = 200

        for _ in range(n_samples):
            _, mode = sampler.sample(sample_label, mode="auto")
            mode_counts[mode] += 1

        # Uniform should be most common (50% target)
        assert mode_counts["uniform"] > n_samples * 0.3

    def test_batch_sampling(self, sample_label):
        """Test batch sampling returns correct shapes."""
        patch_size = (8, 8, 8)
        sampler = PatchSampler(patch_size=patch_size)

        centers, modes = sampler.sample_batch(sample_label, n_samples=10)

        assert centers.shape == (10, 3)
        assert len(modes) == 10

    def test_fallback_to_uniform(self):
        """Test fallback to uniform when no foreground exists."""
        # Label with no foreground
        label = torch.zeros((64, 64, 64), dtype=torch.long)
        patch_size = (8, 8, 8)
        sampler = PatchSampler(patch_size=patch_size)

        # Positive should fall back to uniform
        center, mode = sampler.sample(label, mode="positive")
        assert mode == "uniform"


class TestSamplingFunctions:
    """Test individual sampling functions."""

    def test_sample_uniform_center_shape(self):
        """Test uniform center has correct shape."""
        center = sample_uniform_center(
            volume_shape=(64, 64, 64),
            patch_size=(16, 16, 16),
        )

        assert center.shape == (3,)

    def test_sample_positive_center_returns_none_when_empty(self):
        """Test positive sampling returns None with no foreground."""
        label = torch.zeros((32, 32, 32), dtype=torch.long)
        center = sample_positive_center(label, patch_size=(8, 8, 8))

        assert center is None

    def test_sample_boundary_band_center_returns_none_when_no_organ(self):
        """Test boundary sampling returns None when organ doesn't exist."""
        label = torch.zeros((32, 32, 32), dtype=torch.long)
        center = sample_boundary_band_center(
            label, organ_class=1, patch_size=(8, 8, 8)
        )

        assert center is None

    def test_sample_mixed_centers_convenience(self):
        """Test convenience function sample_mixed_centers."""
        label = torch.zeros((64, 64, 64), dtype=torch.long)
        label[20:40, 20:40, 20:40] = 1

        centers, modes = sample_mixed_centers(
            label_full=label,
            n_samples=5,
            patch_size=(8, 8, 8),
        )

        assert centers.shape == (5, 3)
        assert len(modes) == 5


class TestSchedulerWithModel:
    """Test scheduler integration with model."""

    def test_apply_to_model_sets_phase(self):
        """Test that apply_to_model calls model.set_phase."""
        class MockModel:
            def __init__(self):
                self.phase = None

            def set_phase(self, phase):
                self.phase = phase

        model = MockModel()
        scheduler = PhaseScheduler(total_steps=100)

        scheduler.apply_to_model(model, 5)
        assert model.phase == 1

        scheduler.apply_to_model(model, 50)
        assert model.phase == 2

        scheduler.apply_to_model(model, 80)
        assert model.phase == 3


class TestGradScalerUtilities:
    """Test gradient scaler utilities."""

    def test_get_grad_scaler_creates_scaler(self):
        """Test that get_grad_scaler returns a valid scaler."""
        from swin3d_dnp.training.utils import get_grad_scaler

        scaler = get_grad_scaler(enabled=True)
        assert scaler is not None

    def test_clip_grad_norm_with_no_grads(self):
        """Test clip_grad_norm_ handles no gradients gracefully."""
        from swin3d_dnp.training.utils import clip_grad_norm_

        param = torch.nn.Parameter(torch.randn(10))
        # No backward called, so no gradient

        norm = clip_grad_norm_([param], max_norm=1.0)
        assert norm.item() == 0.0


class TestMoveToDevice:
    """Test batch device moving utility."""

    def test_move_batch_to_device(self, device):
        """Test moving batch dict to device."""
        from swin3d_dnp.training.utils import move_batch_to_device

        batch = {
            "tensor1": torch.randn(3, 3),
            "tensor2": torch.randn(5),
            "non_tensor": "string_value",
        }

        moved = move_batch_to_device(batch, device)

        assert moved["tensor1"].device == device
        assert moved["tensor2"].device == device
        assert moved["non_tensor"] == "string_value"
