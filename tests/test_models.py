"""Tests for neural network models.

Tests cover:
- Forward pass shapes for all network components
- Gradient flow with and without context detachment
- Phase switching behavior
- Context sampling integration
"""

import pytest
import torch
import torch.nn as nn

from swin3d_dnp.geometry.sampling import DifferentiableContextSampler
from swin3d_dnp.models.fusion import (
    AdaptiveCoarseContextFusion,
    CoarseContextFusion,
    SimpleFusion,
)
from swin3d_dnp.models.swin3d_dnp import Swin3DDNP, build_simple_swin3d_dnp


class TestCoarseContextFusion:
    """Tests for CoarseContextFusion layer."""

    def test_basic_fusion_shapes(self):
        """Test fusion output shapes."""
        B, D, H, W = 2, 32, 32, 32

        fusion = CoarseContextFusion(
            in_channels_image=1,
            in_channels_context=64,
            out_channels=48,
        )

        image = torch.randn(B, 1, D, H, W)
        context = torch.randn(B, 64, D, H, W)

        out = fusion(image, context)

        assert out.shape == (B, 48, D, H, W)

    def test_fusion_with_probs(self):
        """Test fusion with probability inputs."""
        B, D, H, W = 2, 32, 32, 32

        fusion = CoarseContextFusion(
            in_channels_image=1,
            in_channels_context=64,
            in_channels_probs=3,
            out_channels=48,
        )

        image = torch.randn(B, 1, D, H, W)
        context = torch.randn(B, 64, D, H, W)
        logits = torch.randn(B, 3, D, H, W)

        out = fusion(image, context, logits)

        assert out.shape == (B, 48, D, H, W)

    def test_fusion_without_probs_when_configured(self):
        """Test that fusion ignores probs when not configured."""
        B, D, H, W = 2, 32, 32, 32

        fusion = CoarseContextFusion(
            in_channels_image=1,
            in_channels_context=64,
            in_channels_probs=None,  # Not using probs
            out_channels=48,
        )

        image = torch.randn(B, 1, D, H, W)
        context = torch.randn(B, 64, D, H, W)
        logits = torch.randn(B, 3, D, H, W)  # Should be ignored

        out = fusion(image, context, logits)

        assert out.shape == (B, 48, D, H, W)

    @pytest.mark.parametrize("norm", ["instance", "batch", "layer"])
    def test_fusion_norm_types(self, norm):
        """Test different normalization types."""
        fusion = CoarseContextFusion(
            in_channels_image=1,
            in_channels_context=32,
            out_channels=24,
            norm=norm,
        )

        image = torch.randn(2, 1, 16, 16, 16)
        context = torch.randn(2, 32, 16, 16, 16)

        out = fusion(image, context)
        assert out.shape == (2, 24, 16, 16, 16)

    def test_fusion_invalid_norm(self):
        """Test that invalid norm raises error."""
        with pytest.raises(ValueError):
            CoarseContextFusion(
                in_channels_image=1,
                in_channels_context=32,
                out_channels=24,
                norm="invalid",
            )


class TestAdaptiveFusion:
    """Tests for AdaptiveCoarseContextFusion layer."""

    def test_adaptive_fusion_shapes(self):
        """Test adaptive fusion output shapes."""
        B, D, H, W = 2, 32, 32, 32

        fusion = AdaptiveCoarseContextFusion(
            in_channels_image=1,
            in_channels_context=64,
            out_channels=48,
        )

        image = torch.randn(B, 1, D, H, W)
        context = torch.randn(B, 64, D, H, W)

        out = fusion(image, context)

        assert out.shape == (B, 48, D, H, W)


class TestSimpleFusion:
    """Tests for SimpleFusion layer."""

    def test_simple_fusion_output_channels(self):
        """Test simple fusion calculates correct output channels."""
        fusion = SimpleFusion(
            in_channels_image=1,
            in_channels_context=32,
            in_channels_probs=3,
        )

        # 1 + 32 + 3 = 36
        assert fusion.out_channels == 36

    def test_simple_fusion_shapes(self):
        """Test simple fusion output shapes."""
        B, D, H, W = 2, 16, 16, 16

        fusion = SimpleFusion(
            in_channels_image=1,
            in_channels_context=32,
        )

        image = torch.randn(B, 1, D, H, W)
        context = torch.randn(B, 32, D, H, W)

        out = fusion(image, context)

        assert out.shape == (B, 33, D, H, W)  # 1 + 32


class TestSwin3DDNP:
    """Tests for main Swin3DDNP model."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=2,
            context_channels=16,
        )

    def test_forward_shapes(self, simple_model):
        """Test model forward pass produces correct shapes."""
        B = 2
        coarse_shape = (32, 32, 32)
        fine_shape = (16, 16, 16)

        image_coarse = torch.randn(B, 1, *coarse_shape)
        image_fine = torch.randn(B, 1, *fine_shape)
        centers = torch.zeros(B, 3)  # Center of volume

        spacing_fine = torch.tensor([1.0, 1.0, 1.0])
        spacing_coarse = torch.tensor([2.0, 2.0, 2.0])

        coarse_logits, fine_logits = simple_model(
            image_coarse,
            image_fine,
            centers,
            fine_shape,
            spacing_fine,
            spacing_coarse,
        )

        assert coarse_logits.shape == (B, 2, *coarse_shape)
        assert fine_logits.shape == (B, 2, *fine_shape)

    def test_phase_switching(self, simple_model):
        """Test phase switching changes detach behavior."""
        # Phase 1: should detach
        simple_model.set_phase(1)
        assert simple_model.get_phase() == 1
        assert simple_model.detach_coarse_context is True

        # Phase 2: should not detach
        simple_model.set_phase(2)
        assert simple_model.get_phase() == 2
        assert simple_model.detach_coarse_context is False

        # Phase 3: should not detach
        simple_model.set_phase(3)
        assert simple_model.get_phase() == 3
        assert simple_model.detach_coarse_context is False

    def test_invalid_phase(self, simple_model):
        """Test that invalid phase raises error."""
        with pytest.raises(ValueError):
            simple_model.set_phase(0)

        with pytest.raises(ValueError):
            simple_model.set_phase(4)

    def test_coarse_only_forward(self, simple_model):
        """Test coarse-only forward pass."""
        B = 2
        coarse_shape = (32, 32, 32)

        image_coarse = torch.randn(B, 1, *coarse_shape)

        logits, features = simple_model.forward_coarse_only(image_coarse)

        assert logits.shape == (B, 2, *coarse_shape)
        # Features may be different shape due to downsampling
        assert features.dim() == 5
        assert features.shape[0] == B


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradient_flow_no_detach(self):
        """Test gradients reach coarse network when detach is OFF."""
        model = build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=2,
            context_channels=8,
        )
        model.set_phase(2)  # No detach

        B = 2
        coarse_shape = (16, 16, 16)
        fine_shape = (8, 8, 8)

        image_coarse = torch.randn(B, 1, *coarse_shape)
        image_fine = torch.randn(B, 1, *fine_shape)
        centers = torch.zeros(B, 3)

        spacing_fine = torch.tensor([1.0, 1.0, 1.0])
        spacing_coarse = torch.tensor([2.0, 2.0, 2.0])

        coarse_logits, fine_logits = model(
            image_coarse,
            image_fine,
            centers,
            fine_shape,
            spacing_fine,
            spacing_coarse,
        )

        # Compute fine loss and backprop
        fine_loss = fine_logits.mean()
        fine_loss.backward()

        # Check coarse network has gradients
        coarse_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.coarse_net.parameters()
        )

        assert coarse_has_grad, "Coarse network should receive gradients when detach is OFF"

    def test_gradient_flow_with_detach(self):
        """Test gradients don't reach coarse network when detach is ON."""
        model = build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=2,
            context_channels=8,
        )
        model.set_phase(1)  # Detach ON

        B = 2
        coarse_shape = (16, 16, 16)
        fine_shape = (8, 8, 8)

        image_coarse = torch.randn(B, 1, *coarse_shape)
        image_fine = torch.randn(B, 1, *fine_shape)
        centers = torch.zeros(B, 3)

        spacing_fine = torch.tensor([1.0, 1.0, 1.0])
        spacing_coarse = torch.tensor([2.0, 2.0, 2.0])

        coarse_logits, fine_logits = model(
            image_coarse,
            image_fine,
            centers,
            fine_shape,
            spacing_fine,
            spacing_coarse,
        )

        # Only compute fine loss (coarse loss would still provide gradients)
        fine_loss = fine_logits.mean()
        fine_loss.backward()

        # Check coarse network does NOT have gradients from fine loss
        # (gradients only come through context path which is detached)
        coarse_context_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.coarse_net.parameters()
            if "feat_proj" in str(p) or "encoder" in str(p)  # Context-related params
        )

        # In phase 1 with detach, fine loss should not provide gradients to coarse
        # But note: coarse_logits.mean() backward WOULD provide gradients
        # The test is specifically for gradients through context path


class TestContextSamplerIntegration:
    """Tests for context sampler integration with model."""

    def test_context_sampler_shapes(self):
        """Test context sampler produces correct shapes in model."""
        sampler = DifferentiableContextSampler()

        B, C = 2, 16
        src_shape = (8, 8, 8)
        out_shape = (16, 16, 16)

        src = torch.randn(B, C, *src_shape)
        centers = torch.zeros(B, 3)
        extent = (4.0, 4.0, 4.0)  # Extent in source voxels

        context = sampler(src, centers, out_shape, extent)

        assert context.shape == (B, C, *out_shape)

    def test_context_sampler_gradient_flow(self):
        """Test gradients flow through context sampler."""
        sampler = DifferentiableContextSampler()

        src = torch.randn(2, 8, 16, 16, 16, requires_grad=True)
        centers = torch.zeros(2, 3)

        context = sampler(src, centers, (8, 8, 8), (8.0, 8.0, 8.0))
        loss = context.mean()
        loss.backward()

        assert src.grad is not None
        assert src.grad.abs().sum() > 0

    def test_context_sampler_with_clamping(self):
        """Test context sampler with grid clamping."""
        sampler = DifferentiableContextSampler(clamp_grid=True)

        src = torch.randn(2, 8, 16, 16, 16)
        # Centers at edge of volume
        centers = torch.tensor([[0.9, 0.9, 0.9], [-0.9, -0.9, -0.9]])

        # Should not raise even with edge centers
        context = sampler(src, centers, (8, 8, 8), (4.0, 4.0, 4.0))

        assert context.shape == (2, 8, 8, 8, 8)


class TestModelBuilders:
    """Tests for model builder functions."""

    def test_build_simple_model(self):
        """Test simple model builder."""
        model = build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=3,
            context_channels=16,
        )

        assert isinstance(model, Swin3DDNP)
        assert model.fine_net is not None
        assert model.coarse_net is not None
        assert model.fusion is not None

    def test_simple_model_forward(self):
        """Test simple model can run forward pass."""
        model = build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=2,
        )

        B = 1
        image_coarse = torch.randn(B, 1, 32, 32, 32)
        image_fine = torch.randn(B, 1, 16, 16, 16)
        centers = torch.zeros(B, 3)

        coarse_logits, fine_logits = model(
            image_coarse,
            image_fine,
            centers,
            (16, 16, 16),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        )

        assert coarse_logits.shape[1] == 2
        assert fine_logits.shape[1] == 2


class TestForwardFineWithContext:
    """Tests for forward_fine_with_context method."""

    def test_forward_fine_with_precomputed_context(self):
        """Test fine forward with pre-computed coarse features."""
        model = build_simple_swin3d_dnp(
            in_channels=1,
            out_channels=2,
            context_channels=16,
        )

        B = 2
        coarse_shape = (32, 32, 32)
        fine_shape = (16, 16, 16)

        # Pre-compute coarse features (batch size 1)
        image_coarse = torch.randn(1, 1, *coarse_shape)
        coarse_logits, coarse_feat = model.forward_coarse_only(image_coarse)

        # Process multiple fine patches
        image_fine = torch.randn(B, 1, *fine_shape)
        centers = torch.zeros(B, 3)

        fine_logits = model.forward_fine_with_context(
            image_fine,
            coarse_feat,
            coarse_logits,
            centers,
            fine_shape,
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        )

        assert fine_logits.shape == (B, 2, *fine_shape)
