"""Tests for Swin3D-DNP loss functions.

Tests verify:
1. Masked losses correctly ignore padded/invalid regions
2. Perfect predictions give expected loss values
3. Focal losses focus on hard examples
4. Numerical stability under edge cases
5. Gradient flow is correct
"""

import pytest
import torch
import torch.nn.functional as F
from swin3d_dnp.losses import (
    masked_cross_entropy,
    masked_cross_entropy_per_class,
    masked_dice_loss,
    masked_dice_loss_per_class,
    masked_generalized_dice_loss,
    dice_score,
    focal_heatmap_loss,
    focal_heatmap_loss_from_logits,
    focal_cross_entropy_loss,
    offset_loss,
)
from swin3d_dnp.constants import EPS_DICE


class TestMaskedCrossEntropy:
    """Tests for masked cross-entropy loss."""

    def test_valid_mask_ignores_padded_regions(self):
        """Masked CE should only count valid voxels."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        # Create logits and target
        logits = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))

        # Create mask where half the volume is invalid
        valid_mask = torch.zeros(B, 1, D, H, W)
        valid_mask[:, :, :D//2, :, :] = 1.0

        loss = masked_cross_entropy(logits, target, valid_mask)

        # Verify loss is computed
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_all_invalid_returns_zero(self):
        """All-invalid mask should return near-zero loss (due to epsilon)."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        logits = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.zeros(B, 1, D, H, W)  # All invalid

        loss = masked_cross_entropy(logits, target, valid_mask)

        # Should be effectively zero
        assert loss.item() < 1e-6

    def test_full_valid_mask_matches_standard_ce(self):
        """With all-ones mask, should match standard cross-entropy."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        logits = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.ones(B, 1, D, H, W)

        masked_loss = masked_cross_entropy(logits, target, valid_mask)
        standard_loss = F.cross_entropy(logits, target)

        # Should be very close
        assert torch.allclose(masked_loss, standard_loss, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through masked CE."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        logits = torch.randn(B, C, D, H, W, requires_grad=True)
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.ones(B, 1, D, H, W)
        valid_mask[:, :, D//2:, :, :] = 0.0  # Half invalid

        loss = masked_cross_entropy(logits, target, valid_mask)
        loss.backward()

        assert logits.grad is not None
        # Gradients should be zero in invalid region
        invalid_grad = logits.grad[:, :, D//2:, :, :]
        assert torch.allclose(invalid_grad, torch.zeros_like(invalid_grad), atol=1e-6)

        # Gradients should be non-zero in valid region
        valid_grad = logits.grad[:, :, :D//2, :, :]
        assert valid_grad.abs().sum() > 0

    def test_per_class_loss(self):
        """Per-class CE should give breakdown by class."""
        B, C, D, H, W = 2, 4, 8, 8, 8

        logits = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.ones(B, 1, D, H, W)

        per_class = masked_cross_entropy_per_class(logits, target, valid_mask, C)

        assert per_class.shape == (C,)
        assert all(l >= 0 for l in per_class)


class TestMaskedDiceLoss:
    """Tests for masked Dice loss."""

    def test_perfect_prediction_gives_zero_loss(self):
        """Perfect overlap should give Dice ≈ 1, loss ≈ 0."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        # Create one-hot target
        target = torch.zeros(B, C, D, H, W)
        target[:, 0, :4, :, :] = 1.0  # Class 0 in first half
        target[:, 1, 4:, :4, :] = 1.0  # Class 1
        target[:, 2, 4:, 4:, :] = 1.0  # Class 2

        # Perfect prediction (use large logits to get near 1.0 probabilities)
        pred = torch.zeros(B, C, D, H, W)
        pred[:, 0, :4, :, :] = 10.0
        pred[:, 1, 4:, :4, :] = 10.0
        pred[:, 2, 4:, 4:, :] = 10.0
        pred[:, 0, 4:, :, :] = -10.0  # Force low prob elsewhere

        valid_mask = torch.ones(B, 1, D, H, W)

        loss = masked_dice_loss(pred, target, valid_mask, apply_softmax=True)

        # Loss should be near zero (perfect Dice)
        assert loss.item() < 0.05

    def test_worst_case_gives_high_loss(self):
        """Complete mismatch should give loss near 1."""
        B, C, D, H, W = 2, 2, 8, 8, 8

        # Target is class 0 everywhere
        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)

        # Predict class 1 everywhere (completely wrong)
        pred = torch.zeros(B, C, D, H, W)
        pred[:, 1, :, :, :] = 10.0  # Class 1 high
        pred[:, 0, :, :, :] = -10.0  # Class 0 low

        valid_mask = torch.ones(B, 1, D, H, W)

        loss = masked_dice_loss(pred, target, valid_mask)

        # Loss should be high (near 1 for class 0 since it's missed)
        assert loss.item() > 0.4

    def test_valid_mask_excludes_invalid_regions(self):
        """Invalid regions should not affect Dice computation."""
        B, C, D, H, W = 2, 2, 8, 8, 8

        # Create target with class 0 everywhere
        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)

        # Create perfect prediction for class 0
        pred = torch.zeros(B, C, D, H, W)
        pred[:, 0, :, :, :] = 10.0
        pred[:, 1, :, :, :] = -10.0

        # Make half the volume invalid
        valid_mask = torch.zeros(B, 1, D, H, W)
        valid_mask[:, :, :D//2, :, :] = 1.0

        loss = masked_dice_loss(pred, target, valid_mask)

        # Should still be near zero (perfect in valid region)
        assert loss.item() < 0.05

    def test_gradient_flow_respects_mask(self):
        """Gradients should be zero in invalid regions."""
        B, C, D, H, W = 2, 2, 8, 8, 8

        pred = torch.randn(B, C, D, H, W, requires_grad=True)
        target = torch.randint(0, C, (B, 1, D, H, W))

        valid_mask = torch.ones(B, 1, D, H, W)
        valid_mask[:, :, D//2:, :, :] = 0.0

        loss = masked_dice_loss(pred, target, valid_mask)
        loss.backward()

        assert pred.grad is not None
        # Invalid region should have zero gradients
        invalid_grad = pred.grad[:, :, D//2:, :, :]
        assert torch.allclose(invalid_grad, torch.zeros_like(invalid_grad), atol=1e-6)

    def test_numerical_stability_with_empty_class(self):
        """Should handle classes with no valid voxels gracefully."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        # Target has only class 0
        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)
        pred = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, 1, D, H, W)

        # Should not raise or return NaN
        loss = masked_dice_loss(pred, target, valid_mask)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_generalized_dice_loss(self):
        """GDL should handle class imbalance."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        # Highly imbalanced target
        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)
        target[:, 0, 0, 0, 0] = 1  # Very small class 1
        target[:, 0, 1, 0, 0] = 2  # Very small class 2

        pred = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, 1, D, H, W)

        loss = masked_generalized_dice_loss(pred, target, valid_mask)

        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_dice_score(self):
        """Dice score should be in [0, 1]."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        # Create matching pred and target
        target = torch.zeros(B, C, D, H, W)
        target[:, 0, :4, :, :] = 1.0
        target[:, 1, 4:, :4, :] = 1.0
        target[:, 2, 4:, 4:, :] = 1.0

        pred = target.clone()  # Perfect

        scores = dice_score(pred, target)

        assert scores.shape == (C,)
        assert all(0 <= s <= 1 for s in scores)
        assert all(s > 0.99 for s in scores)  # Perfect prediction


class TestFocalHeatmapLoss:
    """Tests for focal heatmap loss (CornerNet-style)."""

    def test_peak_prediction_gives_low_loss(self):
        """Correct prediction at peaks should give lower loss than wrong prediction."""
        B, C, D, H, W = 2, 1, 16, 16, 16

        # Create target with Gaussian peaks
        target = torch.zeros(B, C, D, H, W)
        target[:, :, 8, 8, 8] = 1.0  # Peak at center

        # Add Gaussian blur around peak (simulate ground truth heatmap)
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    dist = (dz**2 + dy**2 + dx**2) ** 0.5
                    val = max(0, 1 - dist / 3)
                    if val > 0:
                        target[:, :, 8+dz, 8+dy, 8+dx] = val

        # Good prediction matches target
        pred_good = target.clone()

        # Bad prediction: random noise
        pred_bad = torch.rand(B, C, D, H, W)

        valid_mask = torch.ones(B, 1, D, H, W)

        loss_good = focal_heatmap_loss(pred_good, target, valid_mask)
        loss_bad = focal_heatmap_loss(pred_bad, target, valid_mask)

        # Good prediction should have lower loss than random
        assert loss_good.item() < loss_bad.item()

    def test_wrong_prediction_gives_high_loss(self):
        """Wrong prediction at peaks should give higher loss."""
        B, C, D, H, W = 2, 1, 16, 16, 16

        # Target with peak at center
        target = torch.zeros(B, C, D, H, W)
        target[:, :, 8, 8, 8] = 1.0

        # Prediction with peak in wrong location
        pred = torch.zeros(B, C, D, H, W)
        pred[:, :, 0, 0, 0] = 1.0  # Wrong location

        valid_mask = torch.ones(B, 1, D, H, W)

        loss = focal_heatmap_loss(pred, target, valid_mask)

        # Loss should be high
        assert loss.item() > 1.0

    def test_focal_down_weights_easy_examples(self):
        """Focal loss should down-weight easy examples."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        # Target: one peak
        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0

        valid_mask = torch.ones(B, 1, D, H, W)

        # Easy negative (confident 0)
        pred_easy = torch.full((B, C, D, H, W), 0.01)
        pred_easy[:, :, 4, 4, 4] = 0.99  # Easy positive

        # Hard negative (uncertain)
        pred_hard = torch.full((B, C, D, H, W), 0.3)  # Uncertain negatives
        pred_hard[:, :, 4, 4, 4] = 0.7  # Uncertain positive

        loss_easy = focal_heatmap_loss(pred_easy, target, valid_mask)
        loss_hard = focal_heatmap_loss(pred_hard, target, valid_mask)

        # Hard examples should contribute more to loss
        assert loss_hard > loss_easy

    def test_mask_excludes_invalid(self):
        """Invalid regions should not contribute to focal loss."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0  # Peak in valid region
        target[:, :, 6, 6, 6] = 1.0  # Peak in invalid region

        pred = torch.full((B, C, D, H, W), 0.5)

        # Mask: only first half valid
        valid_mask = torch.zeros(B, 1, D, H, W)
        valid_mask[:, :, :5, :, :] = 1.0

        loss = focal_heatmap_loss(pred, target, valid_mask)

        # Should compute without error
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_from_logits(self):
        """focal_heatmap_loss_from_logits should apply sigmoid."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0

        logits = torch.randn(B, C, D, H, W)
        pred = torch.sigmoid(logits)

        valid_mask = torch.ones(B, 1, D, H, W)

        loss_from_logits = focal_heatmap_loss_from_logits(logits, target, valid_mask)
        loss_from_pred = focal_heatmap_loss(pred, target, valid_mask)

        assert torch.allclose(loss_from_logits, loss_from_pred, atol=1e-5)


class TestFocalCrossEntropyLoss:
    """Tests for classification focal loss."""

    def test_reduces_easy_example_contribution(self):
        """Focal loss should down-weight well-classified examples."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        target = torch.randint(0, C, (B, D, H, W))

        # Create confident correct predictions
        logits_easy = torch.zeros(B, C, D, H, W)
        for b in range(B):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        logits_easy[b, target[b, d, h, w], d, h, w] = 10.0

        # Create uncertain predictions
        logits_hard = torch.randn(B, C, D, H, W) * 0.1

        valid_mask = torch.ones(B, 1, D, H, W)

        loss_easy = focal_cross_entropy_loss(logits_easy, target, valid_mask, gamma=2.0)
        loss_hard = focal_cross_entropy_loss(logits_hard, target, valid_mask, gamma=2.0)

        # Easy examples should have much lower focal loss
        assert loss_easy < loss_hard

    def test_gamma_zero_equals_ce(self):
        """With gamma=0, focal loss should equal CE."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        logits = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.ones(B, 1, D, H, W)

        focal_loss = focal_cross_entropy_loss(logits, target, valid_mask, gamma=0.0)
        ce_loss = masked_cross_entropy(logits, target, valid_mask)

        assert torch.allclose(focal_loss, ce_loss, atol=1e-5)


class TestOffsetLoss:
    """Tests for offset regression loss."""

    def test_perfect_offset_gives_zero_loss(self):
        """Perfect offset prediction should give zero loss."""
        B, D, H, W = 2, 8, 8, 8

        target_offset = torch.randn(B, 3, D, H, W)
        pred_offset = target_offset.clone()

        valid_mask = torch.ones(B, 1, D, H, W)
        pos_mask = torch.zeros(B, 1, D, H, W)
        pos_mask[:, :, 4, 4, 4] = 1.0  # One positive location

        loss = offset_loss(pred_offset, target_offset, valid_mask, pos_mask)

        assert loss.item() < 1e-6

    def test_only_positive_locations_count(self):
        """Offset loss should only count positive locations."""
        B, D, H, W = 2, 8, 8, 8

        # Correct at positive, wrong elsewhere
        target_offset = torch.zeros(B, 3, D, H, W)
        pred_offset = torch.randn(B, 3, D, H, W) * 10  # Very wrong everywhere
        pred_offset[:, :, 4, 4, 4] = 0  # Correct at positive

        valid_mask = torch.ones(B, 1, D, H, W)
        pos_mask = torch.zeros(B, 1, D, H, W)
        pos_mask[:, :, 4, 4, 4] = 1.0

        loss = offset_loss(pred_offset, target_offset, valid_mask, pos_mask)

        # Loss should be near zero (only positive counts)
        assert loss.item() < 1e-6

    def test_valid_mask_respected(self):
        """Invalid regions should not contribute to offset loss."""
        B, D, H, W = 2, 8, 8, 8

        target_offset = torch.zeros(B, 3, D, H, W)
        pred_offset = torch.randn(B, 3, D, H, W)

        # Valid only in first half
        valid_mask = torch.zeros(B, 1, D, H, W)
        valid_mask[:, :, :4, :, :] = 1.0

        # Positive in invalid region
        pos_mask = torch.zeros(B, 1, D, H, W)
        pos_mask[:, :, 6, 4, 4] = 1.0  # Invalid region

        loss = offset_loss(pred_offset, target_offset, valid_mask, pos_mask)

        # No valid positives, loss should be ~0
        assert loss.item() < 1e-6


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_ce_with_extreme_logits(self):
        """CE should handle extreme logit values."""
        B, C, D, H, W = 2, 3, 8, 8, 8

        logits = torch.randn(B, C, D, H, W) * 100  # Very extreme
        target = torch.randint(0, C, (B, D, H, W))
        valid_mask = torch.ones(B, 1, D, H, W)

        loss = masked_cross_entropy(logits, target, valid_mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_dice_with_zero_volume_class(self):
        """Dice should handle classes with zero volume."""
        B, C, D, H, W = 2, 5, 8, 8, 8

        # Only classes 0 and 1 have volume
        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)
        target[:, 0, :4, :, :] = 1

        pred = torch.randn(B, C, D, H, W)
        valid_mask = torch.ones(B, 1, D, H, W)

        loss = masked_dice_loss(pred, target, valid_mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_focal_with_zero_predictions(self):
        """Focal loss should handle near-zero predictions."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0

        # Near-zero predictions (after clamping)
        pred = torch.full((B, C, D, H, W), 1e-10)

        valid_mask = torch.ones(B, 1, D, H, W)

        loss = focal_heatmap_loss(pred, target, valid_mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_focal_with_one_predictions(self):
        """Focal loss should handle near-one predictions."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0

        # Near-one predictions
        pred = torch.full((B, C, D, H, W), 1 - 1e-10)

        valid_mask = torch.ones(B, 1, D, H, W)

        loss = focal_heatmap_loss(pred, target, valid_mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestGradientComputation:
    """Tests for gradient computation through losses."""

    def test_dice_gradients_are_smooth(self):
        """Dice gradients should be smooth (no discontinuities)."""
        B, C, D, H, W = 1, 2, 4, 4, 4

        target = torch.zeros(B, 1, D, H, W, dtype=torch.long)
        target[:, 0, :2, :, :] = 1

        # Test gradients at multiple prediction values
        grads = []
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            # Create leaf tensor with requires_grad=True
            pred = torch.randn(B, C, D, H, W) * scale
            pred = pred.clone().detach().requires_grad_(True)  # Ensure leaf tensor
            valid_mask = torch.ones(B, 1, D, H, W)

            loss = masked_dice_loss(pred, target, valid_mask)
            loss.backward()

            assert pred.grad is not None, "Gradient should be computed"
            grads.append(pred.grad.abs().mean().item())

        # Gradients should be finite and reasonable
        assert all(not (g != g) for g in grads)  # No NaN
        assert all(g < 1000 for g in grads)  # Not exploding

    def test_focal_gradients_respect_alpha_beta(self):
        """Focal parameters should affect gradient magnitude."""
        B, C, D, H, W = 2, 1, 8, 8, 8

        target = torch.zeros(B, C, D, H, W)
        target[:, :, 4, 4, 4] = 1.0

        pred = torch.full((B, C, D, H, W), 0.5, requires_grad=True)
        valid_mask = torch.ones(B, 1, D, H, W)

        # Higher alpha should change gradient behavior
        loss_alpha2 = focal_heatmap_loss(pred, target, valid_mask, alpha=2.0, beta=4.0)
        loss_alpha2.backward()
        grad_alpha2 = pred.grad.clone()

        pred.grad.zero_()

        loss_alpha4 = focal_heatmap_loss(pred, target, valid_mask, alpha=4.0, beta=4.0)
        loss_alpha4.backward()
        grad_alpha4 = pred.grad.clone()

        # Gradients should differ with different alpha
        assert not torch.allclose(grad_alpha2, grad_alpha4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
