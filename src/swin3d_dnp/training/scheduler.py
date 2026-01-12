"""Phase scheduler for Swin3D-DNP training.

This module implements the training phase scheduling system that controls:
- Loss weighting (lambda0 for coarse, lambda1 for fine)
- Gradient flow (context detachment)
- Hard negative mining activation
- Learning rate scaling

Training phases:
- Phase 1 (0-10%): Coarse warmup, detach context, no hard negatives
- Phase 2 (10-60%): Transition, enable gradients, enable hard negatives
- Phase 3 (60-100%): End-to-end fine-tuning, lower LR
"""

import torch
import torch.nn as nn

from swin3d_dnp.constants import (
    HARDNEG_BATCH_PROB,
    HARDNEG_WARMUP_STEPS,
    LAMBDA0_PHASE1,
    LAMBDA0_PHASE2,
    LAMBDA0_PHASE3,
    LAMBDA1_PHASE1,
    LAMBDA1_PHASE2,
    LAMBDA1_PHASE3,
    PHASE1_END,
    PHASE2_END,
)


class PhaseScheduler:
    """Training phase scheduler for Swin3D-DNP.

    Controls training dynamics including loss weights, gradient flow,
    and hard negative mining based on training progress.

    Example:
        >>> scheduler = PhaseScheduler(total_steps=100000)
        >>> for step in range(100000):
        ...     phase_info = scheduler.step(step)
        ...     model.set_phase(phase_info["phase"])
        ...     loss = phase_info["lambda0"] * coarse_loss + phase_info["lambda1"] * fine_loss
        ...     if phase_info["use_hard_neg"]:
        ...         # Add hard negative samples
    """

    def __init__(
        self,
        total_steps: int,
        phase1_end: float = PHASE1_END,
        phase2_end: float = PHASE2_END,
        lambda0_phase1: float = LAMBDA0_PHASE1,
        lambda1_phase1: float = LAMBDA1_PHASE1,
        lambda0_phase2: float = LAMBDA0_PHASE2,
        lambda1_phase2: float = LAMBDA1_PHASE2,
        lambda0_phase3: float = LAMBDA0_PHASE3,
        lambda1_phase3: float = LAMBDA1_PHASE3,
        hardneg_warmup_steps: int = HARDNEG_WARMUP_STEPS,
        hardneg_batch_prob: float = HARDNEG_BATCH_PROB,
        lr_scale_phase3: float = 0.1,
    ):
        """Initialize phase scheduler.

        Args:
            total_steps: Total number of training steps.
            phase1_end: Fraction of total steps for phase 1 end (default: 0.10).
            phase2_end: Fraction of total steps for phase 2 end (default: 0.60).
            lambda0_phase1: Coarse loss weight in phase 1.
            lambda1_phase1: Fine loss weight in phase 1.
            lambda0_phase2: Coarse loss weight in phase 2.
            lambda1_phase2: Fine loss weight in phase 2.
            lambda0_phase3: Coarse loss weight in phase 3.
            lambda1_phase3: Fine loss weight in phase 3.
            hardneg_warmup_steps: Steps before enabling hard negative mining.
            hardneg_batch_prob: Probability of hard negative batch after warmup.
            lr_scale_phase3: LR multiplier for phase 3.
        """
        self.total_steps = total_steps
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end

        # Loss weights per phase
        self._lambda0 = {1: lambda0_phase1, 2: lambda0_phase2, 3: lambda0_phase3}
        self._lambda1 = {1: lambda1_phase1, 2: lambda1_phase2, 3: lambda1_phase3}

        # Hard negative parameters
        self.hardneg_warmup_steps = hardneg_warmup_steps
        self.hardneg_batch_prob = hardneg_batch_prob

        # LR scaling
        self.lr_scale_phase3 = lr_scale_phase3

        # Phase boundaries in steps
        self._phase1_end_step = int(total_steps * phase1_end)
        self._phase2_end_step = int(total_steps * phase2_end)

        # Current state
        self._current_phase = 1
        self._current_step = 0

    def get_phase(self, step: int) -> int:
        """Get training phase for given step.

        Args:
            step: Current training step.

        Returns:
            Phase number (1, 2, or 3).
        """
        if step < self._phase1_end_step:
            return 1
        elif step < self._phase2_end_step:
            return 2
        else:
            return 3

    def get_lambdas(self, step: int) -> tuple[float, float]:
        """Get loss weights for given step.

        Args:
            step: Current training step.

        Returns:
            Tuple of (lambda0, lambda1) loss weights.
        """
        phase = self.get_phase(step)
        return self._lambda0[phase], self._lambda1[phase]

    def get_lambda0(self, step: int) -> float:
        """Get coarse loss weight."""
        return self.get_lambdas(step)[0]

    def get_lambda1(self, step: int) -> float:
        """Get fine loss weight."""
        return self.get_lambdas(step)[1]

    def should_use_hard_negatives(self, step: int) -> bool:
        """Check if hard negative mining should be used.

        Hard negatives are enabled when:
        1. We're past phase 1
        2. We've completed the hard negative warmup steps

        Args:
            step: Current training step.

        Returns:
            True if hard negatives should be sampled.
        """
        phase = self.get_phase(step)
        return phase > 1 and step >= self.hardneg_warmup_steps

    def sample_hard_negative_batch(self, step: int) -> bool:
        """Determine if this specific batch should include hard negatives.

        Uses probabilistic sampling based on hardneg_batch_prob.

        Args:
            step: Current training step.

        Returns:
            True if this batch should include hard negatives.
        """
        if not self.should_use_hard_negatives(step):
            return False

        return torch.rand(1).item() < self.hardneg_batch_prob

    def should_detach_context(self, step: int) -> bool:
        """Check if coarse context should be detached.

        Context is detached in phase 1 for memory efficiency and
        to focus learning on the coarse network.

        Args:
            step: Current training step.

        Returns:
            True if context should be detached.
        """
        return self.get_phase(step) == 1

    def get_lr_scale(self, step: int) -> float:
        """Get learning rate scale factor.

        LR is scaled down in phase 3 for fine-tuning.

        Args:
            step: Current training step.

        Returns:
            LR multiplier (1.0 for phases 1-2, lr_scale_phase3 for phase 3).
        """
        phase = self.get_phase(step)
        return self.lr_scale_phase3 if phase == 3 else 1.0

    def step(self, current_step: int) -> dict:
        """Get all phase information for current step.

        Args:
            current_step: Current training step.

        Returns:
            Dictionary with:
            - phase: Current phase (1, 2, 3)
            - lambda0: Coarse loss weight
            - lambda1: Fine loss weight
            - detach_context: Whether to detach coarse context
            - use_hard_neg: Whether to use hard negatives this batch
            - lr_scale: Learning rate multiplier
            - progress: Training progress (0 to 1)
        """
        self._current_step = current_step
        self._current_phase = self.get_phase(current_step)

        lambda0, lambda1 = self.get_lambdas(current_step)

        return {
            "phase": self._current_phase,
            "lambda0": lambda0,
            "lambda1": lambda1,
            "detach_context": self.should_detach_context(current_step),
            "use_hard_neg": self.sample_hard_negative_batch(current_step),
            "lr_scale": self.get_lr_scale(current_step),
            "progress": current_step / self.total_steps,
        }

    def apply_to_model(self, model: nn.Module, step: int) -> None:
        """Apply phase settings to model.

        Sets the model's phase via set_phase() method.

        Args:
            model: Model with set_phase() method.
            step: Current training step.
        """
        phase = self.get_phase(step)
        if hasattr(model, "set_phase"):
            model.set_phase(phase)

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "total_steps": self.total_steps,
            "phase1_end": self.phase1_end,
            "phase2_end": self.phase2_end,
            "hardneg_warmup_steps": self.hardneg_warmup_steps,
            "hardneg_batch_prob": self.hardneg_batch_prob,
            "lr_scale_phase3": self.lr_scale_phase3,
            "current_step": self._current_step,
            "current_phase": self._current_phase,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.total_steps = state_dict["total_steps"]
        self.phase1_end = state_dict["phase1_end"]
        self.phase2_end = state_dict["phase2_end"]
        self.hardneg_warmup_steps = state_dict["hardneg_warmup_steps"]
        self.hardneg_batch_prob = state_dict["hardneg_batch_prob"]
        self.lr_scale_phase3 = state_dict["lr_scale_phase3"]
        self._current_step = state_dict["current_step"]
        self._current_phase = state_dict["current_phase"]

        # Recompute boundaries
        self._phase1_end_step = int(self.total_steps * self.phase1_end)
        self._phase2_end_step = int(self.total_steps * self.phase2_end)

    def __repr__(self) -> str:
        return (
            f"PhaseScheduler("
            f"total_steps={self.total_steps}, "
            f"phase1_end={self.phase1_end}, "
            f"phase2_end={self.phase2_end})"
        )


class LRSchedulerWrapper:
    """Wrapper that applies phase-based LR scaling to any scheduler.

    Combines a base LR scheduler with phase-based scaling from PhaseScheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: torch.optim.lr_scheduler.LRScheduler,
        phase_scheduler: PhaseScheduler,
    ):
        """Initialize LR scheduler wrapper.

        Args:
            optimizer: Optimizer instance.
            base_scheduler: Base LR scheduler (e.g., CosineAnnealingLR).
            phase_scheduler: Phase scheduler for phase-based scaling.
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.phase_scheduler = phase_scheduler
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, current_step: int) -> None:
        """Update learning rate.

        Args:
            current_step: Current training step.
        """
        # Step base scheduler
        self.base_scheduler.step()

        # Apply phase scaling
        lr_scale = self.phase_scheduler.get_lr_scale(current_step)
        if lr_scale != 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * lr_scale

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "base_lrs": self._base_lrs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self._base_lrs = state_dict["base_lrs"]


def create_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a cosine schedule with linear warmup.

    Args:
        optimizer: Optimizer instance.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of initial LR.

    Returns:
        LR scheduler with warmup and cosine decay.
    """
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return current_step / max(1, num_warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - num_warmup_steps) / max(
                1, num_training_steps - num_warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
