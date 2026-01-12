"""Training loop for Swin3D-DNP.

This module implements the main training pipeline with:
- Mixed precision (AMP) support
- Gradient accumulation
- Phase-based scheduling
- Checkpointing
- Logging and metrics
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from swin3d_dnp.geometry.mapping import center_full_to_coarse_norm
from swin3d_dnp.geometry.sampling import sample_patch_from_full
from swin3d_dnp.losses import masked_cross_entropy, masked_dice_loss
from swin3d_dnp.training.scheduler import PhaseScheduler
from swin3d_dnp.training.utils import (
    clip_grad_norm_,
    get_grad_scaler,
    move_batch_to_device,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the Trainer.

    Attributes:
        total_steps: Total number of training steps.
        batch_size: Batch size per GPU.
        accumulation_steps: Gradient accumulation steps.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: AMP dtype ("bfloat16" or "float16").
        max_grad_norm: Maximum gradient norm for clipping.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_every: Save checkpoint every N steps.
        log_every: Log metrics every N steps.
        fine_shape: (Df, Hf, Wf) fine patch shape.
        coarse_shape: (Dc, Hc, Wc) coarse volume shape.
        device: Training device.
    """

    total_steps: int = 100000
    batch_size: int = 1
    accumulation_steps: int = 1
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5000
    log_every: int = 100
    fine_shape: tuple[int, int, int] = (96, 96, 96)
    coarse_shape: tuple[int, int, int] = (128, 128, 128)
    device: str = "cuda"
    # Loss weights
    dice_weight: float = 1.0
    ce_weight: float = 1.0


@dataclass
class TrainerState:
    """Mutable training state.

    Attributes:
        step: Current training step.
        epoch: Current epoch.
        best_loss: Best validation loss seen.
        metrics_history: List of logged metrics.
    """

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    metrics_history: list[dict] = field(default_factory=list)


class Trainer:
    """Main training class for Swin3D-DNP.

    Handles the complete training pipeline including:
    - Forward pass through coarse and fine networks
    - Loss computation with masking
    - Backward pass with gradient accumulation
    - Phase scheduling
    - Checkpointing and logging

    Example:
        >>> model = build_swin3d_dnp(...)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trainer = Trainer(model, optimizer, train_loader, config)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig,
        val_loader: DataLoader | None = None,
        lr_scheduler: Any | None = None,
        phase_scheduler: PhaseScheduler | None = None,
        loss_fn: Callable | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Swin3DDNP model.
            optimizer: Optimizer instance.
            train_loader: Training data loader.
            config: Trainer configuration.
            val_loader: Optional validation loader.
            lr_scheduler: Optional learning rate scheduler.
            phase_scheduler: Optional phase scheduler (created from config if None).
            loss_fn: Optional custom loss function.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

        # Create phase scheduler if not provided
        self.phase_scheduler = phase_scheduler or PhaseScheduler(
            total_steps=config.total_steps
        )

        # Initialize state
        self.state = TrainerState()

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup AMP
        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )
        self.scaler = get_grad_scaler(enabled=config.use_amp and config.amp_dtype != "bfloat16")

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self._step_metrics: dict[str, float] = {}
        self._accumulated_loss = 0.0
        self._accumulated_count = 0

    def _compute_loss(
        self,
        coarse_logits: Tensor,
        fine_logits: Tensor,
        label_coarse: Tensor,
        label_fine: Tensor,
        valid_mask: Tensor,
        phase_info: dict,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined coarse and fine losses.

        Args:
            coarse_logits: (B, C, Dc, Hc, Wc) coarse predictions.
            fine_logits: (B, C, Df, Hf, Wf) fine predictions.
            label_coarse: (B, Dc, Hc, Wc) coarse labels.
            label_fine: (B, Df, Hf, Wf) fine labels.
            valid_mask: (B, 1, Df, Hf, Wf) fine patch valid mask.
            phase_info: Phase information from scheduler.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        lambda0 = phase_info["lambda0"]
        lambda1 = phase_info["lambda1"]

        metrics = {}

        # Coarse loss (no masking needed, full volume is valid)
        coarse_ce = masked_cross_entropy(
            coarse_logits,
            label_coarse,
            valid_mask=None,
        )
        coarse_dice = masked_dice_loss(
            torch.sigmoid(coarse_logits),
            (label_coarse > 0).float().unsqueeze(1),
            valid_mask=None,
        )
        loss_coarse = (
            self.config.ce_weight * coarse_ce + self.config.dice_weight * coarse_dice
        )

        metrics["loss_coarse"] = loss_coarse.item()
        metrics["loss_coarse_ce"] = coarse_ce.item()
        metrics["loss_coarse_dice"] = coarse_dice.item()

        # Fine loss (with valid mask for out-of-bounds regions)
        fine_ce = masked_cross_entropy(
            fine_logits,
            label_fine,
            valid_mask=valid_mask,
        )
        fine_dice = masked_dice_loss(
            torch.sigmoid(fine_logits),
            (label_fine > 0).float().unsqueeze(1),
            valid_mask=valid_mask,
        )
        loss_fine = self.config.ce_weight * fine_ce + self.config.dice_weight * fine_dice

        metrics["loss_fine"] = loss_fine.item()
        metrics["loss_fine_ce"] = fine_ce.item()
        metrics["loss_fine_dice"] = fine_dice.item()

        # Combined loss with phase weights
        total_loss = lambda0 * loss_coarse + lambda1 * loss_fine

        metrics["loss_total"] = total_loss.item()
        metrics["lambda0"] = lambda0
        metrics["lambda1"] = lambda1
        metrics["phase"] = phase_info["phase"]

        return total_loss, metrics

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batch dictionary from data loader.

        Returns:
            Dictionary of step metrics.
        """
        # Get phase info
        phase_info = self.phase_scheduler.step(self.state.step)
        self.phase_scheduler.apply_to_model(self.model, self.state.step)

        # Move batch to device
        batch = move_batch_to_device(batch, self.device)

        # Extract batch data
        image_full = batch["image_full"]  # (B, 1, D, H, W)
        label_full = batch["label_full"]  # (B, D, H, W)
        affine_full = batch["affine_full"]  # (B, 4, 4)
        spacing_full = batch["spacing_full_dhw_mm"]  # (B, 3)
        image_coarse = batch["image_coarse"]  # (B, 1, Dc, Hc, Wc)
        label_coarse = batch["label_coarse"]  # (B, Dc, Hc, Wc)
        affine_coarse = batch["affine_coarse"]  # (B, 4, 4)
        spacing_coarse = batch["spacing_coarse_dhw_mm"]  # (B, 3)
        center_full_zyx = batch["center_full_index_zyx"]  # (B, 3)

        B = image_full.shape[0]

        # Sample fine patches from full volume
        image_fine, label_fine, valid_mask = sample_patch_from_full(
            image_full=image_full,
            label_full=label_full.unsqueeze(1).float(),
            affine_full=affine_full,
            center_full_index_zyx=center_full_zyx,
            out_shape=self.config.fine_shape,
            spacing_fine_dhw_mm=spacing_full,
        )
        label_fine = label_fine.squeeze(1).long() if label_fine is not None else None

        # Compute normalized centers in coarse space
        centers_coarse_norm = center_full_to_coarse_norm(
            center_full_zyx,
            affine_full,
            affine_coarse,
            tuple(image_full.shape[2:]),
            tuple(image_coarse.shape[2:]),
        )

        # Forward pass with AMP
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.config.use_amp,
        ):
            coarse_logits, fine_logits = self.model(
                image_coarse=image_coarse,
                image_fine=image_fine,
                centers_coarse_norm_dhw=centers_coarse_norm,
                fine_shape=self.config.fine_shape,
                spacing_fine_dhw_mm=spacing_full,
                spacing_coarse_dhw_mm=spacing_coarse,
            )

            # Compute loss
            if self.loss_fn is not None:
                loss, metrics = self.loss_fn(
                    coarse_logits,
                    fine_logits,
                    label_coarse,
                    label_fine,
                    valid_mask,
                    phase_info,
                )
            else:
                loss, metrics = self._compute_loss(
                    coarse_logits,
                    fine_logits,
                    label_coarse,
                    label_fine,
                    valid_mask,
                    phase_info,
                )

            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps

        # Backward pass
        if self.config.use_amp and self.config.amp_dtype != "bfloat16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Track accumulated loss
        self._accumulated_loss += loss.item() * self.config.accumulation_steps
        self._accumulated_count += 1

        # Optimizer step (every accumulation_steps)
        if (self.state.step + 1) % self.config.accumulation_steps == 0:
            if self.config.use_amp and self.config.amp_dtype != "bfloat16":
                self.scaler.unscale_(self.optimizer)

            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics["grad_norm"] = grad_norm.item()

            if self.config.use_amp and self.config.amp_dtype != "bfloat16":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.state.step)

        # Add LR to metrics
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return metrics

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            metrics = self._train_step(batch)
            epoch_loss += metrics.get("loss_total", 0.0)
            n_batches += 1

            # Logging
            if self.state.step % self.config.log_every == 0:
                self._log_metrics(metrics)

            # Checkpointing
            if self.state.step % self.config.checkpoint_every == 0:
                self.save_checkpoint()

            self.state.step += 1

            if self.state.step >= self.config.total_steps:
                break

        self.state.epoch += 1
        return epoch_loss / max(n_batches, 1)

    def train(self) -> None:
        """Run full training loop."""
        logger.info(f"Starting training for {self.config.total_steps} steps")

        start_time = time.time()

        while self.state.step < self.config.total_steps:
            epoch_loss = self.train_epoch()
            logger.info(
                f"Epoch {self.state.epoch} complete, "
                f"step {self.state.step}/{self.config.total_steps}, "
                f"loss: {epoch_loss:.4f}"
            )

            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                logger.info(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.state.best_loss:
                    self.state.best_loss = val_loss
                    self.save_checkpoint(filename="best.pt")

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 3600:.2f} hours")

        # Final checkpoint
        self.save_checkpoint(filename="final.pt")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        phase_info = self.phase_scheduler.step(self.state.step)

        for batch in self.val_loader:
            batch = move_batch_to_device(batch, self.device)

            # Similar processing as train_step but without gradients
            image_full = batch["image_full"]
            label_full = batch["label_full"]
            affine_full = batch["affine_full"]
            spacing_full = batch["spacing_full_dhw_mm"]
            image_coarse = batch["image_coarse"]
            label_coarse = batch["label_coarse"]
            affine_coarse = batch["affine_coarse"]
            spacing_coarse = batch["spacing_coarse_dhw_mm"]
            center_full_zyx = batch["center_full_index_zyx"]

            image_fine, label_fine, valid_mask = sample_patch_from_full(
                image_full=image_full,
                label_full=label_full.unsqueeze(1).float(),
                affine_full=affine_full,
                center_full_index_zyx=center_full_zyx,
                out_shape=self.config.fine_shape,
                spacing_fine_dhw_mm=spacing_full,
            )
            label_fine = label_fine.squeeze(1).long() if label_fine is not None else None

            centers_coarse_norm = center_full_to_coarse_norm(
                center_full_zyx,
                affine_full,
                affine_coarse,
                tuple(image_full.shape[2:]),
                tuple(image_coarse.shape[2:]),
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.config.use_amp,
            ):
                coarse_logits, fine_logits = self.model(
                    image_coarse=image_coarse,
                    image_fine=image_fine,
                    centers_coarse_norm_dhw=centers_coarse_norm,
                    fine_shape=self.config.fine_shape,
                    spacing_fine_dhw_mm=spacing_full,
                    spacing_coarse_dhw_mm=spacing_coarse,
                )

                loss, _ = self._compute_loss(
                    coarse_logits,
                    fine_logits,
                    label_coarse,
                    label_fine,
                    valid_mask,
                    phase_info,
                )

            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log training metrics."""
        avg_loss = self._accumulated_loss / max(self._accumulated_count, 1)

        log_msg = (
            f"Step {self.state.step}: "
            f"loss={avg_loss:.4f}, "
            f"phase={metrics.get('phase', 1)}, "
            f"lr={metrics.get('lr', 0):.2e}"
        )

        if "grad_norm" in metrics:
            log_msg += f", grad_norm={metrics['grad_norm']:.2f}"

        logger.info(log_msg)

        # Store in history
        self.state.metrics_history.append({
            "step": self.state.step,
            "loss": avg_loss,
            **metrics,
        })

        # Reset accumulators
        self._accumulated_loss = 0.0
        self._accumulated_count = 0

    def save_checkpoint(self, filename: str | None = None) -> None:
        """Save training checkpoint.

        Args:
            filename: Optional checkpoint filename.
        """
        if filename is None:
            filename = f"checkpoint_{self.state.step:08d}.pt"

        path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_state": {
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_loss": self.state.best_loss,
            },
            "phase_scheduler_state": self.phase_scheduler.state_dict(),
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.config.use_amp and self.config.amp_dtype != "bfloat16":
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        state = checkpoint["trainer_state"]
        self.state.step = state["step"]
        self.state.epoch = state["epoch"]
        self.state.best_loss = state["best_loss"]

        self.phase_scheduler.load_state_dict(checkpoint["phase_scheduler_state"])

        if "lr_scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path}, step {self.state.step}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    total_steps: int = 100000,
    warmup_steps: int = 1000,
    use_amp: bool = True,
    checkpoint_dir: str = "checkpoints",
    **kwargs,
) -> Trainer:
    """Convenience function to create a configured Trainer.

    Args:
        model: Swin3DDNP model.
        train_loader: Training data loader.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        total_steps: Total training steps.
        warmup_steps: LR warmup steps.
        use_amp: Use automatic mixed precision.
        checkpoint_dir: Checkpoint directory.
        **kwargs: Additional TrainerConfig arguments.

    Returns:
        Configured Trainer instance.
    """
    from swin3d_dnp.training.scheduler import create_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    lr_scheduler = create_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    config = TrainerConfig(
        total_steps=total_steps,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
        lr_scheduler=lr_scheduler,
    )
