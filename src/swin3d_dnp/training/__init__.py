"""Training utilities for Swin3D-DNP.

This module provides complete training infrastructure:
- Trainer: Main training loop with checkpointing
- PhaseScheduler: Phase-based training control
- Utility functions for reproducibility and memory management
"""

from swin3d_dnp.training.scheduler import (
    PhaseScheduler,
    LRSchedulerWrapper,
    create_cosine_schedule_with_warmup,
)
from swin3d_dnp.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainerState,
    create_trainer,
)
from swin3d_dnp.training.utils import (
    seed_everything,
    get_worker_init_fn,
    estimate_memory_gb,
    get_memory_mitigation_tips,
    get_grad_scaler,
    clip_grad_norm_,
    move_batch_to_device,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "create_trainer",
    # Scheduler
    "PhaseScheduler",
    "LRSchedulerWrapper",
    "create_cosine_schedule_with_warmup",
    # Utilities
    "seed_everything",
    "get_worker_init_fn",
    "estimate_memory_gb",
    "get_memory_mitigation_tips",
    "get_grad_scaler",
    "clip_grad_norm_",
    "move_batch_to_device",
]
