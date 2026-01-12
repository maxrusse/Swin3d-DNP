"""Training utilities for Swin3D-DNP.

This module provides:
- Reproducibility utilities (seeding)
- Memory estimation
- DataLoader worker initialization
"""

import random
from typing import Callable

import numpy as np
import torch

from swin3d_dnp.constants import (
    DEFAULT_COARSE_CHANNELS,
    DEFAULT_COARSE_SHAPE,
    DEFAULT_FINE_CHANNELS,
    DEFAULT_FINE_SHAPE,
)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy random
    - PyTorch CPU and CUDA
    - CuDNN deterministic mode

    Note: Setting CuDNN to deterministic mode may reduce performance.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Create a worker initialization function for DataLoader.

    Each worker gets a unique seed derived from the base seed and worker ID.
    This ensures reproducibility even with multiple workers.

    Args:
        base_seed: Base seed value.

    Returns:
        Worker initialization function that takes worker_id.

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     worker_init_fn=get_worker_init_fn(42),
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """

    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker with unique seed."""
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # MONAI has its own random state management
        try:
            from monai.utils import set_determinism

            set_determinism(seed=seed)
        except (ImportError, TypeError):
            pass  # MONAI not available or API changed

    return worker_init_fn


def estimate_memory_gb(
    coarse_shape: tuple[int, int, int] = DEFAULT_COARSE_SHAPE,
    fine_shape: tuple[int, int, int] = DEFAULT_FINE_SHAPE,
    batch_size: int = 1,
    coarse_channels: int = DEFAULT_COARSE_CHANNELS,
    fine_channels: int = DEFAULT_FINE_CHANNELS,
    dtype_bytes: int = 2,
    num_fine_patches: int = 4,
) -> dict[str, float]:
    """Estimate GPU memory requirements for training.

    This provides a rough estimate based on:
    - Activation memory (4x for intermediate buffers)
    - Gradient memory (same as activations)
    - Parameter memory
    - Optimizer state memory (2x parameters for Adam)

    Args:
        coarse_shape: (Dc, Hc, Wc) coarse volume shape.
        fine_shape: (Df, Hf, Wf) fine patch shape.
        batch_size: Training batch size.
        coarse_channels: Number of coarse network channels.
        fine_channels: Number of fine network channels.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).
        num_fine_patches: Number of fine patches per coarse volume.

    Returns:
        Dictionary with memory estimates in GB:
        - activations_gb: Activation memory
        - gradients_gb: Gradient memory
        - parameters_gb: Model parameter memory
        - optimizer_gb: Optimizer state memory
        - total_estimated_gb: Total estimate
    """
    Dc, Hc, Wc = coarse_shape
    Df, Hf, Wf = fine_shape

    coarse_vox = Dc * Hc * Wc
    fine_vox = Df * Hf * Wf

    # Activation memory: 4x for intermediate feature maps
    coarse_act = batch_size * coarse_channels * coarse_vox * 4 * dtype_bytes
    fine_act = (
        batch_size * num_fine_patches * fine_channels * fine_vox * 4 * dtype_bytes
    )

    # Gradient memory: roughly same as activations
    grad = coarse_act + fine_act

    # Parameters: rough estimate ~100M parameters
    param = 100e6 * 4  # Always stored in fp32

    # Optimizer: Adam stores momentum and variance (2x params)
    optim = 100e6 * 4 * 2

    total = coarse_act + fine_act + grad + param + optim
    total_gb = total / 1e9

    return {
        "activations_gb": (coarse_act + fine_act) / 1e9,
        "gradients_gb": grad / 1e9,
        "parameters_gb": param / 1e9,
        "optimizer_gb": optim / 1e9,
        "total_estimated_gb": total_gb,
    }


def get_memory_mitigation_tips(available_gb: float, required_gb: float) -> list[str]:
    """Get tips for reducing memory usage.

    Args:
        available_gb: Available GPU memory in GB.
        required_gb: Required memory estimate in GB.

    Returns:
        List of mitigation suggestions.
    """
    tips = []

    if required_gb <= available_gb:
        tips.append("Memory requirements satisfied")
        return tips

    ratio = required_gb / available_gb

    tips.append(f"Memory ratio: {ratio:.1f}x available")

    # Always recommend
    tips.append("Use AMP (bf16/fp16) for reduced memory")

    if ratio > 1.5:
        tips.append("Enable gradient checkpointing for Swin blocks")

    if ratio > 2.0:
        tips.append("Use gradient accumulation (2-4 steps)")
        tips.append("Reduce number of fine patches per step")

    if ratio > 3.0:
        tips.append("Use Phase 1 context detachment")
        tips.append("Reduce coarse/fine volume sizes")
        tips.append("Reduce batch size to 1")

    return tips


def get_grad_scaler(
    enabled: bool = True,
    init_scale: float = 2.0**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
) -> torch.amp.GradScaler:
    """Create a gradient scaler for mixed precision training.

    Args:
        enabled: Whether scaling is enabled.
        init_scale: Initial scale factor.
        growth_factor: Factor to grow scale by.
        backoff_factor: Factor to reduce scale by on overflow.
        growth_interval: Steps between scale increases.

    Returns:
        Configured GradScaler.
    """
    return torch.amp.GradScaler(
        device="cuda",
        enabled=enabled,
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
    )


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = True,
) -> torch.Tensor:
    """Clip gradient norm with additional safety checks.

    Wrapper around torch.nn.utils.clip_grad_norm_ with better error handling.

    Args:
        parameters: Model parameters to clip.
        max_norm: Maximum gradient norm.
        norm_type: Type of norm (default: 2).
        error_if_nonfinite: Raise error on inf/nan gradients.

    Returns:
        Total norm of the parameters before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )

    return total_norm


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, torch.Tensor]:
    """Move a batch dictionary to specified device.

    Args:
        batch: Dictionary of tensors.
        device: Target device.
        non_blocking: Use non-blocking transfer.

    Returns:
        Dictionary with tensors on target device.
    """
    return {
        k: v.to(device=device, non_blocking=non_blocking)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }
