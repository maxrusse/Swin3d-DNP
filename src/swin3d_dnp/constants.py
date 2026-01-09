"""Global constants for Swin3D-DNP.

This module serves as the single source of truth for all constants
used throughout the framework. Import from here instead of using
magic numbers in code.
"""

# =============================================================================
# Numerical Stability
# =============================================================================

EPS_DICE = 1e-5
"""Smoothing factor for Dice loss denominator."""

EPS_STITCH = 1e-8
"""Epsilon for stitching denominator to prevent division by zero."""

EPS_LOG = 1e-7
"""Epsilon for log operations in focal loss."""

# =============================================================================
# Training Phase Schedule
# =============================================================================

PHASE1_END = 0.10
"""Fraction of total steps for Phase 1 (coarse warmup).

Phase 1: steps [0, PHASE1_END * T)
- lambda0=1.0, lambda1=0.5
- Hard negative mining off
- detach_coarse_context on (optional memory saver)
"""

PHASE2_END = 0.60
"""Fraction of total steps for Phase 2 end (transition).

Phase 2: steps [PHASE1_END * T, PHASE2_END * T)
- lambda0=0.5, lambda1=1.0
- Hard negative mining on after HARDNEG_WARMUP_STEPS
- detach off if memory allows

Phase 3: steps [PHASE2_END * T, T)
- lambda0=0.3, lambda1=1.0
- detach off (end-to-end)
- Lower LR
"""

# =============================================================================
# Patch Sampling Ratios
# =============================================================================

RATIO_UNIFORM = 0.30
"""Fraction of patches sampled uniformly."""

RATIO_POSITIVE = 0.30
"""Fraction of patches sampled from GT positive regions."""

RATIO_BOUNDARY = 0.20
"""Fraction of patches sampled from organ boundary bands."""

RATIO_HARDNEG = 0.20
"""Fraction of patches sampled from hard negative proposals."""

# =============================================================================
# Hard Negative Mining
# =============================================================================

HARDNEG_WARMUP_STEPS = 5000
"""Number of steps before enabling hard negative mining."""

HARDNEG_BATCH_PROB = 0.30
"""Probability of including hard negatives in a batch after warmup."""

# =============================================================================
# Default Training Parameters
# =============================================================================

DEFAULT_COARSE_SHAPE = (128, 128, 128)
"""Default coarse volume shape (D, H, W)."""

DEFAULT_FINE_SHAPE = (96, 96, 96)
"""Default fine patch shape (D, H, W)."""

DEFAULT_COARSE_CHANNELS = 48
"""Default channel count for coarse network."""

DEFAULT_FINE_CHANNELS = 48
"""Default channel count for fine network."""

# =============================================================================
# NMS Parameters
# =============================================================================

DEFAULT_NMS_MIN_DIST_MM = 10.0
"""Default minimum distance in mm for NMS."""

DEFAULT_NMS_THRESHOLD = 0.5
"""Default probability threshold for NMS candidates."""

DEFAULT_NMS_TOPK = 64
"""Default maximum number of proposals from NMS."""

# =============================================================================
# Loss Weights (Phase-specific defaults)
# =============================================================================

LAMBDA0_PHASE1 = 1.0
"""Coarse loss weight in Phase 1."""

LAMBDA1_PHASE1 = 0.5
"""Fine loss weight in Phase 1."""

LAMBDA0_PHASE2 = 0.5
"""Coarse loss weight in Phase 2."""

LAMBDA1_PHASE2 = 1.0
"""Fine loss weight in Phase 2."""

LAMBDA0_PHASE3 = 0.3
"""Coarse loss weight in Phase 3."""

LAMBDA1_PHASE3 = 1.0
"""Fine loss weight in Phase 3."""
