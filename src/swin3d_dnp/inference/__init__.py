"""Inference utilities for Swin3D-DNP.

This module provides:
- NMS: Non-maximum suppression for lesion/landmark detection
- Stitching: Patch stitching with weighted overlap-add
- Predictor: High-level inference interface
"""

from swin3d_dnp.inference.nms import nms_3d_aniso_mm, nms_3d_isotropic
from swin3d_dnp.inference.stitching import (
    cos2_window_1d,
    cos2_window_3d,
    stitch_patches_to_volume,
    generate_tile_positions,
)
from swin3d_dnp.inference.predictor import (
    InferenceConfig,
    Predictor,
    BoundaryRefinementPredictor,
)

__all__ = [
    # NMS
    "nms_3d_aniso_mm",
    "nms_3d_isotropic",
    # Stitching
    "cos2_window_1d",
    "cos2_window_3d",
    "stitch_patches_to_volume",
    "generate_tile_positions",
    # Predictor
    "InferenceConfig",
    "Predictor",
    "BoundaryRefinementPredictor",
]
