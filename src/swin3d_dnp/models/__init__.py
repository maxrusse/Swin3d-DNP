"""Neural network models for Swin3D-DNP.

This module provides the core network components:
- CoarseNet: Low-resolution global context network
- FineNet: High-resolution patch refinement network
- CoarseContextFusion: Feature fusion layer
- Swin3DDNP: Complete hierarchical model

Example usage:
    from swin3d_dnp.models import build_swin3d_dnp

    model = build_swin3d_dnp(
        in_channels=1,
        out_channels=3,
        coarse_img_size=(128, 128, 128),
        fine_img_size=(96, 96, 96),
    )

    coarse_logits, fine_logits = model(
        image_coarse, image_fine,
        centers_coarse_norm_dhw,
        fine_shape=(96, 96, 96),
        spacing_fine_dhw_mm=fine_spacing,
        spacing_coarse_dhw_mm=coarse_spacing,
    )
"""

from swin3d_dnp.models.coarse_net import CoarseNet, CoarseNetLite
from swin3d_dnp.models.fine_net import FineNet, FineNetLite, SimpleFineNet
from swin3d_dnp.models.fusion import (
    AdaptiveCoarseContextFusion,
    CoarseContextFusion,
    SimpleFusion,
)
from swin3d_dnp.models.swin3d_dnp import (
    Swin3DDNP,
    build_simple_swin3d_dnp,
    build_swin3d_dnp,
    build_swin3d_dnp_lite,
)

__all__ = [
    # Coarse network
    "CoarseNet",
    "CoarseNetLite",
    # Fine network
    "FineNet",
    "FineNetLite",
    "SimpleFineNet",
    # Fusion layers
    "CoarseContextFusion",
    "AdaptiveCoarseContextFusion",
    "SimpleFusion",
    # Main model
    "Swin3DDNP",
    # Builder functions
    "build_swin3d_dnp",
    "build_swin3d_dnp_lite",
    "build_simple_swin3d_dnp",
]
