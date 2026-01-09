"""Geometry utilities for coordinate transforms and sampling."""

from swin3d_dnp.geometry.coordinates import index_to_norm_acfalse, norm_to_index_acfalse
from swin3d_dnp.geometry.mapping import center_coarse_to_full_index, center_full_to_coarse_norm
from swin3d_dnp.geometry.sampling import (
    DifferentiableContextSampler,
    extent_vox_in_src_from_spacings,
    sample_patch_from_full,
)

__all__ = [
    "index_to_norm_acfalse",
    "norm_to_index_acfalse",
    "center_full_to_coarse_norm",
    "center_coarse_to_full_index",
    "sample_patch_from_full",
    "DifferentiableContextSampler",
    "extent_vox_in_src_from_spacings",
]
