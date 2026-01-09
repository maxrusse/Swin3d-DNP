"""Pytest configuration and fixtures for Swin3D-DNP tests."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def device():
    """Return available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    return seed_val


@pytest.fixture
def sample_affine():
    """Create a sample NIfTI affine matrix (1mm isotropic, RAS orientation)."""
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, -64.0],
            [0.0, 1.0, 0.0, -64.0],
            [0.0, 0.0, 1.0, -64.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def sample_affine_anisotropic():
    """Create a sample anisotropic affine (2mm z, 1mm x/y)."""
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, -64.0],
            [0.0, 1.0, 0.0, -64.0],
            [0.0, 0.0, 2.0, -64.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
