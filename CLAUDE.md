# CLAUDE.md - Swin3D-DNP Development Guide

## Project Overview

Swin3D-DNP is a unified hierarchical 3D deep learning framework for biomedical imaging tasks across:
- **Skull MRI**
- **Whole-body MRI**
- **Whole-body CT**

### Core Capabilities
- Landmark-like masks / keypoint heatmaps (Nlm channels)
- Organ segmentation with large size variance (Corg classes)
- Lesion detection and segmentation with extreme size variance

### Architecture
- **Coarse stage**: Global, low-res Swin3D encoder/decoder for coarse features + task logits
- **Fine stage**: Local, high-res Swin3D processing patches conditioned on coarse context
- **Compute allocation**: Top-K proposals + anisotropic mm-NMS, boundary-band patches for organs
- **Stitching**: Windowed overlap-add to full-volume predictions

## Build & Test Commands

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_geometry.py -v

# Run with coverage
pytest tests/ --cov=swin3d_dnp --cov-report=html

# Type checking
mypy src/swin3d_dnp/

# Linting
ruff check src/

# Format code
ruff format src/
```

## Project Structure

```
swin3d_dnp/
├── src/swin3d_dnp/
│   ├── __init__.py
│   ├── constants.py           # Global constants (EPS, phases, ratios)
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── coordinates.py     # Coordinate transforms (index<->norm)
│   │   ├── sampling.py        # Patch sampling, context sampling
│   │   └── mapping.py         # Full<->coarse index mapping
│   ├── models/
│   │   ├── __init__.py
│   │   ├── coarse_net.py      # Coarse Swin3D encoder/decoder
│   │   ├── fine_net.py        # Fine Swin3D network
│   │   ├── fusion.py          # CoarseContextFusion layer
│   │   └── swin3d_dnp.py      # Main Swin3DDNP model
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py            # Masked dice loss
│   │   ├── ce.py              # Masked cross-entropy
│   │   └── focal.py           # Focal heatmap loss
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset classes
│   │   ├── transforms.py      # Data augmentation
│   │   └── sampling.py        # Patch sampling strategies
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── nms.py             # NMS implementation
│   │   ├── stitching.py       # Overlap-add stitching
│   │   └── predictor.py       # Inference pipeline
│   └── training/
│       ├── __init__.py
│       ├── trainer.py         # Training loop
│       └── scheduler.py       # Phase scheduling
├── tests/
│   ├── test_geometry.py
│   ├── test_losses.py
│   ├── test_models.py
│   └── test_inference.py
└── configs/
    └── default.yaml
```

## Non-Negotiable Invariants (CRITICAL)

1. **`align_corners=False`** for every `grid_sample` usage
2. Spatial tensor order in code: **(D,H,W)**
3. `grid_sample` last dim order: **(x,y,z) = (W,H,D)**
4. Crop `[s, s+S)` has discrete indices `s..s+S-1`
5. Patch center in index space: `c = s + (S-1)/2 = s + S/2 - 0.5`
6. Labels sampled with `mode="nearest"`, `padding_mode="zeros"`
7. Image/features sampled with `mode="bilinear"`, `padding_mode="border"`
8. Padding/out-of-bounds must not affect loss: **valid_mask** applied to CE and Dice
9. Coarse context sampled **inside the model** after coarse forward
10. Extent semantics: coarse context covers **same physical FOV** as fine patch

## Coordinate Systems

### Index space (voxel centers)
- Per axis length N: index `u ∈ [0, N-1]` at voxel centers

### Normalized grid space (align_corners=False)
```python
# Index to normalized
n(u) = 2.0 * (u + 0.5) / N - 1.0

# Normalized to index
u(n) = ((n + 1.0) * N) / 2.0 - 0.5
```

### World space (mm)
- NIfTI affine maps `(i,j,k)` voxel indices to world `(x,y,z)` mm
- Internal tensors use **(z,y,x)** ordering
- Convert zyx -> xyz before applying affine

## Code Style Guidelines

- Use type hints throughout
- Follow Google-style docstrings
- Keep functions focused and testable
- All geometry functions must have corresponding unit tests
- Use constants from `constants.py` instead of magic numbers

## Key Implementation Notes

### Patch Sampling
- Fine patches sampled from full volume using world-space transformations
- Augmentation applied in patch grid (rotation, scaling, translation)
- Valid mask computed for out-of-bounds regions

### Context Sampling
- Coarse features sampled differentiably at same physical FOV as fine patch
- Uses extent in coarse voxels computed from spacings
- Grid clamped to [-1, 1] with border padding

### Loss Functions
- All losses support valid_mask for excluding padded regions
- Dice uses smooth factor EPS_DICE for numerical stability
- Class weights computed per-class based on valid voxels

### Training Phases
- Phase 1 (0-10%): Coarse warmup, detach coarse context
- Phase 2 (10-60%): Transition, enable hard negative mining
- Phase 3 (60-100%): End-to-end fine-tuning

## Dependencies

- PyTorch >= 2.0
- MONAI >= 1.3
- einops
- nibabel
- numpy
- scipy
