# CLAUDE.md - Swin3D-DNP Development Guide

> **IMPORTANT:** New developers must read this file completely before starting work.
> See also: [AGENTS.md](AGENTS.md) for onboarding checklist.

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

---

## Required Developer Skills

### Must Have (Non-Negotiable)

1. **PyTorch Proficiency**
   - `grid_sample`, `affine_grid`, coordinate systems
   - Custom loss functions with masking
   - Mixed precision training (AMP)
   - Gradient flow debugging (`requires_grad`, `.backward()`)

2. **3D Medical Imaging**
   - NIfTI format and affine matrices
   - Voxel vs world coordinates
   - Anisotropic spacing handling
   - Understanding of align_corners semantics

3. **Deep Learning Fundamentals**
   - Encoder-decoder architectures (U-Net style)
   - Attention mechanisms (for Swin Transformer)
   - Multi-task learning
   - Loss function design (Dice, focal, cross-entropy)

4. **Software Engineering**
   - Type hints and static typing
   - Unit testing with pytest
   - Git workflow (branching, PRs)
   - Code review practices

### Good to Have

- MONAI library experience
- Distributed training (DDP)
- Memory optimization techniques
- Medical imaging metrics (Dice, HD95, FROC)

---

## Documentation Structure

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Development guide (this file) - conventions, invariants, style |
| `AGENTS.md` | Onboarding checklist for new developers |
| `projectplan.md` | **Engineering specification** - detailed math, contracts, reference code |
| `workplan.md` | Task tracking with milestone checkboxes |
| `HANDOVER.md` | Current status and next steps for handover |

**Note:** `projectplan.md` is the source of truth for implementation details. When in doubt, refer to it.

---

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

---

## Handover Testing Protocol

**CRITICAL:** Every milestone handover must include verified test results.

### Pre-Handover Checklist

1. **Run tests locally** (if PyTorch available):
   ```bash
   python -m pytest tests/ -v --tb=short
   ```

2. **Run tests on Colab** (for GPU verification):
   - Open `notebooks/run_tests.ipynb` in Google Colab
   - Enable GPU runtime
   - Run all cells
   - Verify all tests pass
   - Optionally push results to repo

3. **Document results** in HANDOVER.md:
   - Test pass/fail summary
   - Any known issues
   - GPU memory usage (if relevant)

### Post-Handover Verification

New developers should:
1. Clone the repo
2. Run the Colab notebook to verify environment
3. Confirm all existing tests pass before making changes

---

## Project Structure

```
swin3d_dnp/
├── src/swin3d_dnp/
│   ├── __init__.py
│   ├── constants.py           # Global constants (EPS, phases, ratios)
│   ├── geometry/
│   │   ├── coordinates.py     # Coordinate transforms (index<->norm)
│   │   ├── sampling.py        # Patch sampling, context sampling
│   │   └── mapping.py         # Full<->coarse index mapping
│   ├── models/
│   │   ├── coarse_net.py      # Coarse Swin3D encoder/decoder
│   │   ├── fine_net.py        # Fine Swin3D network
│   │   ├── fusion.py          # CoarseContextFusion layer
│   │   └── swin3d_dnp.py      # Main Swin3DDNP model
│   ├── losses/
│   │   ├── dice.py            # Masked dice loss
│   │   ├── ce.py              # Masked cross-entropy
│   │   └── focal.py           # Focal heatmap loss
│   ├── data/
│   │   ├── transforms.py      # Data augmentation
│   │   └── sampling.py        # Patch sampling strategies
│   ├── inference/
│   │   ├── nms.py             # NMS implementation
│   │   ├── stitching.py       # Overlap-add stitching
│   │   └── predictor.py       # Inference pipeline
│   └── training/
│       ├── trainer.py         # Training loop
│       └── scheduler.py       # Phase scheduling
├── tests/
│   ├── test_geometry.py       # Coordinate and sampling tests
│   ├── test_losses.py         # Loss function tests
│   ├── test_models.py         # Network tests
│   └── test_inference.py      # NMS and inference tests
├── notebooks/
│   └── run_tests.ipynb        # Colab test runner
└── configs/
    └── default.yaml
```

---

## Non-Negotiable Invariants (CRITICAL)

**These MUST be followed. Violations will cause silent bugs.**

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

### Common Mistakes to Avoid

| Mistake | Consequence | Correct Approach |
|---------|-------------|------------------|
| `align_corners=True` | Off-by-half-voxel sampling | Always use `False` |
| Mixing (x,y,z) and (z,y,x) | Flipped volumes | Explicit conversion at boundaries |
| Forgetting valid_mask | Loss on padded regions | Always compute and apply mask |
| Same affine for full/coarse | Wrong physical mapping | Coarse has scaled spacing |

---

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

---

## Code Style Guidelines

- Use type hints throughout
- Follow Google-style docstrings
- Keep functions focused and testable
- All geometry functions must have corresponding unit tests
- Use constants from `constants.py` instead of magic numbers
- Prefer explicit over clever code
- Fail fast with assertions; no silent error handling

---

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

---

## Dependencies

- PyTorch >= 2.0
- MONAI >= 1.3
- einops
- nibabel
- numpy
- scipy
- pytest (for testing)
