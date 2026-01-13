# CLAUDE.md - Development Guide

> **New developers:** Read this file completely before starting work.
> See [README.md](README.md) for project overview and [projectplan.md](projectplan.md) for detailed engineering specification.

## Onboarding Checklist

### Environment Setup

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Install test deps: `pip install pytest pytest-cov`
4. Verify PyTorch: `python -c "import torch; print(torch.__version__)"`

### Before Writing Code

1. Run tests: `pytest tests/ -v` (all should pass)
2. Read this file completely (especially invariants section)
3. Skim `projectplan.md` for relevant sections

---

## Non-Negotiable Invariants (CRITICAL)

**These MUST be followed. Violations cause silent bugs.**

| # | Invariant |
|---|-----------|
| 1 | **`align_corners=False`** for every `grid_sample` usage |
| 2 | Spatial tensor order in code: **(D,H,W)** |
| 3 | `grid_sample` last dim order: **(x,y,z) = (W,H,D)** |
| 4 | Crop `[s, s+S)` has discrete indices `s..s+S-1` |
| 5 | Patch center in index space: `c = s + (S-1)/2 = s + S/2 - 0.5` |
| 6 | Labels sampled with `mode="nearest"`, `padding_mode="zeros"` |
| 7 | Image/features sampled with `mode="bilinear"`, `padding_mode="border"` |
| 8 | **valid_mask** applied to CE and Dice losses |
| 9 | Coarse context sampled **inside the model** after coarse forward |
| 10 | Extent semantics: coarse context covers **same physical FOV** as fine patch |

### Common Mistakes to Avoid

| Mistake | Consequence | Correct Approach |
|---------|-------------|------------------|
| `align_corners=True` | Off-by-half-voxel sampling | Always use `False` |
| Mixing (x,y,z) and (z,y,x) | Flipped volumes | Explicit conversion at boundaries |
| Forgetting valid_mask | Loss on padded regions | Always compute and apply mask |
| Same affine for full/coarse | Wrong physical mapping | Coarse has scaled spacing |

---

## Coordinate Systems

### Index Space (voxel centers)
Per axis length N: index `u in [0, N-1]` at voxel centers.

### Normalized Grid Space (align_corners=False)
```python
# Index to normalized
n(u) = 2.0 * (u + 0.5) / N - 1.0

# Normalized to index
u(n) = ((n + 1.0) * N) / 2.0 - 0.5
```

### World Space (mm)
- NIfTI affine maps `(i,j,k)` voxel indices to world `(x,y,z)` mm
- Internal tensors use **(z,y,x)** ordering
- Convert zyx -> xyz before applying affine

---

## Required Skills

### Must Have

1. **PyTorch** - `grid_sample`, `affine_grid`, custom losses, AMP, gradient debugging
2. **3D Medical Imaging** - NIfTI, affines, voxel vs world coordinates, anisotropic spacing
3. **Deep Learning** - Encoder-decoder, attention, multi-task learning, loss design
4. **Software Engineering** - Type hints, pytest, git workflow

### Verify Understanding

```python
# Can you explain what this does?
grid = torch.nn.functional.affine_grid(theta, size, align_corners=False)
sampled = torch.nn.functional.grid_sample(
    input, grid, mode='bilinear', padding_mode='border', align_corners=False
)

# Can you convert between these?
n = 2.0 * (u + 0.5) / N - 1.0  # Index to normalized
u = ((n + 1.0) * N) / 2.0 - 0.5  # Normalized to index
```

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

## Build & Test Commands

```bash
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

## Testing Protocol

**Workflow:** Developer writes code -> External partner runs tests -> Results committed.

### For Developer
- Write tests in `tests/` for new functionality
- Run locally if PyTorch available: `pytest tests/ -v`

### For External Tester
1. Open `notebooks/run_tests.ipynb` in Google Colab
2. Enable GPU runtime
3. Run all cells
4. Download `results.txt` and place in `test_results/`
5. Commit and push

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

## Git Workflow

```bash
# Check status
git status

# Stage and commit
git add <files>
git commit -m "Descriptive message"

# Push to branch
git push -u origin <branch-name>
```

---

## Getting Help

- **Code questions**: Check `projectplan.md` for reference implementations
- **Convention questions**: Check this file for style and invariants
- **Project overview**: See `README.md`
