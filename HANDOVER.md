# Handover Protocol - Swin3D-DNP Development

**Date:** 2026-01-11
**Branch:** `claude/implement-swin3d-dnp-GFsJW`
**Last Commit:** Milestone 4 - Loss Functions

---

## 1. Project Overview

Swin3D-DNP is a unified hierarchical 3D deep learning framework for biomedical imaging tasks (skull MRI, whole-body MRI/CT). It uses a coarse-to-fine architecture with Swin3D encoder/decoder.

### Key Documentation
- `CLAUDE.md` - Development guide with invariants and conventions
- `projectplan.md` - Detailed engineering specification
- `workplan.md` - Task tracking (milestones with checkboxes)

---

## 2. Current State

### Completed Milestones

#### Milestone 1: Geometry Foundation ✅
| Task | File | Status |
|------|------|--------|
| 1.1 Coordinate Helpers | `geometry/coordinates.py` | ✅ Done |
| 1.2 Full-to-Coarse Mapping | `geometry/mapping.py` | ✅ Done |
| 1.3 Coarse-to-Full Mapping | `geometry/mapping.py` | ✅ Done |
| 1.4 Fine Patch Sampler | `geometry/sampling.py` | ✅ Done |
| 1.5 Context Sampler | `geometry/sampling.py` | ✅ Done |
| 1.6 Geometry Tests | `tests/test_geometry.py` | ✅ Done |

#### Milestone 2: Proposal Engine ✅
| Task | File | Status |
|------|------|--------|
| 2.1 Anisotropic NMS | `inference/nms.py` | ✅ Done |
| 2.2 Boundary Band Sampling | `data/sampling.py` | ✅ Done |
| 2.3 Label Downsampling | `data/transforms.py` | ✅ Done |
| 2.4 Proposal Tests | `tests/test_inference.py` | ✅ Done |

#### Milestone 3: Networks & Fusion ✅
| Task | File | Status |
|------|------|--------|
| 3.1 Coarse Network Wrapper | `models/coarse_net.py` | ✅ Done |
| 3.2 Fine Network Wrapper | `models/fine_net.py` | ✅ Done |
| 3.3 Context Fusion Layer | `models/fusion.py` | ✅ Done |
| 3.4 Main Model | `models/swin3d_dnp.py` | ✅ Done |
| 3.5 Model Tests | `tests/test_models.py` | ✅ Done |

#### Milestone 4: Loss Functions ✅ (NEW)
| Task | File | Status |
|------|------|--------|
| 4.1 Masked Cross-Entropy | `losses/ce.py` | ✅ Done |
| 4.2 Masked Dice Loss | `losses/dice.py` | ✅ Done |
| 4.3 Focal Heatmap Loss | `losses/focal.py` | ✅ Done |
| 4.4 Loss Tests | `tests/test_losses.py` | ✅ Done |

### Remaining Milestones
- **Milestone 5:** Training Pipeline (NEXT)
- **Milestone 6:** Inference Pipeline (Stitching)
- **Milestone 7:** Integration Tests

---

## 3. Milestone 4 Implementation Summary

### 4.1 Masked Cross-Entropy (`losses/ce.py`)
- `masked_cross_entropy()` - CE loss with valid_mask exclusion
- `masked_cross_entropy_per_class()` - Per-class breakdown for analysis
- Handles all-invalid masks gracefully (returns ~0)
- Supports class weights and label smoothing

### 4.2 Masked Dice Loss (`losses/dice.py`)
- `masked_dice_loss()` - Standard Dice with valid_mask
- `masked_dice_loss_per_class()` - Per-class Dice for analysis
- `masked_generalized_dice_loss()` - GDL for class imbalance
- `dice_score()` - Dice coefficient for evaluation
- Division-safe with EPS_DICE smooth factor
- Supports one-hot or class index targets

### 4.3 Focal Heatmap Loss (`losses/focal.py`)
- `focal_heatmap_loss()` - CornerNet-style for keypoint heatmaps
- `focal_heatmap_loss_from_logits()` - Accepts raw logits
- `focal_cross_entropy_loss()` - Lin et al. style for classification
- `quality_focal_loss()` - For joint classification + quality
- `offset_loss()` - Sub-voxel offset regression
- Configurable alpha/beta/gamma focusing parameters

### 4.4 Loss Tests (`tests/test_losses.py`)
- 25+ test cases covering:
  - Valid mask exclusion behavior
  - Perfect/worst prediction loss values
  - Gradient flow and backpropagation
  - Numerical stability edge cases
  - Focal focusing behavior verification

---

## 4. Bug Fixes

### Fixed: test_full_to_coarse_at_center
- **Issue:** Test used same affine for full and coarse volumes
- **Root cause:** With identical affines, index 63.5 maps to itself, giving normalized 1.0 not 0.0
- **Fix:** Test now uses proper 2x spacing coarse affine to maintain same physical FOV
- **Location:** `tests/test_geometry.py:92-128`

---

## 5. Testing

### Colab GPU Testing
A Colab notebook is available for GPU-accelerated testing:
- **File:** `notebooks/run_tests.ipynb`
- **Features:**
  - Auto-clones repo and installs dependencies
  - Runs full test suite with coverage
  - Optional push of results back to git
  - GPU memory profiling

### Local Testing
```bash
# Install
pip install -e .
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_losses.py -v

# With coverage
python -m pytest tests/ --cov=swin3d_dnp
```

---

## 6. File Structure Reference

```
src/swin3d_dnp/
├── __init__.py
├── constants.py          # ✅ Complete
├── geometry/
│   ├── __init__.py       # ✅ Complete
│   ├── coordinates.py    # ✅ Complete
│   ├── mapping.py        # ✅ Complete
│   └── sampling.py       # ✅ Complete
├── data/
│   ├── __init__.py       # ✅ Complete
│   ├── sampling.py       # ✅ Complete
│   └── transforms.py     # ✅ Complete
├── inference/
│   ├── __init__.py       # ✅ Complete
│   ├── nms.py            # ✅ Complete
│   ├── stitching.py      # ⬜ Milestone 6
│   └── predictor.py      # ⬜ Milestone 6
├── losses/
│   ├── __init__.py       # ✅ Complete (NEW)
│   ├── ce.py             # ✅ Complete (NEW)
│   ├── dice.py           # ✅ Complete (NEW)
│   └── focal.py          # ✅ Complete (NEW)
├── models/
│   ├── __init__.py       # ✅ Complete
│   ├── coarse_net.py     # ✅ Complete
│   ├── fine_net.py       # ✅ Complete
│   ├── fusion.py         # ✅ Complete
│   └── swin3d_dnp.py     # ✅ Complete
└── training/
    ├── __init__.py       # ⬜ Milestone 5
    ├── trainer.py        # ⬜ Milestone 5
    └── scheduler.py      # ⬜ Milestone 5

notebooks/
└── run_tests.ipynb       # ✅ Complete (NEW) - Colab test runner

tests/
├── conftest.py           # ✅ Fixtures
├── test_geometry.py      # ✅ Complete (fixed)
├── test_models.py        # ✅ Complete
├── test_inference.py     # ✅ Complete
└── test_losses.py        # ✅ Complete (NEW)
```

---

## 7. Next Milestone: Training Pipeline

### Milestone 5 Tasks (from workplan.md)

#### 5.1 Dataset Implementation
- **File:** `src/swin3d_dnp/data/dataset.py`
- Define data contracts (image_full, label_full, affine, spacing)
- Implement case loading and preprocessing
- Support NIfTI and numpy formats

#### 5.2 Patch Sampling Strategies
- **File:** `src/swin3d_dnp/data/sampling.py` (extend)
- Uniform sampling (30%)
- Positive sampling from GT (30%)
- Boundary band sampling (20%)
- Hard negative mining (20%)

#### 5.3 Phase Scheduler
- **File:** `src/swin3d_dnp/training/scheduler.py`
- Phase 1 (0-10%): Coarse warmup, detach context
- Phase 2 (10-60%): Transition, enable hard negative mining
- Phase 3 (60-100%): End-to-end fine-tuning
- Lambda scheduling for loss weighting

#### 5.4 Training Loop
- **File:** `src/swin3d_dnp/training/trainer.py`
- Mixed precision (bf16/fp16)
- Gradient accumulation
- Checkpointing
- Logging and metrics

#### 5.5 Worker Seeding
- **File:** `src/swin3d_dnp/training/utils.py`
- `seed_everything()` for reproducibility
- `get_worker_init_fn()` for dataloader
- DDP-safe seeding

---

## 8. Usage Example

```python
from swin3d_dnp.models import build_simple_swin3d_dnp
from swin3d_dnp.losses import masked_cross_entropy, masked_dice_loss
import torch

# Build model
model = build_simple_swin3d_dnp(
    in_channels=1,
    out_channels=3,
    context_channels=32,
)

# Prepare inputs
B = 2
image_coarse = torch.randn(B, 1, 64, 64, 64)
image_fine = torch.randn(B, 1, 32, 32, 32)
centers_coarse_norm = torch.zeros(B, 3)
target = torch.randint(0, 3, (B, 32, 32, 32))
valid_mask = torch.ones(B, 1, 32, 32, 32)

spacing_fine = torch.tensor([1.0, 1.0, 1.0])
spacing_coarse = torch.tensor([2.0, 2.0, 2.0])

# Forward pass
model.set_phase(2)
coarse_logits, fine_logits = model(
    image_coarse,
    image_fine,
    centers_coarse_norm,
    fine_shape=(32, 32, 32),
    spacing_fine_dhw_mm=spacing_fine,
    spacing_coarse_dhw_mm=spacing_coarse,
)

# Compute losses
ce_loss = masked_cross_entropy(fine_logits, target, valid_mask)
dice_loss = masked_dice_loss(fine_logits, target.unsqueeze(1), valid_mask)
total_loss = ce_loss + dice_loss
```

---

## 9. Commands Reference

```bash
# Install package
pip install -e .

# Install dependencies
pip install torch monai einops pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_losses.py -v

# Type checking
mypy src/swin3d_dnp/

# Linting
ruff check src/
```

---

## 10. Git Workflow

- **Current Branch:** `claude/implement-swin3d-dnp-GFsJW`
- **Remote:** `origin`
- After changes:
  ```bash
  git add <files>
  git commit -m "Descriptive message"
  git push -u origin claude/implement-swin3d-dnp-GFsJW
  ```

---

## 11. Summary Checklist for Next Developer

- [x] Review `CLAUDE.md` for invariants (especially align_corners=False)
- [x] Review `projectplan.md` for specifications
- [x] Fixed pre-existing test failure in `test_full_to_coarse_at_center`
- [x] Implement Milestone 4.1: Masked Cross-Entropy
- [x] Implement Milestone 4.2: Masked Dice Loss
- [x] Implement Milestone 4.3: Focal Heatmap Loss
- [x] Implement Milestone 4.4: Loss Tests
- [x] Update `workplan.md` to mark tasks complete
- [ ] Implement Milestone 5.1: Dataset Implementation
- [ ] Implement Milestone 5.2: Patch Sampling Strategies
- [ ] Implement Milestone 5.3: Phase Scheduler
- [ ] Implement Milestone 5.4: Training Loop
- [ ] Implement Milestone 5.5: Worker Seeding

---

*End of Handover Protocol*
