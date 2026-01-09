# Handover Protocol - Swin3D-DNP Development

**Date:** 2026-01-09
**Branch:** `claude/review-project-status-B66VD`
**Last Commit:** Milestone 3 - Networks & Fusion

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

### Remaining Milestones
- **Milestone 4:** Loss Functions (NEXT)
- **Milestone 5:** Training Pipeline
- **Milestone 6:** Inference Pipeline
- **Milestone 7:** Integration Tests

---

## 3. Milestone 3 Implementation Summary

### 3.1 CoarseNet (`models/coarse_net.py`)
- Wraps MONAI's `SwinUNETR` for coarse-resolution processing
- Returns both task logits and intermediate features for context fusion
- Features extracted from encoder bottleneck and projected
- Variants: `CoarseNet` (full), `CoarseNetLite` (testing)

### 3.2 FineNet (`models/fine_net.py`)
- Wraps `SwinUNETR` for high-resolution patch processing
- Accepts fused input (image + context features)
- Variants: `FineNet`, `FineNetLite`, `SimpleFineNet` (conv-based for fast testing)

### 3.3 CoarseContextFusion (`models/fusion.py`)
- Fuses fine image with sampled coarse context features
- Optional coarse probability conditioning via softmax/sigmoid
- Configurable normalization (instance/batch/layer)
- Variants: `CoarseContextFusion`, `AdaptiveCoarseContextFusion`, `SimpleFusion`

### 3.4 Swin3DDNP (`models/swin3d_dnp.py`)
- Complete hierarchical model combining all components
- Phase-controlled gradient flow via `set_phase()`
- Phase 1: detach context (memory-efficient warmup)
- Phase 2-3: end-to-end gradients
- Builder functions: `build_swin3d_dnp()`, `build_swin3d_dnp_lite()`, `build_simple_swin3d_dnp()`

### 3.5 Model Tests (`tests/test_models.py`)
- Forward pass shape tests
- Gradient flow tests with/without detach
- Phase switching tests
- Context sampler integration tests
- Fusion layer tests

---

## 4. Known Issues

### Pre-existing Test Failure
```
FAILED tests/test_geometry.py::TestMapping::test_full_to_coarse_at_center
```
- **Location:** `tests/test_geometry.py:106`
- **Issue:** Test expects center point (63.5, 63.5, 63.5) in 128³ volume to map to normalized (0,0,0) in 64³ coarse volume, but gets (1,1,1)
- **Likely cause:** Either bug in `center_full_to_coarse_norm()` or incorrect test expectation
- **Impact:** Does not block Milestone 3+, but should be investigated

### Test Environment Note
- Dependencies: PyTorch, MONAI, einops required
- Install: `pip install torch monai einops`
- Run tests: `python -m pytest tests/ -v`

---

## 5. Next Milestone: Loss Functions

### Milestone 4 Tasks (from workplan.md)

#### 4.1 Masked Cross-Entropy
- **File:** `src/swin3d_dnp/losses/ce.py`
- **Implementation:**
  ```python
  def masked_cross_entropy(logits, target, valid_mask):
      # logits: (B,C,D,H,W), target: (B,D,H,W) long
      # valid_mask: (B,1,D,H,W)
      loss = F.cross_entropy(logits, target, reduction="none")
      vm = valid_mask[:,0]
      return (loss * vm).sum() / (vm.sum() + 1e-8)
  ```

#### 4.2 Masked Dice Loss
- **File:** `src/swin3d_dnp/losses/dice.py`
- Per-class weighting based on valid voxels
- Division-safe with EPS_DICE smooth factor

#### 4.3 Focal Heatmap Loss
- **File:** `src/swin3d_dnp/losses/focal.py`
- CornerNet-style for keypoint/lesion heatmaps
- Alpha/beta focusing parameters

#### 4.4 Loss Tests
- **File:** `tests/test_losses.py`
- Test masked loss ignores padded regions
- Test dice is zero for perfect predictions
- Test focal loss focuses on hard examples

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
│   ├── __init__.py       # ⬜ Milestone 4
│   ├── dice.py           # ⬜ Milestone 4
│   ├── ce.py             # ⬜ Milestone 4
│   └── focal.py          # ⬜ Milestone 4
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
```

---

## 7. Usage Example

```python
from swin3d_dnp.models import build_simple_swin3d_dnp
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
centers_coarse_norm = torch.zeros(B, 3)  # Center of volume

spacing_fine = torch.tensor([1.0, 1.0, 1.0])
spacing_coarse = torch.tensor([2.0, 2.0, 2.0])

# Forward pass
model.set_phase(2)  # Enable end-to-end gradients
coarse_logits, fine_logits = model(
    image_coarse,
    image_fine,
    centers_coarse_norm,
    fine_shape=(32, 32, 32),
    spacing_fine_dhw_mm=spacing_fine,
    spacing_coarse_dhw_mm=spacing_coarse,
)
```

---

## 8. Commands Reference

```bash
# Install package
pip install -e .

# Install dependencies
pip install torch monai einops pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v

# Type checking
mypy src/swin3d_dnp/

# Linting
ruff check src/
```

---

## 9. Git Workflow

- **Current Branch:** `claude/review-project-status-B66VD`
- **Remote:** `origin`
- After changes:
  ```bash
  git add <files>
  git commit -m "Descriptive message"
  git push -u origin claude/review-project-status-B66VD
  ```

---

## 10. Summary Checklist for Next Developer

- [ ] Review `CLAUDE.md` for invariants (especially align_corners=False)
- [ ] Review `projectplan.md` for loss function specifications
- [ ] Investigate pre-existing test failure in `test_full_to_coarse_at_center`
- [ ] Implement Milestone 4.1: Masked Cross-Entropy
- [ ] Implement Milestone 4.2: Masked Dice Loss
- [ ] Implement Milestone 4.3: Focal Heatmap Loss
- [ ] Implement Milestone 4.4: Loss Tests
- [ ] Update `workplan.md` to mark tasks complete
- [ ] Commit and push changes

---

*End of Handover Protocol*
