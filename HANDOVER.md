# Handover Protocol - Swin3D-DNP Development

**Date:** 2026-01-09
**Branch:** `claude/review-and-next-task-EzNTL`
**Last Commit:** `fe72c93` - Implement Milestone 2: Proposal Engine

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

### Remaining Milestones
- **Milestone 3:** Networks & Fusion (NEXT)
- **Milestone 4:** Loss Functions
- **Milestone 5:** Training Pipeline
- **Milestone 6:** Inference Pipeline
- **Milestone 7:** Integration Tests

---

## 3. Known Issues

### Pre-existing Test Failure
```
FAILED tests/test_geometry.py::TestMapping::test_full_to_coarse_at_center
```
- **Location:** `tests/test_geometry.py:106`
- **Issue:** Test expects center point (63.5, 63.5, 63.5) in 128³ volume to map to normalized (0,0,0) in 64³ coarse volume, but gets (1,1,1)
- **Likely cause:** Either bug in `center_full_to_coarse_norm()` or incorrect test expectation
- **Impact:** Does not block Milestone 2+, but should be investigated

### Test Environment Note
- Use `/usr/local/bin/python3 -m pytest` to run tests (avoids PATH issues)
- Dependencies installed: torch 2.9.1, monai 1.5.1, scipy 1.16.3

---

## 4. Next Milestone: Networks & Fusion

### Milestone 3 Tasks (from workplan.md)

#### 3.1 Coarse Network Wrapper
- **File:** `src/swin3d_dnp/models/coarse_net.py`
- **Purpose:** Wrap MONAI's SwinUNETR for coarse-resolution processing
- **Implementation:**
  ```python
  class CoarseNet(nn.Module):
      def __init__(self, in_channels, out_channels, feature_size=48, ...):
          # Use monai.networks.nets.SwinUNETR
          # Output: coarse logits + encoder features for context

      def forward(self, x_coarse):
          # Returns: (logits, features_for_context)
  ```
- **Key considerations:**
  - Feature size configurable (default 48)
  - Must expose intermediate features for context fusion
  - Output channels = Nlm (landmarks) + Corg (organs) + 1 (lesion)

#### 3.2 Fine Network Wrapper
- **File:** `src/swin3d_dnp/models/fine_net.py`
- **Purpose:** Process high-res patches with coarse context
- **Implementation:**
  ```python
  class FineNet(nn.Module):
      def __init__(self, in_channels, context_channels, out_channels, ...):
          # SwinUNETR + context fusion at bottleneck

      def forward(self, x_fine, context_features):
          # Fuse context at encoder bottleneck
          # Returns: fine logits
  ```

#### 3.3 Context Fusion Layer
- **File:** `src/swin3d_dnp/models/fusion.py`
- **Purpose:** Fuse coarse context features with fine encoder features
- **Implementation:**
  ```python
  class CoarseContextFusion(nn.Module):
      def __init__(self, fine_channels, context_channels):
          # Projection + concatenation or addition

      def forward(self, fine_features, context_features):
          # context_features already sampled to match fine patch FOV
          # Returns: fused features
  ```
- **Reference:** See `DifferentiableContextSampler` in `geometry/sampling.py` for context extraction

#### 3.4 Main Model Assembly
- **File:** `src/swin3d_dnp/models/swin3d_dnp.py`
- **Purpose:** Combine coarse + fine networks with context sampling
- **Implementation:**
  ```python
  class Swin3DDNP(nn.Module):
      def __init__(self, config):
          self.coarse_net = CoarseNet(...)
          self.fine_net = FineNet(...)
          self.context_sampler = DifferentiableContextSampler(...)

      def forward(self, x_full, x_coarse, patch_centers, spacing_full, spacing_coarse):
          # 1. Coarse forward
          coarse_logits, coarse_features = self.coarse_net(x_coarse)

          # 2. Sample context for each patch
          context = self.context_sampler(coarse_features, patch_centers, ...)

          # 3. Fine forward (per patch or batched)
          fine_logits = self.fine_net(patches, context)

          return coarse_logits, fine_logits
  ```

#### 3.5 Network Tests
- **File:** `tests/test_models.py`
- **Tests needed:**
  - Coarse net output shapes
  - Fine net output shapes with context
  - Fusion layer gradient flow
  - Full model forward pass
  - Context sampling integration

---

## 5. Implementation Guidance

### Critical Invariants (from CLAUDE.md)
1. **`align_corners=False`** for ALL `grid_sample` usage
2. Spatial tensor order: **(D, H, W)**
3. `grid_sample` last dim order: **(x, y, z) = (W, H, D)**
4. Labels: `mode="nearest"`, `padding_mode="zeros"`
5. Images/features: `mode="bilinear"`, `padding_mode="border"`

### Architecture Notes
- Coarse context covers **same physical FOV** as fine patch
- Context sampled **inside the model** after coarse forward
- Use `extent_vox_in_src_from_spacings()` from `geometry/sampling.py`

### Existing Utilities to Use
```python
# Context sampling (already implemented)
from swin3d_dnp.geometry.sampling import (
    DifferentiableContextSampler,
    extent_vox_in_src_from_spacings,
    sample_patch_from_full,
)

# Coordinate transforms
from swin3d_dnp.geometry.coordinates import (
    index_to_norm_acfalse,
    norm_to_index_acfalse,
)

# Constants
from swin3d_dnp.constants import (
    EPS_DICE,
    PHASE1_END, PHASE2_END,  # Training phases
    DEFAULT_NMS_TOPK,
)
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
│   ├── __init__.py       # ⬜ Milestone 4
│   ├── dice.py           # ⬜ Milestone 4
│   ├── ce.py             # ⬜ Milestone 4
│   └── focal.py          # ⬜ Milestone 4
├── models/
│   ├── __init__.py       # ⬜ Milestone 3
│   ├── coarse_net.py     # ⬜ Milestone 3
│   ├── fine_net.py       # ⬜ Milestone 3
│   ├── fusion.py         # ⬜ Milestone 3
│   └── swin3d_dnp.py     # ⬜ Milestone 3
└── training/
    ├── __init__.py       # ⬜ Milestone 5
    ├── trainer.py        # ⬜ Milestone 5
    └── scheduler.py      # ⬜ Milestone 5
```

---

## 7. Commands Reference

```bash
# Install package
pip install -e .

# Run all tests
/usr/local/bin/python3 -m pytest tests/ -v

# Run specific test file
/usr/local/bin/python3 -m pytest tests/test_models.py -v

# Type checking
mypy src/swin3d_dnp/

# Linting
ruff check src/
```

---

## 8. Git Workflow

- **Branch:** `claude/review-and-next-task-EzNTL`
- **Remote:** `origin`
- After changes:
  ```bash
  git add <files>
  git commit -m "Descriptive message"
  git push -u origin claude/review-and-next-task-EzNTL
  ```

---

## 9. Summary Checklist for Next Developer

- [ ] Review `CLAUDE.md` for invariants
- [ ] Review `projectplan.md` for architecture details
- [ ] Investigate pre-existing test failure in `test_full_to_coarse_at_center`
- [ ] Implement Milestone 3.1: Coarse Network Wrapper
- [ ] Implement Milestone 3.2: Fine Network Wrapper
- [ ] Implement Milestone 3.3: Context Fusion Layer
- [ ] Implement Milestone 3.4: Main Model Assembly
- [ ] Implement Milestone 3.5: Network Tests
- [ ] Update `workplan.md` to mark tasks complete
- [ ] Commit and push changes

---

*End of Handover Protocol*
