# Swin3D-DNP Workplan

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Milestone 1: Geometry Foundation

Core coordinate transformations and sampling functions.

### Tasks

- [x] **1.1 Coordinate Helpers**
  - Implement `index_to_norm_acfalse(u, N)`
  - Implement `norm_to_index_acfalse(n, N)`
  - File: `src/swin3d_dnp/geometry/coordinates.py`

- [x] **1.2 Full-to-Coarse Mapping**
  - Implement `center_full_to_coarse_norm()`
  - Handle zyx <-> xyz conversion with affines
  - File: `src/swin3d_dnp/geometry/mapping.py`

- [x] **1.3 Coarse-to-Full Mapping**
  - Implement `center_coarse_to_full_index()`
  - Required for inference proposal mapping
  - File: `src/swin3d_dnp/geometry/mapping.py`

- [x] **1.4 Fine Patch Sampler**
  - Implement `sample_patch_from_full()`
  - Support rotation, scaling, translation in world space
  - Compute valid_mask for out-of-bounds
  - File: `src/swin3d_dnp/geometry/sampling.py`

- [x] **1.5 Context Sampler**
  - Implement `extent_vox_in_src_from_spacings()`
  - Implement `DifferentiableContextSampler` class
  - File: `src/swin3d_dnp/geometry/sampling.py`

- [x] **1.6 Geometry Tests**
  - Test coordinate mapping identity (n(u) inverse of u(n))
  - Test round-trip sampling on synthetic ramp volume
  - Test context sampler with encoded z,y,x ramps
  - Test masking correctness with partial out-of-bounds
  - File: `tests/test_geometry.py`

---

## Milestone 2: Proposal Engine

NMS and proposal selection for lesion/landmark inference.

### Tasks

- [x] **2.1 Anisotropic NMS**
  - Implement `nms_3d_aniso_mm()`
  - Handle mm-based distances with spacing
  - Support top-k selection
  - File: `src/swin3d_dnp/inference/nms.py`

- [x] **2.2 Boundary Band Sampling**
  - Implement `sample_boundary_band_center()`
  - Morphological dilation/erosion for boundary detection
  - Valid region masking for patch placement
  - File: `src/swin3d_dnp/data/sampling.py`

- [x] **2.3 Label Downsampling**
  - Implement `downsample_label_coarse()`
  - Nearest for multi-class, maxpool for binary lesions
  - File: `src/swin3d_dnp/data/transforms.py`

- [x] **2.4 Proposal Tests**
  - Test NMS produces correct number of proposals
  - Test boundary band correctly identifies edges
  - Test label downsampling preserves small objects
  - File: `tests/test_inference.py`

---

## Milestone 3: Networks & Fusion

Core neural network components.

### Tasks

- [x] **3.1 Coarse Network Wrapper**
  - Create `CoarseNet` wrapping Swin3D encoder/decoder
  - Return both features and logits
  - File: `src/swin3d_dnp/models/coarse_net.py`

- [x] **3.2 Fine Network Wrapper**
  - Create `FineNet` for high-res processing
  - Accept fused input (image + context)
  - File: `src/swin3d_dnp/models/fine_net.py`

- [x] **3.3 Context Fusion Layer**
  - Implement `CoarseContextFusion`
  - Support optional coarse probability conditioning
  - Configurable normalization (instance/batch/layer)
  - File: `src/swin3d_dnp/models/fusion.py`

- [x] **3.4 Main Model**
  - Implement `Swin3DDNP` combining all components
  - Phase-controlled context detachment
  - `set_phase()` method for training control
  - File: `src/swin3d_dnp/models/swin3d_dnp.py`

- [x] **3.5 Model Tests**
  - Test forward pass shapes
  - Test gradient flow with/without detach
  - Test phase switching
  - File: `tests/test_models.py`

---

## Milestone 4: Loss Functions

Masked losses for handling partial volumes.

### Tasks

- [x] **4.1 Masked Cross-Entropy**
  - Implement `masked_cross_entropy()`
  - Apply valid_mask to exclude padded regions
  - File: `src/swin3d_dnp/losses/ce.py`

- [x] **4.2 Masked Dice Loss**
  - Implement `masked_dice_loss()`
  - Per-class weighting based on valid voxels
  - Division-safe with smooth factor
  - Includes Generalized Dice Loss variant
  - File: `src/swin3d_dnp/losses/dice.py`

- [x] **4.3 Focal Heatmap Loss**
  - Implement `focal_heatmap_loss()` (CornerNet-style)
  - Implement `focal_cross_entropy_loss()` (Lin et al. style)
  - Implement `offset_loss()` for sub-voxel localization
  - File: `src/swin3d_dnp/losses/focal.py`

- [x] **4.4 Loss Tests**
  - Test masked loss ignores padded regions
  - Test dice is zero for perfect predictions
  - Test focal loss focuses on hard examples
  - Test numerical stability edge cases
  - Test gradient flow through all losses
  - File: `tests/test_losses.py`

---

## Milestone 5: Training Pipeline

Training loop with phase scheduling.

### Tasks

- [x] **5.1 Dataset Implementation**
  - Define data contracts (image_full, label_full, affine, etc.)
  - Implement case loading and preprocessing
  - File: `src/swin3d_dnp/data/dataset.py`

- [x] **5.2 Patch Sampling Strategies**
  - Uniform sampling (30%)
  - Positive sampling from GT (30%)
  - Boundary band sampling (20%)
  - Hard negative mining (20%)
  - File: `src/swin3d_dnp/data/sampling.py`

- [x] **5.3 Augmentation**
  - Random rotation in world space (supported via sample_patch_from_full)
  - Random scaling (supported via sample_patch_from_full)
  - Random translation (supported via sample_patch_from_full)
  - Checkerboard rotation test for correctness
  - File: `src/swin3d_dnp/data/transforms.py`, `tests/test_transforms.py`

- [x] **5.4 Phase Scheduler**
  - Implement phase transitions (warmup/transition/final)
  - Lambda scheduling for loss weighting
  - Hard negative warmup
  - File: `src/swin3d_dnp/training/scheduler.py`

- [x] **5.5 Training Loop**
  - Mixed precision support (bf16/fp16)
  - Gradient accumulation
  - Checkpointing
  - Logging and metrics
  - File: `src/swin3d_dnp/training/trainer.py`

- [x] **5.6 Worker Seeding**
  - Implement `seed_everything()`
  - Implement `get_worker_init_fn()`
  - Ensure reproducibility
  - File: `src/swin3d_dnp/training/utils.py`

---

## Milestone 6: Inference Pipeline

Full volume inference with stitching.

### Tasks

- [x] **6.1 Stitching Window**
  - Implement `cos2_window_1d()` and `cos2_window_3d()`
  - Non-zero edges for numerical stability
  - File: `src/swin3d_dnp/inference/stitching.py`

- [x] **6.2 Patch Stitching**
  - Implement `stitch_patches_to_volume()`
  - Weighted overlap-add
  - Boundary handling
  - File: `src/swin3d_dnp/inference/stitching.py`

- [x] **6.3 Proposal-Driven Inference**
  - Lesion/landmark mode with NMS proposals
  - Map proposals through affines
  - File: `src/swin3d_dnp/inference/predictor.py`

- [x] **6.4 Dense Tiling Inference**
  - Organ mode with overlapping tiles
  - Configurable stride (50%/25% overlap)
  - File: `src/swin3d_dnp/inference/predictor.py`

- [x] **6.5 Inference Tests**
  - Test stitching uniformity (constant patches -> constant output)
  - Test proposal mapping round-trip
  - File: `tests/test_inference.py`

---

## Milestone 7: Validation & Integration

End-to-end testing and robustness.

### Tasks

- [x] **7.1 DDP Alignment Test**
  - Verify batch scatter maintains alignment
  - Test multi-GPU consistency
  - File: `tests/test_distributed.py`

- [x] **7.2 Augmentation Correctness Test**
  - Checkerboard rotation test
  - Verify image/label consistency under transforms
  - File: `tests/test_transforms.py`

- [x] **7.3 End-to-End Integration Test**
  - Full training step (forward + backward)
  - Full inference pipeline
  - File: `tests/test_integration.py`

- [x] **7.4 Memory Estimation Utility**
  - Implement `estimate_memory_gb()`
  - Document mitigation strategies
  - File: `src/swin3d_dnp/training/utils.py`

---

## Quick Reference: Key Files

| Component | File |
|-----------|------|
| Constants | `src/swin3d_dnp/constants.py` |
| Coordinates | `src/swin3d_dnp/geometry/coordinates.py` |
| Sampling | `src/swin3d_dnp/geometry/sampling.py` |
| Mapping | `src/swin3d_dnp/geometry/mapping.py` |
| Main Model | `src/swin3d_dnp/models/swin3d_dnp.py` |
| Losses | `src/swin3d_dnp/losses/*.py` |
| NMS | `src/swin3d_dnp/inference/nms.py` |
| Stitching | `src/swin3d_dnp/inference/stitching.py` |
| Trainer | `src/swin3d_dnp/training/trainer.py` |

---

## Next Steps

1. ~~Complete Milestone 1 (Geometry Foundation)~~ ✅ Done
2. ~~Complete Milestone 2 (Proposal Engine)~~ ✅ Done
3. ~~Complete Milestone 3 (Networks & Fusion)~~ ✅ Done
4. ~~Complete Milestone 4 (Loss Functions)~~ ✅ Done
5. ~~Complete Milestone 5 (Training Pipeline)~~ ✅ Done
6. ~~Complete Milestone 6 (Inference Pipeline)~~ ✅ Done
7. ~~Complete Milestone 7 (Validation & Integration)~~ ✅ Done

**All milestones complete!** The Swin3D-DNP implementation is ready for external testing.

---

## Maintenance: Code Simplification

- [x] Create `code_simplification_worklist.md` and begin tracking simplification progress.
- [x] Simplify `generate_tile_positions` range construction without changing behavior.
- [x] Add dataset/trainer smoke tests and align trainer loss/mapping with core contracts.

## Testing

See CLAUDE.md "Testing Protocol" for details.

Testing is performed by external partner after each work step using `notebooks/run_tests.ipynb`.
