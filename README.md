# Swin3D-DNP

A unified hierarchical 3D deep learning framework for biomedical imaging tasks.

## Overview

Swin3D-DNP implements a coarse-to-fine architecture using Swin3D encoder/decoder for 3D medical image analysis across:

- **Skull MRI**
- **Whole-body MRI**
- **Whole-body CT**

### Core Capabilities

- Landmark-like masks / keypoint heatmaps
- Organ segmentation with large size variance
- Lesion detection and segmentation with extreme size variance

### Architecture

| Stage | Description |
|-------|-------------|
| **Coarse** | Global, low-res Swin3D encoder/decoder for coarse features + task logits |
| **Fine** | Local, high-res Swin3D processing patches conditioned on coarse context |
| **Compute Allocation** | Top-K proposals + anisotropic mm-NMS, boundary-band patches for organs |
| **Stitching** | Windowed overlap-add to full-volume predictions |

## Installation

```bash
# Clone the repository
git clone https://github.com/maxrusse/Swin3d-DNP.git
cd Swin3d-DNP

# Install the package
pip install -e .

# Install test dependencies (optional)
pip install pytest pytest-cov
```

### Dependencies

- PyTorch >= 2.0
- MONAI >= 1.3
- einops
- nibabel
- numpy
- scipy

## Quick Start

```python
from swin3d_dnp.models import build_simple_swin3d_dnp
from swin3d_dnp.losses import masked_cross_entropy, masked_dice_loss
from swin3d_dnp.training import PhaseScheduler, seed_everything
import torch

# Set seed for reproducibility
seed_everything(42)

# Build model
model = build_simple_swin3d_dnp(
    in_channels=1,
    out_channels=3,
    context_channels=32,
)

# Create phase scheduler
scheduler = PhaseScheduler(total_steps=100000)

# Get phase info for current step
phase_info = scheduler.step(current_step=5000)
print(f"Phase: {phase_info['phase']}")
print(f"Lambda0: {phase_info['lambda0']}, Lambda1: {phase_info['lambda1']}")
```

## Project Structure

```
swin3d_dnp/
├── src/swin3d_dnp/
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
│   │   ├── dataset.py         # Dataset implementations
│   │   ├── transforms.py      # Data augmentation
│   │   └── sampling.py        # Patch sampling strategies
│   ├── inference/
│   │   ├── nms.py             # NMS implementation
│   │   ├── stitching.py       # Overlap-add stitching
│   │   └── predictor.py       # Inference pipeline
│   └── training/
│       ├── trainer.py         # Training loop
│       ├── scheduler.py       # Phase scheduling
│       └── utils.py           # Utilities (seeding, memory estimation)
├── tests/                     # Unit tests
├── notebooks/                 # Colab notebooks
└── configs/                   # Configuration files
```

## Training Phases

The framework uses a three-phase training schedule:

| Phase | Progress | Lambda0 | Lambda1 | Context | Hard Negatives |
|-------|----------|---------|---------|---------|----------------|
| 1 (Warmup) | 0-10% | 1.0 | 0.5 | Detached | Off |
| 2 (Transition) | 10-60% | 0.5 | 1.0 | Attached | On (after warmup) |
| 3 (Final) | 60-100% | 0.3 | 1.0 | Attached | On |

## Inference Modes

### Proposal-Driven (Lesions/Landmarks)

```python
from swin3d_dnp.inference import Predictor, InferenceConfig

config = InferenceConfig(
    mode="proposal",
    nms_threshold=0.5,
    nms_min_dist_mm=10.0,
)
predictor = Predictor(model, config)
result = predictor.predict(image, affine, spacing)
```

### Dense Tiling (Organs)

```python
config = InferenceConfig(
    mode="dense",
    overlap_fraction=0.5,
)
predictor = Predictor(model, config)
result = predictor.predict(image, affine, spacing)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_geometry.py -v

# Run with coverage
pytest tests/ --cov=swin3d_dnp --cov-report=html
```

### Testing Protocol (for External Testers)

1. Open `notebooks/run_tests.ipynb` in Google Colab
2. Enable GPU runtime (Runtime > Change runtime type > GPU)
3. Run all cells
4. Download and commit results to `test_results/`

## Documentation

| File | Description |
|------|-------------|
| `README.md` | This file - project overview and quick start |
| `CLAUDE.md` | Development guide - conventions, invariants, code style |
| `projectplan.md` | Engineering specification - detailed math and contracts |

## Key Implementation Notes

### Non-Negotiable Invariants

These **MUST** be followed to avoid silent bugs:

1. **`align_corners=False`** for every `grid_sample` usage
2. Spatial tensor order: **(D,H,W)**
3. `grid_sample` last dim order: **(x,y,z) = (W,H,D)**
4. Labels: `mode="nearest"`, `padding_mode="zeros"`
5. Images/features: `mode="bilinear"`, `padding_mode="border"`
6. **valid_mask** applied to all losses

### Coordinate Systems

```python
# Index to normalized (align_corners=False)
n = 2.0 * (u + 0.5) / N - 1.0

# Normalized to index
u = ((n + 1.0) * N) / 2.0 - 0.5
```

### World Space Conventions

- Internal tensors use **(z,y,x)** ordering
- NIfTI affines map **(x,y,z)** indices to world coordinates
- Convert zyx -> xyz before applying affines

## License

[Add license information]

## Citation

[Add citation information if applicable]
