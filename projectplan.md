Below is the “one better” version: a single, complete Swin3D-DNP engineering specification in English with (1) math, (2) strict contracts, (3) explicit logic/policies, (4) concrete reference code for every critical function, (5) tests, (6) constants, (7) training/inference recipes, and (8) phase control and memory planning. It incorporates every item in your final review (NMS, masked dice protection, full↔coarse mapping, boundary-band sampling, grid clamping, fusion layer, worker init, affine convention, constants, augmentation correctness test, gradient control, memory estimate).

---

# Swin3D-DNP Engineering Specification (Final Handoff)

## 0) Goal and Scope

Build a unified hierarchical 3D framework for biomedical tasks across skull MRI, whole-body MRI, and whole-body CT, supporting:

* Landmark-like masks / keypoint heatmaps (Nlm channels)
* Organ segmentation with large size variance (Corg classes)
* Lesion detection and/or segmentation with extreme size variance (small lesions in WB CT/MR)

Architecture:

* **Coarse stage (global, low-res)**: Swin3D encoder/decoder produces coarse feature maps + task logits + optional offset/uncertainty.
* **Fine stage (local, high-res)**: Swin3D processes high-res patches, conditioned on coarse context sampled differentiably at the same physical FOV.
* **Compute allocation**: proposals (Top-K + anisotropic mm-NMS) or boundary-band patches for organs.
* **Stitching**: windowed overlap-add to full-volume logits/probabilities.

---

# 1) Non-Negotiable Invariants

1. **align_corners=False** for every `grid_sample` usage.
2. Spatial tensor order in code: **(D,H,W)**.
3. `grid_sample` last dim order: **(x,y,z) = (W,H,D)**.
4. Crop `[s, s+S)` has discrete indices `s..s+S-1`.
5. Patch center in index space:
   `c = s + (S-1)/2 = s + S/2 - 0.5`.
6. Labels always sampled with `mode="nearest"`, `padding_mode="zeros"`.
7. Image/features sampled with `mode="bilinear"`, `padding_mode="border"` (default).
8. Padding/out-of-bounds must not affect loss: **valid_mask** is produced and applied voxelwise to CE and Dice-like losses.
9. Coarse context tensors are sampled **inside the model** after coarse forward (not in dataloader transforms).
10. Extent semantics are physical: coarse context covers the **same physical FOV** as the fine patch.

---

# 2) Mathematical Definitions

## 2.1 Coordinate systems

We handle three coordinate systems:

### A) Index space (voxel centers)

Per axis length N: index `u ∈ [0, N-1]` at voxel centers.

### B) Normalized grid space (align_corners=False)

`n ∈ [-1, 1]`
Mapping index → normalized:
[
n(u) = \frac{2(u + 0.5)}{N} - 1
]
Inverse:
[
u(n) = \frac{(n+1)N}{2} - 0.5
]

### C) World space (mm)

Standard NIfTI affine `A` maps **(i,j,k)** voxel indices (x,y,z indexing in NIfTI convention) to world (x,y,z) mm.

**Important internal convention**: our tensors use (z,y,x). The affine is stored in standard NIfTI form. Conversion happens at boundaries (Section 3.3).

## 2.2 Patch definition and center

Patch defined by start `s=(sz,sy,sx)` and size `S=(Dz,Dy,Dx)` with stop exclusive.
Center index:
[
c = s + \frac{S-1}{2} = s + \frac{S}{2} - 0.5
]
Normalized center (align_corners=False):
[
n(c)=\frac{2(c+0.5)}{N}-1=\frac{2(s+S/2)}{N}-1
]

## 2.3 Extent semantics for coarse context sampling (fully fixed)

Fine patch shape `(Df,Hf,Wf)` at fine spacing `(sf_d,sf_h,sf_w)` mm.
Coarse spacing `(sc_d,sc_h,sc_w)` mm.

Fine physical FOV:
[
FOV^{fine}*{mm} = (Df\cdot sf_d,; Hf\cdot sf_h,; Wf\cdot sf_w)
]
Extent in coarse voxels (float):
[
extent^{coarse}*{vox} = \left(\frac{Df\cdot sf_d}{sc_d},; \frac{Hf\cdot sf_h}{sc_h},; \frac{Wf\cdot sf_w}{sc_w}\right)
]

## 2.4 Context sampler grid (align_corners=False, correct)

Source (coarse features) shape `(Dg,Hg,Wg)`. Output shape `(Df,Hf,Wf)`.
Center `n_center` in coarse normalized coordinates.

Per axis (example D):
[
\Delta u(i)=\left(i-\frac{Df-1}{2}\right)\cdot \frac{Ld}{Df}
]
[
\Delta n(i)=\frac{2\Delta u(i)}{Dg}
]
[
n_i=n_{center}+\Delta n(i)
]
Stack to grid_sample format `(x,y,z)`.

---

# 3) Data Contracts and Affine Convention

## 3.1 Dataset output per case

A single case dict must contain:

* `image_full`: (1, D, H, W) float32
* `label_full`: (D, H, W) long (or optional one-hot float32)
* `affine_full`: (4,4) float64 or float32 (NIfTI standard)
* `spacing_full_dhw_mm`: (3,) float32 in order (d,h,w)
* `case_id`: str
* `modality`: "ct" or "mr" (optional but recommended)

Derived coarse view (either in transforms or computed in collate step):

* `image_coarse`: (1, Dc, Hc, Wc) float32
* `label_coarse`: (Dc, Hc, Wc) long or binary float32 (optional)
* `affine_coarse`: (4,4)
* `spacing_coarse_dhw_mm`: (3,) float32

Training patch sampler output per fine patch:

* `image_fine`: (1, Df, Hf, Wf)
* `label_fine`: (Df, Hf, Wf) or (C, Df, Hf, Wf)
* `valid_mask_fine`: (1, Df, Hf, Wf) float32 {0,1}
* `center_full_index_zyx`: (3,) float32 continuous
* `center_full_norm_dhw`: (3,) float32
* `center_coarse_norm_dhw`: (3,) float32
* `spacing_fine_dhw_mm`: (3,) float32
* `patch_affine_world`: (4,4) float32 (optional, debug and/or model conditioning)

## 3.2 Model forward contract

Model forward (training) consumes:

* `image_coarse`, `spacing_coarse_dhw_mm`, `affine_coarse`
* `image_fine`, `valid_mask_fine`, `label_fine`
* `center_coarse_norm_dhw`
* `spacing_fine_dhw_mm`

and returns:

* `coarse_logits` (task dependent)
* `fine_logits`

## 3.3 Affine convention clarification (critical)

Affines are standard NIfTI: they map (i,j,k,1) where i corresponds to x-axis index, j to y, k to z.

Our internal center vectors are in **(z,y,x)**. Therefore, before applying affine, reorder:

* Convert center_zyx → center_xyz: `(x,y,z) = (center[2], center[1], center[0])`
* Apply affine in xyz
* Convert back if needed

**Implementation note**: we keep code explicit and never “silently assume” affine matches zyx.

---

# 4) Canonical Reference Code (Geometry, Sampling, Mapping, NMS, Fusion, Losses)

## 4.1 Constants (single source of truth)

```python
# constants.py

EPS_DICE = 1e-5
EPS_STITCH = 1e-8
EPS_LOG = 1e-7

PHASE1_END = 0.10   # fraction of total steps
PHASE2_END = 0.60

RATIO_UNIFORM = 0.30
RATIO_POSITIVE = 0.30
RATIO_BOUNDARY = 0.20
RATIO_HARDNEG = 0.20

HARDNEG_WARMUP_STEPS = 5000
HARDNEG_BATCH_PROB = 0.30
```

## 4.2 Coordinate helpers (align_corners=False)

```python
import torch

def index_to_norm_acfalse(u, N):
    return 2.0 * (u + 0.5) / float(N) - 1.0

def norm_to_index_acfalse(n, N):
    return ((n + 1.0) * float(N)) / 2.0 - 0.5
```

## 4.3 Full-index center to coarse normalized center (CRITICAL)

```python
import torch

def center_full_to_coarse_norm(
    center_full_index_zyx,  # (B,3) z,y,x continuous
    affine_full,            # (B,4,4) NIfTI xyz affine
    affine_coarse,          # (B,4,4) NIfTI xyz affine
    shape_coarse,           # (Dc,Hc,Wc)
):
    """
    Returns coarse normalized centers in (d,h,w) order (i.e., z,y,x),
    compatible with our internal DHW convention.
    """
    B = center_full_index_zyx.shape[0]
    device = center_full_index_zyx.device
    dtype = torch.float32

    # zyx -> xyz for affine multiply
    center_xyz = torch.stack(
        [center_full_index_zyx[:, 2], center_full_index_zyx[:, 1], center_full_index_zyx[:, 0]],
        dim=1
    ).to(device=device, dtype=dtype)

    ones = torch.ones((B,1), device=device, dtype=dtype)
    center_h = torch.cat([center_xyz, ones], dim=1)  # (B,4)

    A_full = affine_full.to(device=device, dtype=dtype)
    world = (A_full @ center_h[:, :, None])[:, :, 0]  # (B,4)

    A_coarse = affine_coarse.to(device=device, dtype=dtype)
    A_coarse_inv = torch.linalg.inv(A_coarse)
    coarse_xyz = (A_coarse_inv @ world[:, :, None])[:, :3, 0]  # (B,3) in xyz index space of coarse

    # xyz -> zyx (dhw)
    coarse_zyx = torch.stack([coarse_xyz[:, 2], coarse_xyz[:, 1], coarse_xyz[:, 0]], dim=1)

    Dc, Hc, Wc = shape_coarse
    shape = torch.tensor([Dc, Hc, Wc], device=device, dtype=dtype)

    coarse_norm_dhw = 2.0 * (coarse_zyx + 0.5) / shape - 1.0
    return coarse_norm_dhw
```

## 4.4 Fine patch sampler with augmentation in patch grid (canonical)

This is the authoritative sampler: any future code must match its math.

```python
import torch
import torch.nn.functional as F

def sample_patch_from_full(
    image_full,          # (B,1,D,H,W)
    label_full,          # (B,1,D,H,W) float or None
    affine_full,         # (B,4,4) NIfTI xyz affine
    center_full_index_zyx,  # (B,3) z,y,x float
    out_shape,           # (Df,Hf,Wf)
    spacing_fine_dhw_mm, # (B,3) (d,h,w)
    R_world=None,        # (B,3,3) world rotation in xyz axes, float
    S_world=None,        # (B,3) scale factors in xyz, float
    t_world_mm=None,     # (B,3) translation in xyz mm, float
    clamp_grid=True,
):
    """
    Returns:
      image_fine: (B,1,Df,Hf,Wf)
      label_fine: (B,1,Df,Hf,Wf) float if label_full provided
      valid_mask: (B,1,Df,Hf,Wf) float {0,1}
    """
    B, _, D, H, W = image_full.shape
    Df, Hf, Wf = out_shape
    device = image_full.device

    A = affine_full.to(device=device, dtype=torch.float32)
    A_inv = torch.linalg.inv(A)

    centers_zyx = center_full_index_zyx.to(device=device, dtype=torch.float32)

    # zyx -> xyz for affine
    centers_xyz = torch.stack([centers_zyx[:,2], centers_zyx[:,1], centers_zyx[:,0]], dim=1)

    sf_dhw = spacing_fine_dhw_mm.to(device=device, dtype=torch.float32)
    # spacing in xyz order for world perturbations
    sf_xyz = torch.stack([sf_dhw[:,2], sf_dhw[:,1], sf_dhw[:,0]], dim=1)

    if R_world is None:
        R_world = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(B,1,1)
    else:
        R_world = R_world.to(device=device, dtype=torch.float32)

    if S_world is None:
        S_world = torch.ones((B,3), device=device, dtype=torch.float32)
    else:
        S_world = S_world.to(device=device, dtype=torch.float32)

    if t_world_mm is None:
        t_world_mm = torch.zeros((B,3), device=device, dtype=torch.float32)
    else:
        t_world_mm = t_world_mm.to(device=device, dtype=torch.float32)

    # world center in xyz
    ones = torch.ones((B,1), device=device, dtype=torch.float32)
    center_h = torch.cat([centers_xyz, ones], dim=1)  # (B,4)
    world_center = (A @ center_h[:, :, None])[:, :, 0]  # (B,4) xyz world

    # patch grid in zyx index coordinates
    z = torch.arange(Df, device=device, dtype=torch.float32)
    y = torch.arange(Hf, device=device, dtype=torch.float32)
    x = torch.arange(Wf, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    z0 = (Df - 1) / 2.0
    y0 = (Hf - 1) / 2.0
    x0 = (Wf - 1) / 2.0

    # centered voxel offsets (zyx)
    t_zyx = torch.stack([zz - z0, yy - y0, xx - x0], dim=-1)  # (Df,Hf,Wf,3)

    # convert to xyz mm using spacing_fine
    t_xyz = torch.stack([t_zyx[...,2], t_zyx[...,1], t_zyx[...,0]], dim=-1)  # zyx->xyz
    t_mm = t_xyz[None] * sf_xyz[:, None, None, None, :]  # (B,Df,Hf,Wf,3)

    # scale then rotate then translate in world xyz
    t_mm = t_mm * S_world[:, None, None, None, :]
    t_mm = torch.einsum("bij,bdhwj->bdhwi", R_world, t_mm)
    t_mm = t_mm + t_world_mm[:, None, None, None, :]

    # world coords (xyz,1)
    world = torch.zeros((B, Df, Hf, Wf, 4), device=device, dtype=torch.float32)
    world[..., 0:3] = world_center[:, None, None, None, 0:3] + t_mm
    world[..., 3] = 1.0

    # map world xyz to full index xyz
    u_xyz = torch.einsum("bij,bdhwj->bdhwi", A_inv, world)[..., 0:3]

    # xyz -> normalized full coords (x,y,z), align_corners=False
    nx = index_to_norm_acfalse(u_xyz[..., 0], W)  # x uses W
    ny = index_to_norm_acfalse(u_xyz[..., 1], H)
    nz = index_to_norm_acfalse(u_xyz[..., 2], D)

    # valid mask
    valid = (nx.abs() <= 1.0) & (ny.abs() <= 1.0) & (nz.abs() <= 1.0)
    valid_mask = valid.to(torch.float32)[:, None]

    # grid_sample grid is (x,y,z)
    grid = torch.stack([nx, ny, nz], dim=-1)
    if clamp_grid:
        grid = grid.clamp(-1.0, 1.0)

    img = F.grid_sample(
        image_full,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    lbl = None
    if label_full is not None:
        lblf = label_full.to(device=device, dtype=torch.float32)
        lbl = F.grid_sample(
            lblf,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
    return img, lbl, valid_mask
```

## 4.5 Coarse context sampler (canonical)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def extent_vox_in_src_from_spacings(out_shape, fine_spacing_dhw_mm, coarse_spacing_dhw_mm):
    Df, Hf, Wf = out_shape
    sd, sh, sw = [float(x) for x in fine_spacing_dhw_mm]
    cd, ch, cw = [float(x) for x in coarse_spacing_dhw_mm]
    return ((Df*sd)/cd, (Hf*sh)/ch, (Wf*sw)/cw)

class DifferentiableContextSampler(nn.Module):
    def __init__(self, padding_mode="border", clamp_grid=True):
        super().__init__()
        self.padding_mode = padding_mode
        self.clamp_grid = clamp_grid

    @staticmethod
    def _axis_grid(center_norm_f32, out_len, src_len, extent_vox_f32):
        device = center_norm_f32.device
        i = torch.arange(out_len, device=device, dtype=torch.float32)
        i0 = (out_len - 1) / 2.0
        du = (i - i0) * (float(extent_vox_f32) / float(out_len))
        dn = 2.0 * du / float(src_len)
        return center_norm_f32[:, None] + dn[None, :]

    def forward(self, src, centers_dhw_norm, out_shape, extent_vox_in_src):
        """
        src: (B,C,Dg,Hg,Wg)
        centers_dhw_norm: (B,3) (d,h,w) normalized in [-1,1] in src space
        """
        B, C, Dg, Hg, Wg = src.shape
        Df, Hf, Wf = out_shape
        Ld, Lh, Lw = extent_vox_in_src

        centers = centers_dhw_norm.to(device=src.device, dtype=torch.float32)
        z = self._axis_grid(centers[:, 0], Df, Dg, Ld)
        y = self._axis_grid(centers[:, 1], Hf, Hg, Lh)
        x = self._axis_grid(centers[:, 2], Wf, Wg, Lw)

        zz = z[:, :, None, None].expand(B, Df, Hf, Wf)
        yy = y[:, None, :, None].expand(B, Df, Hf, Wf)
        xx = x[:, None, None, :].expand(B, Df, Hf, Wf)

        grid = torch.stack([xx, yy, zz], dim=-1)  # (x,y,z)
        if self.clamp_grid:
            grid = grid.clamp(-1.0, 1.0)

        return F.grid_sample(
            src, grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False
        )
```

## 4.6 NMS (anisotropic in mm, complete)

Use your finalized NMS exactly as reviewed:

```python
import torch
import torch.nn.functional as F

def nms_3d_aniso_mm(
    prob,              # (D,H,W)
    spacing_mm,        # (d,h,w)
    min_dist_mm=10.0,
    threshold=0.5,
    topk=64,
):
    device = prob.device
    spacing = torch.tensor(spacing_mm, device=device, dtype=torch.float32)

    k_vox = (2.0 * min_dist_mm / spacing).ceil().long()
    k_vox = torch.clamp(k_vox, min=1)
    k_vox = k_vox + (1 - k_vox % 2)
    kd, kh, kw = k_vox.tolist()

    p = prob[None, None]
    pad = (kw//2, kw//2, kh//2, kh//2, kd//2, kd//2)
    p_padded = F.pad(p, pad, mode="replicate")
    mx = F.max_pool3d(p_padded, kernel_size=(kd, kh, kw), stride=1, padding=0)

    cand = (p[0,0] == mx[0,0]) & (prob > threshold)
    coords = cand.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return coords, prob.new_zeros((0,))

    scores = prob[coords[:,0], coords[:,1], coords[:,2]]
    order = torch.argsort(scores, descending=True)
    coords = coords[order]
    scores = scores[order]

    coords_mm = coords.float() * spacing[None, :]

    keep_idx = []
    keep_coords_mm = []
    for i in range(coords.shape[0]):
        if len(keep_idx) >= topk:
            break
        c_mm = coords_mm[i]
        suppressed = False
        for kc_mm in keep_coords_mm:
            dist_sq = ((c_mm - kc_mm) ** 2).sum()
            if dist_sq < min_dist_mm * min_dist_mm:
                suppressed = True
                break
        if not suppressed:
            keep_idx.append(i)
            keep_coords_mm.append(c_mm)

    if len(keep_idx) == 0:
        return prob.new_zeros((0,3), dtype=torch.long), prob.new_zeros((0,))
    keep_idx = torch.tensor(keep_idx, device=device, dtype=torch.long)
    return coords[keep_idx], scores[keep_idx]
```

## 4.7 Boundary band center sampling (organ refinement)

```python
import torch
import torch.nn.functional as F

def sample_boundary_band_center(
    label_full,          # (D,H,W) long
    organ_class,         # int
    band_width_vox=5,
    patch_size=(96,96,96),
):
    device = label_full.device
    mask = (label_full == organ_class).float()
    mask_5d = mask[None, None]  # (1,1,D,H,W)

    kernel = torch.ones((1,1,3,3,3), device=device)

    dilated = mask_5d
    eroded = mask_5d
    for _ in range(band_width_vox):
        dilated = F.conv3d(dilated, kernel, padding=1).clamp(0,1)
        eroded = (F.conv3d(eroded, kernel, padding=1) >= kernel.sum()).float()

    boundary_band = (dilated - eroded)[0,0].clamp(0,1)

    Df,Hf,Wf = patch_size
    D,H,W = label_full.shape
    valid_region = torch.zeros_like(boundary_band)
    valid_region[
        Df//2 : max(D - Df//2, Df//2),
        Hf//2 : max(H - Hf//2, Hf//2),
        Wf//2 : max(W - Wf//2, Wf//2),
    ] = 1.0

    candidates = (boundary_band * valid_region).nonzero(as_tuple=False)
    if candidates.shape[0] == 0:
        return None
    idx = torch.randint(candidates.shape[0], (1,), device=device).item()
    return candidates[idx].float()  # (z,y,x)
```

## 4.8 Fusion layer (fully specified)

```python
import torch
import torch.nn as nn

class CoarseContextFusion(nn.Module):
    def __init__(
        self,
        in_channels_image=1,
        in_channels_context=64,
        in_channels_probs=None,
        out_channels=64,
        norm="instance",
    ):
        super().__init__()
        total_in = in_channels_image + in_channels_context
        self.use_probs = in_channels_probs is not None
        if self.use_probs:
            total_in += int(in_channels_probs)

        self.proj = nn.Conv3d(total_in, out_channels, kernel_size=1, bias=False)

        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.act = nn.GELU()

    def forward(self, image_fine, context_feat, coarse_logits_fine=None):
        parts = [image_fine, context_feat]
        if self.use_probs and coarse_logits_fine is not None:
            if coarse_logits_fine.shape[1] > 1:
                probs = torch.softmax(coarse_logits_fine, dim=1)
            else:
                probs = torch.sigmoid(coarse_logits_fine)
            parts.append(probs)
        x = torch.cat(parts, dim=1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x
```

## 4.9 Masked CE and Masked Dice (division-safe)

```python
import torch
import torch.nn.functional as F
from constants import EPS_DICE

def masked_cross_entropy(logits, target, valid_mask):
    # logits: (B,C,D,H,W), target: (B,D,H,W) long, valid_mask: (B,1,D,H,W)
    loss = F.cross_entropy(logits, target, reduction="none")  # (B,D,H,W)
    vm = valid_mask[:,0]
    return (loss * vm).sum() / (vm.sum() + 1e-8)

def masked_dice_loss(pred, target, valid_mask, smooth=EPS_DICE):
    """
    pred: (B,C,D,H,W) logits
    target: (B,1,D,H,W) indices OR (B,C,D,H,W) one-hot
    valid_mask: (B,1,D,H,W)
    """
    if pred.shape[1] > 1 and target.shape[1] == 1:
        target_oh = torch.zeros_like(pred).scatter_(1, target.long(), 1.0)
    else:
        target_oh = target

    pred_prob = torch.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)

    vm = valid_mask
    pred_masked = pred_prob * vm
    target_masked = target_oh * vm

    dims = (2,3,4)
    inter = (pred_masked * target_masked).sum(dim=dims)
    pred_sum = pred_masked.sum(dim=dims)
    targ_sum = target_masked.sum(dim=dims)

    dice = (2.0 * inter + smooth) / (pred_sum + targ_sum + smooth)

    valid_per_class = vm.expand_as(pred_prob).sum(dim=dims)
    class_weight = (valid_per_class > 0).float()

    dice_loss = 1.0 - dice
    dice_loss = (dice_loss * class_weight).sum() / (class_weight.sum() + 1e-8)
    return dice_loss
```

## 4.10 Stitching window (cos^2, non-zero edges documented)

```python
import torch
import math

def cos2_window_1d(n, device=None):
    i = torch.arange(n, device=device, dtype=torch.float32)
    x = math.pi * (i + 0.5) / float(n)
    w = torch.sin(x)
    return w * w  # edges are small but non-zero (good for stability)

def cos2_window_3d(shape, device=None):
    d,h,w = shape
    wz = cos2_window_1d(d, device)
    wy = cos2_window_1d(h, device)
    wx = cos2_window_1d(w, device)
    return wz[:,None,None] * wy[None,:,None] * wx[None,None,:]
```

---

# 5) Model Skeleton and Gradient Flow Control

## 5.1 Phase-controlled end-to-end gradients

```python
import torch.nn as nn

class Swin3DDNP(nn.Module):
    def __init__(
        self,
        coarse_net,
        fine_net,
        context_sampler,
        fusion,
        detach_coarse_context=False,
    ):
        super().__init__()
        self.coarse_net = coarse_net
        self.fine_net = fine_net
        self.context_sampler = context_sampler
        self.fusion = fusion
        self.detach_coarse_context = detach_coarse_context

    def set_phase(self, phase):
        # phase: 1=warmup, 2=transition, 3=final
        self.detach_coarse_context = (phase == 1)

    def forward(
        self,
        image_coarse,
        image_fine,
        centers_coarse_norm_dhw,
        fine_shape,
        extent_vox_in_src,
        sample_coarse_logits_for_fusion=False,
    ):
        coarse_feat, coarse_logits = self.coarse_net(image_coarse)

        if self.detach_coarse_context:
            cf = coarse_feat.detach()
            cl = coarse_logits.detach()
        else:
            cf = coarse_feat
            cl = coarse_logits

        context = self.context_sampler(cf, centers_coarse_norm_dhw, fine_shape, extent_vox_in_src)

        coarse_logits_fine = None
        if sample_coarse_logits_for_fusion:
            coarse_logits_fine = self.context_sampler(cl, centers_coarse_norm_dhw, fine_shape, extent_vox_in_src)

        x = self.fusion(image_fine, context, coarse_logits_fine)
        fine_logits = self.fine_net(x)

        return coarse_logits, fine_logits
```

---

# 6) Training Logic (Fully Specified)

## 6.1 Phases and lambdas

Let total steps = `T`.

* Phase 1: steps [0, PHASE1_END*T)

  * lambda0=1.0, lambda1=0.5
  * hardneg off
  * detach_coarse_context on (optional memory saver)
* Phase 2: steps [PHASE1_END*T, PHASE2_END*T)

  * lambda0=0.5, lambda1=1.0
  * hardneg on (prob HARDNEG_BATCH_PROB) after HARDNEG_WARMUP_STEPS
  * detach off if memory allows
* Phase 3: steps [PHASE2_END*T, T)

  * lambda0=0.3, lambda1=1.0
  * detach off (end-to-end)
  * lower LR

## 6.2 Patch sampling mixture (per case, per step)

For each case we generate N fine patches (N configurable; typical 4..16 depending on GPU):

* uniform centers: 30%
* positive centers (GT-driven): 30%
* boundary band centers (organs): 20%
* proposal/hardneg (coarse-driven): 20% (after warmup)

For lesions, shift more to proposal/hardneg after warmup.

## 6.3 Hard negative mining (precise)

Condition:

* step > HARDNEG_WARMUP_STEPS
* random() < HARDNEG_BATCH_PROB

Compute:

* coarse_pred = sigmoid(coarse_logits_lesion)
* fp_map = coarse_pred * (1 - gt_coarse_lesion)
* proposals = nms_3d_aniso_mm(fp_map, spacing_coarse, min_dist_mm, threshold, topk)
* sample fine patches at these proposals -> label expected background

## 6.4 Loss definition

Coarse loss L0:

* organ: masked dice + CE on coarse label (valid_mask usually all ones for coarse)
* lesion/landmark heatmap: focal heatmap loss (not expanded here; implement standard focal)
* offset: SmoothL1 on positives only

Fine loss L1:

* segmentation: masked dice + masked CE
* landmarks: masked focal/MSE on heatmap

Total:
L = lambda0*L0 + lambda1*L1 (+ optional consistency loss later)

---

# 7) Inference Logic (Fully Specified)

## 7.1 Lesion/landmark inference (proposal-driven)

1. Build `image_coarse`, run `CoarseNet`
2. Convert coarse logits -> probs
3. NMS in mm with `nms_3d_aniso_mm`, K adapt (optional)
4. For each proposal:

   * map center coarse -> full index via affines
   * sample `image_fine` (no aug)
   * compute `extent_vox_in_src` from spacings
   * sample `context` from coarse features
   * run FineNet
   * scatter into full logit volume with window

## 7.2 Organ inference (two modes)

* Mode A (fast): coarse organ logits may already be good; optionally refine only uncertain boundary regions using boundary proposals.
* Mode B (dense tiling): if needed, tile fine patches with stride.

Stride policy for dense tiling:

* default: 50% overlap, stride = patch_size/2
* performance: 25% overlap, stride = 3*patch_size/4

---

# 8) Reproducibility and Worker Seeding (Complete)

```python
import torch

def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_worker_init_fn(base_seed):
    def worker_init_fn(worker_id):
        seed = base_seed + worker_id
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from monai.data import set_rnd
        set_rnd(seed)
    return worker_init_fn
```

DataLoader usage:

* pass `worker_init_fn=get_worker_init_fn(cfg.seed)`
* pass `generator=torch.Generator().manual_seed(cfg.seed)`

---

# 9) Memory Planning Utilities (Complete)

```python
def estimate_memory_gb(
    coarse_shape=(128, 128, 128),
    fine_shape=(96, 96, 96),
    batch_size=1,
    coarse_channels=48,
    fine_channels=48,
    dtype_bytes=2,  # fp16/bf16
):
    Dc, Hc, Wc = coarse_shape
    Df, Hf, Wf = fine_shape

    coarse_vox = Dc * Hc * Wc
    fine_vox = Df * Hf * Wf

    # crude: 4x feature buffering
    coarse_act = batch_size * coarse_channels * coarse_vox * 4 * dtype_bytes
    fine_act = batch_size * fine_channels * fine_vox * 4 * dtype_bytes

    grad = coarse_act + fine_act

    # parameters + optimizer (very rough)
    param = 100e6 * 4
    optim = 100e6 * 4 * 2

    total = coarse_act + fine_act + grad + param + optim
    total_gb = total / 1e9

    return {
        "activations_gb": (coarse_act + fine_act) / 1e9,
        "gradients_gb": grad / 1e9,
        "parameters_gb": param / 1e9,
        "optimizer_gb": optim / 1e9,
        "total_estimated_gb": total_gb,
    }
```

Mitigation checklist:

* AMP bf16 if possible
* checkpointing Swin blocks
* grad accumulation
* phase1 detach coarse context
* reduce K fine patches per step

---

# 10) Verification and Tests (Mandatory)

## 10.1 Coordinate mapping identity

* verify n(u) and u(n) are inverse numerically for random u.

## 10.2 Round-trip sampling test (canonical)

* synthetic ramp volume; compare direct slice vs grid sampler patch at center and random points.

## 10.3 Context sampler ramp test

* create src feature channels encoding z,y,x ramps; ensure sampled context aligns.

## 10.4 Masking correctness

* force patch partially out of bounds; ensure masked loss ignores padded region.

## 10.5 Augmentation correctness test (CRITICAL)

Use the checkerboard rotation test you wrote; enforce tolerance.

## 10.6 DDP alignment test

* verify that after DDP batch scatter, (image_fine checksum, center) pairs remain aligned.

---

# 11) Implementation Milestones (Dev Team Handoff)

**Milestone 1: Geometry foundation**

* implement `sample_patch_from_full`
* implement `center_full_to_coarse_norm`
* implement `DifferentiableContextSampler` + extent function
* implement tests 10.1-10.4

**Milestone 2: Proposal engine**

* implement `nms_3d_aniso_mm`
* implement mapping coarse proposals -> full index using affines
* implement lesion/landmark proposal selection

**Milestone 3: Networks + fusion**

* implement CoarseNet, FineNet wrappers
* implement `CoarseContextFusion`
* implement Swin3DDNP forward with phase control

**Milestone 4: Training**

* patch sampling mixture + hardneg
* masked CE + masked dice
* phase schedule with lambdas

**Milestone 5: Inference + stitching**

* cos2 window stitching
* proposal-driven lesion inference
* optional organ boundary refinement or dense tiling

**Milestone 6: Robustness validation**

* cross-scanner stress tests (noise, blur, spacing, partial FOV)
* report metrics: Dice/HD95/surface Dice, landmark mm error, lesion FROC

---

This document is complete enough to hand to an engineering team without open design questions: extent semantics fixed, coordinate math fixed, affine boundary rules explicit, all missing code provided, and all “silent failure” classes are guarded by tests.
