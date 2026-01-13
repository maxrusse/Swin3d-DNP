# Code Simplification Worklist

## Scope Plan

1. Review core inference utilities (`src/swin3d_dnp/inference/`), starting with stitching and predictor flow.
2. Expand to geometry helpers (`src/swin3d_dnp/geometry/`) and mapping utilities.
3. Review data transforms and sampling (`src/swin3d_dnp/data/`).
4. Review training utilities and schedulers (`src/swin3d_dnp/training/`).
5. Review loss functions (`src/swin3d_dnp/losses/`).
6. Review model wrappers and fusion (`src/swin3d_dnp/models/`).
7. Review tests for clarity and consistency (`tests/`).

## Behavior Lock Notes

- All changes must preserve align_corners=False behavior for sampling paths.
- Preserve tensor order conventions (D,H,W) and grid order (x,y,z).
- Keep loss masking behavior identical; valid_mask is always applied.
- Maintain inference stitching coverage and overlap behavior.

## Progress Tracking

- [x] Simplified `generate_tile_positions` range construction to reuse computed ranges while preserving edge coverage logic.
- [x] Simplified `Predictor` coarse-stage setup and patch-start construction with shared helpers.
- [x] Reviewed geometry module (`coordinates.py`, `sampling.py`, `mapping.py`): Removed unused import of `index_to_norm_acfalse` from `mapping.py`.
- [x] Reviewed data module (`dataset.py`, `sampling.py`, `transforms.py`): Clean code, no simplifications needed.
- [x] Reviewed training module (`trainer.py`, `scheduler.py`, `utils.py`): Well-structured, duplication between train/validate is intentional for explicitness.
- [x] Reviewed losses module: Extracted `_ensure_one_hot` and `_logits_to_probs` helpers in `dice.py` to reduce repeated one-hot conversion and softmax logic.
- [x] Reviewed models module (`coarse_net.py`, `fine_net.py`, `fusion.py`, `swin3d_dnp.py`): Duplication between lite/full variants is intentional for clarity.

## Remaining Work

- [ ] Review tests for clarity and consistency (`tests/`).

## Bugs/Follow-ups Noted

- None so far.
