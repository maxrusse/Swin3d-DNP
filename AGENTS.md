# AGENTS.md - Developer Onboarding

> **STOP:** Before writing any code, complete the onboarding checklist below.

---

## Mandatory Reading

**You MUST read these files before starting work:**

| Priority | File | Time | Purpose |
|----------|------|------|---------|
| 1 | **[CLAUDE.md](CLAUDE.md)** | 15 min | Development guide, invariants, conventions |
| 2 | **[HANDOVER.md](HANDOVER.md)** | 10 min | Current status, completed work, next tasks |
| 3 | **[workplan.md](workplan.md)** | 5 min | Task tracking, milestone progress |
| 4 | **[projectplan.md](projectplan.md)** | 30 min | Detailed engineering specification (reference) |

---

## Onboarding Checklist

### Phase 1: Environment Setup

- [ ] Clone the repository
- [ ] Install dependencies: `pip install -e .`
- [ ] Install test deps: `pip install pytest pytest-cov`
- [ ] Verify PyTorch available: `python -c "import torch; print(torch.__version__)"`

### Phase 2: Verify Existing Code

- [ ] Run tests locally: `python -m pytest tests/ -v`
- [ ] **OR** Run Colab notebook: `notebooks/run_tests.ipynb`
- [ ] All tests should pass before you start

### Phase 3: Understand the Codebase

- [ ] Read CLAUDE.md completely (especially invariants section)
- [ ] Read HANDOVER.md for current status
- [ ] Review workplan.md for your assigned milestone
- [ ] Skim projectplan.md for relevant sections

### Phase 4: Start Development

- [ ] Identify your milestone from workplan.md
- [ ] Create your implementation following conventions in CLAUDE.md
- [ ] Write tests for new functionality
- [ ] Run full test suite before committing

---

## Required Skills Assessment

Before starting, confirm you understand these concepts:

### PyTorch Fundamentals
```python
# Can you explain what this does?
grid = torch.nn.functional.affine_grid(theta, size, align_corners=False)
sampled = torch.nn.functional.grid_sample(
    input, grid, mode='bilinear', padding_mode='border', align_corners=False
)
```

### Coordinate Systems
```python
# Can you convert between these?
# Index space: u ∈ [0, N-1]
# Normalized space: n ∈ [-1, 1] (align_corners=False)
n = 2.0 * (u + 0.5) / N - 1.0  # Index to normalized
u = ((n + 1.0) * N) / 2.0 - 0.5  # Normalized to index
```

### NIfTI Affines
```python
# Do you understand this transformation?
# Internal tensors: (z, y, x) ordering
# NIfTI affine: maps (x, y, z) indices to world (x, y, z) mm
# Conversion required at boundaries
```

If any of these are unclear, review the relevant sections in CLAUDE.md and projectplan.md before starting.

---

## Quick Reference

### Key Invariants (memorize these)

1. `align_corners=False` ALWAYS
2. Spatial order: (D, H, W) = (z, y, x)
3. grid_sample order: (x, y, z) = (W, H, D)
4. valid_mask applied to ALL losses
5. Coarse context sampled INSIDE model

### Common Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_losses.py -v

# Type checking
mypy src/swin3d_dnp/

# Linting
ruff check src/
```

### Git Workflow

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

## Handover Protocol

When you complete your work:

1. **Run tests** - All must pass
2. **Update HANDOVER.md** - Document what you completed
3. **Update workplan.md** - Mark tasks as complete
4. **Commit with clear message** - Describe changes
5. **Push to branch** - Include test results

---

## Getting Help

- **Code questions**: Check projectplan.md for reference implementations
- **Convention questions**: Check CLAUDE.md for style and invariants
- **Status questions**: Check HANDOVER.md for current state

---

## Summary

1. Read CLAUDE.md first (mandatory)
2. Verify tests pass before starting
3. Follow the invariants exactly
4. Test your changes
5. Document your work in HANDOVER.md
