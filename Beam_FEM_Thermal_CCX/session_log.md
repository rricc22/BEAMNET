# Session Log — Beam Thermal PINN Development

## Overview

Two-session development log for the **ThermalNet PINN** project: a Physics-Informed Neural
Network predicting steady-state thermal response of a 1000 mm × 100 mm × 100 mm beam under
1D heat conduction (∇²T = 0, no internal sources).

---

## Session 1

### Project exploration and remote setup

- Explored the full project structure: `src/`, `scripts/`, `utils/`, `notebooks/`, `elmer_cases/`
- Set up `rsync` to a **vast.ai GPU machine** (`ssh -p 21766 root@103.116.53.20`) to sync
  dataset and source code for remote training
- Verified the FEM dataset: 525 Elmer/CalculiX cases across 5 materials
  (`Steel_A36`, `Steel_S355`, `Aluminium_6061`, `Titanium_Ti6Al4V`, `Concrete_C30`),
  split into 500 train + 25 test cases

---

### Architecture and data pipeline review (`src/`)

Reviewed and verified:

- **`src/arch.py`** — `ThermalNet`: fully-connected network, input 6 features
  `(x, y, z, log(q), k, T_fix)`, output normalised temperature T̂
- **`src/train.py`** — training loop with DataLoader, StepLR scheduler, W&B logging,
  DataParallel multi-GPU support
- **`src/losses.py`** — three loss functions:
  - `physics_loss`: Laplacian residual ∇²T̂ = 0 (autograd Hessian diagonal)
  - `dirichlet_loss`: T̂(x=0) = T_fix (forward pass only, face nodes)
  - `neumann_loss`: ∂T̂/∂x|_{x=L} = q/(k·A) (first-order autograd, face nodes)

---

### Notebook: BC losses analysis (`notebooks/bc_losses_analysis.ipynb`)

Built a notebook to visualise boundary condition residuals on a single test case:

- **2×3 subplot overview** for one case:
  - Row 1: FEM temperature, ThermalNet prediction, |ΔT| error (3D scatter)
  - Row 2: Laplacian |∇²T̂|, Dirichlet residual (2D face at x=0), Neumann residual (all nodes)
- **Dirichlet face detail**: histogram of T̂ − T_fix + 2D spatial map on y-z face
- **Neumann face detail**: gradient histogram, predicted vs target scatter, spatial face map
- **Multi-case Steel_A36**: BC residuals vs Q for all 5 test Q levels
- **All-25-cases summary**: table + scatter of BC residuals vs MAE, aggregate by regime

Key fixes during notebook development:
- `RuntimeError: Trying to backward through the graph a second time` — fixed with
  `retain_graph=True` in the inner Hessian autograd loop
- `UserWarning: FigureCanvasAgg is non-interactive` — caused by `matplotlib.use('Agg')` in
  setup cell; fixed by removing it and using `ipy_display(fig)` instead of `plt.show()`
- Panel 5 changed from "all-node |T̂ − T_fix|" to a **2D face view** at x=0, because
  |T̂ − T_fix| away from the face is just the temperature field — not a meaningful Dirichlet residual

---

### Physics loss: removed k-weighting

Original `physics_loss` had `return torch.mean((k_norm * lap) ** 2)`.

**Rationale for removal:** in steady-state heat conduction with no internal sources,
k cancels out of ∇²T = 0 — weighting by k is an arbitrary heuristic that potentially
harms training on low-k materials (Titanium, Concrete).

Changed to: `return torch.mean(lap ** 2)`

---

### train.py improvements

- Added **`argparse`**: `--epochs`, `--lambda_physics`, `--lambda_dir`, `--lambda_neu`,
  `--hidden`, `--lr` — allows launching sweeps from the command line without editing CONFIG
- Model save path includes hyperparameters:
  `thermal_pinn_h{H}-lp{X}-ld{Y}-ln{Z}-e{E}.pt`
- `load_dataset()` extended to return `materials` array for per-material val MAE tracking
- **Per-material validation MAE** logged to W&B: `val_mae_Steel_A36`,
  `val_mae_Aluminium_6061`, etc.
- W&B logs: `data_loss`, `weighted_phys_loss`, `weighted_dir_loss`, `weighted_neu_loss`,
  `total_loss`, `val_mae_C`, per-material MAEs, `lr`, `epoch`
- **Validation split**: every 5th Q value per material (sorted) reassigned from train → val,
  giving 400 train / 100 val / 25 test

---

### inference.py and visualize_results.py improvements

- Added `--model` and `--outdir` CLI arguments to both scripts
- `load_norm_params()` always reads from `ROOT / saves / norm_params.npz`
  (not from `--outdir`) — norm params are shared across all models trained on the same data

---

### Lambda hyperparameter sweep — Round 1 (Neumann loss)

Trained 4 configs for 50 epochs each:

| Model | λ_physics | λ_dir | λ_neu | interp rel% | extrap-above rel% | extrap-below MAE |
|-------|-----------|-------|-------|------------|------------------|-----------------|
| lp10-ld1-ln1  | 10 | 1  | 1 | 3.06 % | 43.15 % | 3.8 °C |
| **lp1-ld10-ln1**  | 1  | 10 | 1 | 6.37 % | **36.29 %** | 3.0 °C |
| lp5-ld2-ln2   | 5  | 2  | 2 | 7.93 % | 41.35 % | 3.5 °C |
| lp5-ld5-ln1   | 5  | 5  | 1 | 4.25 % | 38.15 % | **2.7 °C** |

Best for extrap-above: `lp1-ld10-ln1` (36.29 %).
Best overall balance: `lp5-ld5-ln1`.

Key observation: **λ_dir = 10 significantly reduces extrapolation error**.
Strong Dirichlet enforcement anchors the temperature scale, helping the model
scale correctly at unseen Q values.

---

### Experiment: full_gradient_loss

**Hypothesis:** since ∇²T = 0 with no internal sources implies ∂T/∂x is constant
everywhere (not just at x=L), enforcing the gradient at ALL nodes is a strictly
stronger constraint than the Neumann face loss.

Implemented `full_gradient_loss` in `src/losses.py`:
```python
def full_gradient_loss(model, x_norm, norm_params):
    # enforce ∂T/∂x = q/(k·A) at ALL nodes
    ...
```

Removed `neumann_loss` entirely; replaced with `full_gradient_loss` (λ_grad).

---

### Lambda sweep — Round 2 (full_gradient_loss)

Trained 4 configs:

| Model | λ_physics | λ_dir | λ_grad | interp rel% | extrap-above rel% | extrap-below MAE |
|-------|-----------|-------|--------|------------|------------------|-----------------|
| lp10-ld1-lg1  | 10 | 1  | 1 | 13.26 % | 48.31 % | 5.2 °C |
| lp1-ld10-lg1  | 1  | 10 | 1 | 19.70 % | 52.47 % | 14.0 °C |
| lp5-ld2-lg2   | 5  | 2  | 2 | **9.23 %** | 48.30 % | **4.2 °C** |
| lp5-ld5-lg1   | 5  | 5  | 1 | 25.33 % | 52.33 % | 11.6 °C |

**Conclusion: `full_gradient_loss` underperforms `neumann_loss` significantly.**
Extrap-above: ~48–52% vs ~36–38% for Neumann face loss.

The face-only Neumann loss is more targeted and less prone to over-constraining
the interior, where the network has already learned the data distribution.

**Decision: reverted to `neumann_loss`.**

---

## Session 2

### Bug fixes: path resolution in inference and visualize scripts

Both `src/inference.py` and `utils/visualize_results.py` used `Path(args.outdir)` 
(relative) when computing `out.relative_to(ROOT)` (absolute), causing:

```
ValueError: 'saves/h512-.../error_vs_q.png' is not in the subpath of
'/home/.../Beam_FEM_Thermal_CCX'
```

**Fix:** `SAVES = Path(args.outdir).resolve()` in both scripts.

---

### Run inference + visualisation on all 8 models

Ran `src/inference.py` then `utils/visualize_results.py` for all 8 models
(4 × `-ln` Neumann, 4 × `-lg` gradient), each in a dedicated output directory
`saves/h512-lp{X}-ld{Y}-l{n|g}{Z}-e50/`.

Confirmed that Neumann models are better across all regimes.

---

### Codebase cleanup after revert

After reverting `full_gradient_loss`:

- `src/losses.py`: removed `full_gradient_loss`, restored `neumann_loss`
- `src/train.py`: imports `neumann_loss`, uses `lambda_neu`, run name uses `-ln`
- Fixed docstring inconsistency: module header said `k∇²T̂ = 0` but k was already
  removed from `physics_loss` — corrected to `∇²T̂ = 0`

---

### Merged analysis notebook (`notebooks/pinn_analysis.ipynb`)

Merged `bc_losses_analysis.ipynb` and `physics_residual_spatial.ipynb` into a single
coherent notebook: **`notebooks/pinn_analysis.ipynb`** (33 cells).

**Structure:**

| Section | Content |
|---------|---------|
| 1 | Load model + norm params |
| 2 | Helper functions: `normalise_X`, `load_case`, `compute_physics_fields`, `compute_dirichlet_residuals`, `compute_neumann_residuals` |
| 3 | Single-case deep dive (case_0102, Steel_A36, Q=52.5 W): 2×3 overview, profile along x, Spearman correlation, Dirichlet face detail, Neumann face detail |
| 4 | Multi-case Steel_A36 (all 5 Q levels): laplacian profile + boxplot, BC residuals vs Q, gradient bar chart |
| 5 | All 25 test cases: combined table (|∇²T̂|, |Dir|, |Neu|%, MAE), scatter plots, aggregate stats by regime + Spearman ρ |

Key design decisions in the merged notebook:
- `compute_physics_fields` computes **both** Laplacian and ∂T/∂x in a single autograd
  pass (more efficient than calling separately)
- No `matplotlib.use('Agg')` — uses `ipy_display(fig)` for inline rendering
- Multi-case section loads laplacian + BC residuals in a **single loop** (previously
  split across two notebooks)
- Combined summary table includes all three residuals + MAE for all 25 test cases

---

## Current model performance (best config: lp1-ld10-ln1)

| Regime | n | mean-MAE | mean-rel | max-rel |
|--------|---|----------|----------|---------|
| interp | 5 | 21.7 °C | 6.4 % | 9.9 % |
| extrap-below | 10 | 3.0 °C | 381 % | 1460 % |
| extrap-above | 10 | 1937 °C | 36.3 % | 66.4 % |

Note: `extrap-below` rel% is high due to the tiny temperature rise at very low Q
(denominator ≈ 0), but the absolute error (3 °C) is small.

---

## Files modified this session

| File | Change |
|------|--------|
| `src/losses.py` | Added then removed `full_gradient_loss`; fixed docstring (`k∇²T̂` → `∇²T̂`) |
| `src/train.py` | Added argparse, per-material val MAE, W&B logging improvements, `-ln`/`-lg` naming |
| `src/inference.py` | Added `--model`, `--outdir`; fixed `Path.resolve()` bug |
| `utils/visualize_results.py` | Added `--model`, `--outdir`; fixed `Path.resolve()` bug |
| `notebooks/pinn_analysis.ipynb` | **New** — merged analysis notebook |
| `notebooks/bc_losses_analysis.ipynb` | Preserved as-is |
| `notebooks/physics_residual_spatial.ipynb` | Preserved as-is |
