# Physics Loss Imbalance — X+/X- Axial Direction

## Problem

The PINN was failing to learn accurate displacements for axial load cases (`X+`, `X-`) while performing well on bending cases (`Y`, `Z`, `YZ`).

### Root cause

For a beam under axial loading, the axial displacement is governed by:

```
delta_axial = F * L / (A * E)
```

For a beam under transverse bending:

```
delta_bending = F * L^3 / (3 * E * I)
```

The bending displacement scales as `(L/r)^2` relative to axial, where `L/r` is the slenderness ratio. For typical beams this ratio is 10–100, meaning **bending displacements are 100–10,000x larger than axial displacements for the same applied force**.

### Where it breaks in the code

In `losses.py`, the Navier-Cauchy physics residuals for all three displacement components were summed without normalisation:

```python
# BEFORE — unbalanced
r1 = (lam + mu) * de[0] + mu * lap1   # axial (X)
r2 = (lam + mu) * de[1] + mu * lap2   # transverse (Y)
r3 = (lam + mu) * de[2] + mu * lap3   # transverse (Z)

return torch.mean(r1**2 + r2**2 + r3**2)
```

Because `r1` involves second derivatives of U1 (axial displacement), and U1 `~` 0.001–0.01 mm while U3 (bending) `~` 1–100 mm, we have:

```
r1^2  <<  r3^2
```

The total physics loss gradient was therefore dominated by the bending components. The network received almost no physics supervision for the axial direction, so it effectively ignored X+/X- cases.

### Why the data loss alone was insufficient

The data MSE loss is computed in normalised space (`Y_n = (Y - Y_mean) / Y_std`), so in principle each output component has unit variance and contributes equally. However:

- The physics loss (weighted by `lambda_physics`) was pushing gradients almost exclusively through U2/U3.
- The validation metric `val_mae` was a single scalar averaged over all three components and all load directions — poor performance on axial cases was invisible, hidden by the larger bending errors dominating the average.

---

## Fix

### 1. Normalise physics residuals per component (`losses.py`)

Divide each residual by `sy[i]` — the displacement standard deviation for that component — before squaring. This makes all three terms dimensionless and of comparable magnitude regardless of the absolute displacement scale:

```python
# AFTER — balanced
r1 = r1 / sy[0]
r2 = r2 / sy[1]
r3 = r3 / sy[2]

return torch.mean(r1**2 + r2**2 + r3**2)
```

**Why `sy[i]` is the right normalisation factor:**

The characteristic magnitude of the Navier-Cauchy residual for component `i` is:

```
r_i_char ~ (lambda + mu) * sy[i] / sx_char^2
```

where `sy[i]` is the displacement scale and `sx_char` is the coordinate scale. Dividing by `sy[i]` removes the per-component displacement scale, leaving a balanced residual independent of whether the load is axial or bending.

### 2. Per-component validation metrics (`train.py`)

Replace the single averaged `val_mae` with per-component MAE:

```python
mae_per_comp = np.abs(pred_mm - true_mm).mean(axis=0)  # [U1, U2, U3]
```

This makes it immediately visible during training if any specific displacement direction is underperforming.

---

## Files changed

| File | Change |
|------|--------|
| `src/losses.py` | Normalise `r1`, `r2`, `r3` by `sy[0]`, `sy[1]`, `sy[2]` before computing MSE |
| `src/train.py`  | Log per-component MAE (`U1`, `U2`, `U3`) in console and W&B |
