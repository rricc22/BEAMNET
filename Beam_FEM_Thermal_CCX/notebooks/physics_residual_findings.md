# Physics Residual Spatial Analysis — Findings

Notebook: `physics_residual_spatial.ipynb`
Date: 2026-04-10
Model: ThermalNet (hidden=512, 20 epochs, λ_physics=10)

---

## 1. How the physics loss works during training

During each epoch, for each batch of N nodes:
- The batch is a **random mix of nodes from many different cases** (different materials, different Q values)
- **Data loss** — forward pass, compare output to FEM label (MSE)
- **Physics loss** — a second forward pass on a subset of the batch, then differentiate the output with respect to the input coordinates (x, y, z) using autograd to compute ∇²T̂. No label needed.
- Total loss = L_data + λ · L_physics, backpropagated in one step

The physics loss is **pointwise** — it evaluates ∇²T = 0 independently at each node. The network never sees the mesh as a connected structure. Adjacent nodes are not coupled — unlike FEM where the stiffness matrix connects neighbours explicitly.

---

## 2. The physics loss is trivially small for this problem

The Laplacian residual |∇²T̂| is on the order of **1e-5**, while prediction errors |ΔT| reach **16°C** for interpolation and **hundreds to thousands of °C** for extrapolation.

**Why it is small:**

The exact analytical solution for steady-state 1D heat conduction is:
```
T(x) = T_fix + (q / (k · A)) · x
```
This is **linear in x** → ∇²T = 0 exactly by definition. Any smooth network trained on linear data will naturally produce near-zero Laplacian almost for free, without needing the physics loss to enforce it.

**Consequence:** increasing λ from 10 to 1000 would not help. It would just put a larger weight on something the network already satisfies. The optimizer would spend gradient budget keeping the Laplacian at 1e-10 instead of 1e-5, potentially hurting the data fit near boundaries.

---

## 3. The Laplacian loss enforces the wrong physics

The current physics loss enforces ∇²T = 0 — but this is trivially satisfied by the linear solution. The **actual physically informative constraints** are the boundary conditions:

| BC | Location | Expression | What it encodes |
|----|----------|------------|-----------------|
| Dirichlet | x = 0 | T = T_fix | Fixed temperature anchor |
| Neumann | x = L | ∂T/∂x = q/(k·A) | Heat flux — directly links Q and k to the gradient |

The Neumann BC is the critical one. It is an explicit algebraic relationship between Q, k, and ∂T/∂x. This is exactly what the network fails to extrapolate. The Laplacian loss provides zero signal about this relationship.

**Proposed fix — add two BC losses:**
```
L_dirichlet = mean( (T̂(x=0) - T_fix)² )             # no autograd needed
L_neumann   = mean( (∂T̂/∂x|_{x=L} - q/(k·A))² )     # first derivative only
L_total = L_data + λ_lap · L_lap + λ_dir · L_dir + λ_neu · L_neu
```

---

## 4. Single-case spatial map (case_0102 — Steel_A36, Q=52.5W, interp)

**4-panel plot:** FEM T | predicted T | |ΔT| | |∇²T̂|

- The Laplacian residual and prediction error share the **same qualitative spatial pattern** — both are elevated near x=0 and x=L and lower in the bulk
- This is a good sign: the physics residual is spatially aware of where the model struggles
- The physics residual is correctly near-zero in the bulk where the solution is linear

---

## 5. Boundary profile (case_0102)

**Profile plot:** mean |∇²T̂| and mean |ΔT| binned along x

Key observations:
- Both quantities are **high near x=0, drop toward the middle, rise again near x=L**
- The shapes are similar but **not identical**:
  - The **prediction error** is strongly concentrated at x=0 (Dirichlet face) — sharp peak, drops fast
  - The **physics residual** is more symmetric — grows continuously toward x=L and ends as high as it started
- This asymmetry reveals that the Laplacian and prediction error capture **different aspects** of the same root cause (boundary difficulty)

---

## 6. Per-node correlation (case_0102)

**Scatter plot:** |∇²T̂| vs |ΔT| per node, coloured by x-position
**Spearman ρ = 0.514** — moderate positive correlation

Three distinct clusters:

| Region | |∇²T̂| | |ΔT| | Correlation |
|--------|---------|-------|-------------|
| Dirichlet face (blue, x≈0) | moderate ~0.4e-5, **uniform** | 0 to 18°C, **highly variable** | **None** — residual is blind here |
| Bulk (pale) | low | low | trivially aligned |
| Neumann face (red, x≈L) | high | high | genuine diagonal correlation |

**Critical finding:** At the Dirichlet face, the Laplacian residual is the same for all nodes regardless of whether the temperature prediction is correct or wrong. The physics loss cannot distinguish T=20°C (correct) from T=35°C (wrong) if both are locally smooth. **The Laplacian is completely blind to Dirichlet BC errors.**

The ρ=0.514 is driven mostly by the Neumann face. The Dirichlet face actively degrades the correlation.

---

## 7. Multi-case comparison — Steel_A36, all Q levels

**Profile plot:** mean |∇²T̂| and mean |ΔT| along x for all 5 test cases

**extrap-below (0.1W, 0.3W):** flat near zero — tiny gradients, network handles these easily.

**interp (52.5W):** small residuals and errors, well-behaved across the whole beam.

**extrap-above (200W, 500W):** large residuals and errors, very different spatial patterns:

**Prediction error (right panel):**
- Moderate at x=0 (anchored by T_fix)
- Drops to minimum around x=200-300mm
- Then **grows monotonically and continuously to x=L**
- Error at x=1000mm is much larger than at x=0

This is a **slope error**, not a boundary error. The network underestimates the gradient q/(k·A). Since both predictions are anchored at T_fix at x=0, the error grows linearly with distance from that anchor:
```
error(x) ≈ (slope_true - slope_pred) · x
```
By x=1000mm you accumulate 1000× the per-unit slope error.

**Physics residual (left panel):**
- Has **two humps** — one near each BC face, lower in the middle
- Does NOT show the monotonic growth that the error shows

**Key mismatch:** The Laplacian residual detects "the field is not smooth near the BCs" — but the actual error is "the slope of the whole field is wrong". These are different problems. The Laplacian cannot see that the overall gradient is incorrect, only that local curvature exists near boundaries.

---

## 8. Root cause of extrapolation failure

The network receives q and k as input features — it knows the material. The failure is not about unknown inputs. It is about **not having learnt the q/(k·A) relationship beyond the training range**.

Within training (Q=0.5 to 100W) the network approximates the slope statistically from data. At Q=500W (5× outside training) the required gradient magnitude has never been seen. The network extrapolates the slope incorrectly — and since the beam is 1000mm long, even a small slope error becomes a massive temperature error at x=L.

The Neumann BC loss would directly address this: by explicitly penalising `∂T̂/∂x|_{x=L} ≠ q/(k·A)`, it forces the network to learn the slope relationship algebraically rather than statistically — giving it a physics-grounded signal that works even outside the training Q range.

---

## Summary of what to do next

| Finding | Action |
|---------|--------|
| Laplacian residual is 1e-5, trivially satisfied | Do not increase λ — it won't help |
| Physics loss enforces the wrong thing | Replace / supplement with BC losses |
| Dirichlet face has high error, Laplacian blind to it | Add L_dirichlet |
| Slope error grows linearly with x for extrap-above | Add L_neumann — directly encodes q/(k·A) |
| ρ=0.514, moderate correlation | Laplacian is a partial but imperfect error indicator |
| Extrapolation fails because slope is underestimated | L_neumann is the highest-priority fix |
