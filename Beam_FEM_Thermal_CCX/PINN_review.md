# PINN Review — Known Issues & Proposals

## 1. Physics loss does not use `k` (conductivity)

**Problem**  
The current physics loss enforces the Laplacian:
```
∇²T = 0
```
But the correct steady-state heat conduction equation is:
```
∇·(k ∇T) = 0
```
For spatially constant `k` (per case), this simplifies to `k ∇²T = 0`, which is equivalent to `∇²T = 0` — so the current loss is not strictly wrong. However, `k` is already in the feature vector and never used in the loss, which is inconsistent.

**Proposal**  
Pull `k` from the input feature (column 4), denormalize it, and weight the Laplacian:
```python
loss = mean( (k_physical * laplacian)² )
```
This makes the residual dimensionally correct and ensures `k` has an active role in the physics constraint, not just the data loss.

---

## 2. `npt` in the physics loss — unnecessary complexity

**Problem**  
`npt` (X_mean, X_std, Y_mean, Y_std as tensors) is passed to `physics_loss` solely to rescale the Laplacian from normalized to physical units. But since the target is zero, the scaling is irrelevant — zero in physical space is zero in normalized space.

**Proposal**  
Enforce ∇²T̂ = 0 directly in normalized space and drop `npt` from `physics_loss`:
```python
def physics_loss(model, x_norm):
    x = x_norm.clone().requires_grad_(True)
    T = model(x)
    g = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, :3]
    lap = torch.zeros(T.shape[0], device=x.device)
    for j in range(3):
        H_j = torch.autograd.grad(g[:, j].sum(), x, create_graph=True)[0][:, j]
        lap = lap + H_j
    return torch.mean(lap**2)
```
Simpler, and `lambda_physics` can be retuned to compensate for the change of scale.

---

## 3. `q` is assigned to every node — not physically realistic

**Problem**  
`q_total_mW` is a case-level scalar, tiled identically across all N nodes. In the actual FEM, `q` is a surface flux applied only to boundary/exposed nodes. Interior nodes see no direct heat input.

The network currently has no way to distinguish boundary nodes (where q acts) from interior nodes (where only conduction governs).

**Proposals**

**Option A — per-node `q` from FEM**  
Extract the actual nodal heat flux from the FEM output (zero for interior nodes, non-zero for boundary nodes). Feature vector becomes:
```
[x, y, z, q_node, k, T_fix]
```
Problem: `log(0)` is undefined for interior nodes → use `log(q + ε)` or `log1p(q)`.

**Option B — add `is_boundary` binary feature**  
Keep `q` as a case-level parameter but add a flag:
```
[x, y, z, log(q), k, T_fix, is_boundary]   # 7 features
```
The network learns: "if `is_boundary=1`, q drives temperature here; if 0, use ∇²T=0."

**Option C — separate boundary and interior losses**  
Keep the current feature vector but add an explicit boundary condition loss:
```
loss_bc = MSE(T_pred at boundary nodes, T_FEM at boundary nodes)
loss    = loss_data + λ_phys * loss_phys + λ_bc * loss_bc
```

---

## 4. Residual normalization

**Problem / open question**  
The physics loss residual (∇²T = 0) is currently not normalized. Its magnitude depends on the scale of the problem (geometry size, temperature range), which means `lambda_physics` needs retuning for every new dataset or geometry.

**Proposal**  
Normalize the residual by a reference scale, e.g. divide by the mean absolute temperature gradient:
```python
lap_normalized = lap / (mean_gradient_scale + 1e-8)
```
Or simply work fully in normalized space (see point 2), which implicitly normalizes the residual.

---

## Summary table

| Issue | Severity | Proposed fix |
|---|---|---|
| `k` unused in physics loss | Medium | Weight Laplacian by `k` |
| `npt` complexity in loss | Low | Drop it, enforce in normalized space |
| `q` uniform across all nodes | High | Per-node `q` or `is_boundary` feature |
| Residual not normalized | Medium | Work in normalized space or divide by reference scale |
