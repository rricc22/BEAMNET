# `losses.py` — Physics Loss Function

## Purpose

Implements the **physics-informed loss** that forces the neural network to respect the laws of solid mechanics — specifically the **Navier-Cauchy equations** of linear elasticity.

This is what makes BeamNet a **PINN** (Physics-Informed Neural Network) rather than a plain data-driven model.

---

## The physics: Navier-Cauchy equations

In linear elasticity (no body forces), every point inside a solid must satisfy:

```
(λ + μ) ∇(∇·u) + μ ∇²u = 0
```

Where:
- **u** = displacement vector field `[U1, U2, U3]`
- **μ** (mu) = shear modulus = `E / (2(1 + ν))`
- **λ** (lambda) = Lamé's first parameter = `E ν / ((1 + ν)(1 − 2ν))`
- **∇²u** = Laplacian of displacement (second spatial derivatives)
- **∇(∇·u)** = gradient of divergence of displacement

If the residual of this equation is zero everywhere, the displacement field is physically consistent.

---

## Function: `physics_loss(model, x_norm, npt)`

### What it does

1. Runs the model forward to get predicted displacements `U = [U1, U2, U3]`
2. Uses **automatic differentiation** (`torch.autograd.grad`) to compute the first and second spatial derivatives of `U` with respect to `x, y, z`
3. Assembles the Navier-Cauchy residuals
4. Returns the **mean squared residual** — zero means the physics are perfectly satisfied

### The normalisation correction

The network works in normalised (z-scored) coordinates. The chain rule requires a correction when converting normalised-space derivatives back to physical units:

```
∂²U / (∂x_j ∂x_k) = H[j][:,k] × Y_std / (X_std[j] × X_std[k])
```

This ensures the physics residual is computed in **physical units (MPa/mm²)**, making it scale-invariant regardless of the data normalisation.

---

## Step-by-step walkthrough

```python
# 1. Enable gradient tracking on the input
x = x_norm.clone().requires_grad_(True)
U = model(x)  # (N, 3) normalised displacements

# 2. Recover physical E and nu from normalised inputs
E_phys = x[:, 7] * X_std[7] + X_mean[7]   # MPa
nu_phys = x[:, 8] * X_std[8] + X_mean[8]

# 3. Compute Lamé parameters
mu  = E / (2(1 + nu))
lam = E * nu / ((1 + nu)(1 - 2*nu))

# 4. Compute first-order gradients (∂U_i / ∂x_j)
g1 = grad(U[:,0], x)[:, :3]   # gradient of U1
g2 = grad(U[:,1], x)[:, :3]   # gradient of U2
g3 = grad(U[:,2], x)[:, :3]   # gradient of U3

# 5. Compute second-order gradients (Hessians)
H1[j] = grad(g1[:,j], x)[:, :3]   # Hessian of U1
# ... same for H2, H3

# 6. Scale back to physical units using p2()
# 7. Compute Laplacians: ∇²U_i = ∂²U_i/∂x² + ∂²U_i/∂y² + ∂²U_i/∂z²
# 8. Compute divergence gradient: ∇(∇·u)_i = Σ_j ∂²U_j/∂x_j∂x_i
# 9. Assemble Navier-Cauchy residuals r1, r2, r3
# 10. Return mean(r1² + r2² + r3²)
```

---

## Why this matters

Without the physics loss, the network would only fit the training data and might predict nonsensical displacements on unseen cases. The physics loss acts as a **regulariser** rooted in the actual governing equations of the problem, improving generalisation.

In `train.py`, the total loss is:

```
total_loss = data_MSE + λ_physics × physics_loss
```

with `λ_physics = 0.1`.
