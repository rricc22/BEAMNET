"""
losses.py — loss functions for the axial beam elasticity PINN.

Currently contains:
  - physics_loss : enforces the Navier-Cauchy linear elasticity equations
                   (λ + μ) ∇(∇·u) + μ ∇²u = 0
"""

import torch


def physics_loss(model, x_norm):
    """
    Navier-Cauchy equilibrium residual for 3D linear elasticity:
      (λ + μ) ∇(∇·u) + μ ∇²u = 0   (no body forces in interior)

    Feature layout (9 columns):
      0-2  x, y, z  (z-scored, no log)
      3-5  fx, fy, fz
      6    log(F)   (log-transformed)
      7    E        (z-scored, no log)   ← used here
      8    nu       (z-scored, no log)   ← used here

    Parameters
    ----------
    model  : BeamNet
    x_norm : Tensor (N, 9)  normalised inputs

    Returns
    -------
    scalar Tensor  —  mean squared Navier-Cauchy residual in normalised space
    """
    x = x_norm.clone().requires_grad_(True)
    U = model(x)  # (N, 3) normalised displacements

    # Use normalised E and nu directly — keeps λ and μ at unit scale
    E_n  = x[:, 7].detach()   # z-scored E  (~O(1))
    nu_n = x[:, 8].detach()   # z-scored nu (~O(1))

    mu  = E_n / (2.0 * (1.0 + nu_n))
    lam = E_n * nu_n / ((1.0 + nu_n) * (1.0 - 2.0 * nu_n))

    # --- First-order spatial gradients (normalised space) ---
    g1 = torch.autograd.grad(U[:, 0].sum(), x, create_graph=True)[0][:, :3]
    g2 = torch.autograd.grad(U[:, 1].sum(), x, create_graph=True)[0][:, :3]
    g3 = torch.autograd.grad(U[:, 2].sum(), x, create_graph=True)[0][:, :3]

    # --- Second-order spatial gradients ---
    H1 = [
        torch.autograd.grad(g1[:, j].sum(), x, create_graph=True)[0][:, :3]
        for j in range(3)
    ]
    H2 = [
        torch.autograd.grad(g2[:, j].sum(), x, create_graph=True)[0][:, :3]
        for j in range(3)
    ]
    H3 = [
        torch.autograd.grad(g3[:, j].sum(), x, create_graph=True)[0][:, :3]
        for j in range(3)
    ]

    # Laplacians ∇²U_i in normalised space
    def p2(H, j, k):
        return H[j][:, k]

    lap1 = p2(H1, 0, 0) + p2(H1, 1, 1) + p2(H1, 2, 2)
    lap2 = p2(H2, 0, 0) + p2(H2, 1, 1) + p2(H2, 2, 2)
    lap3 = p2(H3, 0, 0) + p2(H3, 1, 1) + p2(H3, 2, 2)

    # Divergence gradient ∇(∇·u)_i in normalised space
    de = [p2(H1, 0, i) + p2(H2, 1, i) + p2(H3, 2, i) for i in range(3)]

    # Navier-Cauchy residuals (normalised space, weighted by material)
    r1 = (lam + mu) * de[0] + mu * lap1
    r2 = (lam + mu) * de[1] + mu * lap2
    r3 = (lam + mu) * de[2] + mu * lap3

    return torch.mean(r1**2 + r2**2 + r3**2)
