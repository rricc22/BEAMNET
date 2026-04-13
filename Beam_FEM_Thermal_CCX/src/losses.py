"""
losses.py — loss functions for the thermal beam PINN.

Contains:
  - physics_loss    : Laplacian residual  ∇²T̂ = 0   (normalised coords)
  - dirichlet_loss  : T̂(x=0)  = T_fix   (Dirichlet BC, forward-pass only)
  - neumann_loss    : ∂T̂/∂x|_{x=L} = q/(k·A)  (Neumann BC, 1st-order autograd)

Feature layout (6 columns, normalised):
  0-2  x, y, z   (z-scored, mm)
  3    log(q)    (log-transformed then z-scored, mW)
  4    k         (z-scored, mW/mm/°C)
  5    T_fix     (z-scored, °C)
"""

import torch

# Reference beam cross-section area: 100 mm × 100 mm
BEAM_AREA_MM2 = 10_000.0

# Beam x-extent in physical space [mm]
_X_MIN_MM = 0.0
_X_MAX_MM = 1000.0
# Tolerance for detecting boundary nodes (2 % of beam length)
_BC_TOL_MM = 20.0


def physics_loss(model, x_norm):
    """
    Laplacian residual for 3D steady-state heat conduction in normalised space:
      ∇²T̂ = 0   (sum of ∂²T̂/∂x̂_j² over j=0,1,2)

    Feature layout (6 columns):
      0-2  x, y, z   (z-scored)
      3    log(q)    (log-transformed heat flux)
      4    k         (z-scored conductivity)
      5    T_fix     (z-scored)

    Parameters
    ----------
    model   : ThermalNet
    x_norm  : Tensor (N, 6)   normalised inputs, no grad required on entry

    Returns
    -------
    scalar Tensor  —  mean squared Laplacian residual in normalised coords
    """
    x = x_norm.clone().requires_grad_(True)
    T = model(x)  # (N, 1) normalised temperature

    # First-order spatial gradients in normalised space
    g = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, :3]  # (N, 3)

    # Diagonal Hessian: ∂²T̂/∂x̂_j²
    lap = torch.zeros(T.shape[0], device=x.device)
    for j in range(3):
        grad2 = torch.autograd.grad(g[:, j].sum(), x, create_graph=True)[0]
        lap = lap + grad2[:, j]

    return torch.mean(lap ** 2)


def dirichlet_loss(model, x_norm, norm_params):
    """
    Dirichlet BC loss: T̂(x=0) = T_fix

    Selects nodes near x=0 (the fixed-temperature face) from the batch and
    penalises the deviation of the predicted (normalised) temperature from the
    known boundary value.  No autograd needed — forward pass only.

    Parameters
    ----------
    model       : ThermalNet
    x_norm      : Tensor (N, 6)  normalised inputs
    norm_params : dict           keys X_mean, X_std (shape 6), Y_mean, Y_std (shape 1)
                                 values may be numpy arrays or torch Tensors

    Returns
    -------
    scalar Tensor  —  MSE of temperature residual at the Dirichlet face,
                      or 0.0 if no Dirichlet nodes are present in the batch
    """
    dev = x_norm.device

    X_mean = torch.as_tensor(norm_params["X_mean"], dtype=torch.float32, device=dev)
    X_std  = torch.as_tensor(norm_params["X_std"],  dtype=torch.float32, device=dev)
    Y_mean = torch.as_tensor(norm_params["Y_mean"], dtype=torch.float32, device=dev).squeeze()
    Y_std  = torch.as_tensor(norm_params["Y_std"],  dtype=torch.float32, device=dev).squeeze()

    # Normalised x-coordinate of the Dirichlet face and its tolerance
    x0_norm  = (_X_MIN_MM - X_mean[0]) / X_std[0]
    tol_norm = _BC_TOL_MM / X_std[0]

    mask = (x_norm[:, 0] - x0_norm).abs() < tol_norm
    if mask.sum() == 0:
        return torch.tensor(0.0, device=dev)

    x_bc   = x_norm[mask]          # (M, 6) — no grad needed
    T_pred = model(x_bc)[:, 0]     # (M,)   normalised temperature

    # Recover physical T_fix from feature col 5, then re-normalise into T-space
    T_fix_phys = x_bc[:, 5].detach() * X_std[5] + X_mean[5]   # °C
    T_fix_norm = (T_fix_phys - Y_mean) / Y_std                  # normalised

    return torch.mean((T_pred - T_fix_norm) ** 2)


def neumann_loss(model, x_norm, norm_params):
    """
    Neumann BC loss: ∂T̂/∂x|_{x=L} = q / (k · A)

    Selects nodes near x=L (the heat-flux face) from the batch and penalises
    the deviation of the predicted temperature gradient from the analytical
    Neumann condition.  Requires one first-order autograd pass (∂T̂/∂x only).

    The gradient is computed in normalised space and converted to physical
    units for a scale-consistent residual in [°C/mm]:

        dT/dx_phys = dT̂/dx̂  ·  (Y_std / X_std[0])

    Parameters
    ----------
    model       : ThermalNet
    x_norm      : Tensor (N, 6)  normalised inputs
    norm_params : dict           keys X_mean, X_std (shape 6), Y_mean, Y_std (shape 1)

    Returns
    -------
    scalar Tensor  —  MSE of gradient residual at the Neumann face in (°C/mm)²,
                      or 0.0 if no Neumann nodes are present in the batch
    """
    dev = x_norm.device

    X_mean = torch.as_tensor(norm_params["X_mean"], dtype=torch.float32, device=dev)
    X_std  = torch.as_tensor(norm_params["X_std"],  dtype=torch.float32, device=dev)
    Y_std  = torch.as_tensor(norm_params["Y_std"],  dtype=torch.float32, device=dev).squeeze()

    # Normalised x-coordinate of the Neumann face and its tolerance
    xL_norm  = (_X_MAX_MM - X_mean[0]) / X_std[0]
    tol_norm = _BC_TOL_MM / X_std[0]

    mask = (x_norm[:, 0] - xL_norm).abs() < tol_norm
    if mask.sum() == 0:
        return torch.tensor(0.0, device=dev)

    x_bc = x_norm[mask].clone().detach().requires_grad_(True)   # (M, 6)
    T_pred = model(x_bc)                                         # (M, 1)

    # ∂T̂/∂x̂  in normalised space
    dT_dx_norm = torch.autograd.grad(
        T_pred.sum(), x_bc, create_graph=True
    )[0][:, 0]                                                    # (M,)

    # Convert to physical gradient [°C/mm]
    dT_dx_phys = dT_dx_norm * Y_std / X_std[0]

    # Recover physical q [mW]: col 3 is log(q), z-scored
    log_q_phys = x_bc[:, 3].detach() * X_std[3] + X_mean[3]
    q_phys = torch.exp(log_q_phys)                               # mW

    # Recover physical k [mW/mm/°C]: col 4, z-scored
    k_phys = x_bc[:, 4].detach() * X_std[4] + X_mean[4]        # mW/mm/°C

    # Analytical target: q / (k · A)  [°C/mm]
    target = q_phys / (k_phys * BEAM_AREA_MM2)

    return torch.mean((dT_dx_phys - target) ** 2)
