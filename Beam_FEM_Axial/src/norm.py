"""
Normalisation pipeline for BeamNet.

This file is the single source of truth for every data transformation
applied between raw FEM data and the model, in pipeline order:

  RAW DATA
    │
    ▼  Step 1 — build_features()
  Feature matrix  X  (N, 9)
    [x, y, z, fx, fy, fz, force_N, E_MPa, nu]
    │
    ▼  Step 2 — log_transform_X()       log(force_N)  [col 6]
  X_log  (N, 9)
    [x, y, z, fx, fy, fz, log(force_N), E_MPa, nu]
    │
    ▼  Step 3 — compute_norm_params()   fit on training split only
  norm_params  →  X_mean, X_std, Y_mean=0, Y_std (global + per-direction)
    │
    ▼  Step 4 — normalise_X()           z-score
  X_norm  (N, 9)   mean=0, std=1 per column
    │
    ▼  Step 5 — normalise_Y_dir()       divide by per-direction Y_std
  Y_norm  (N, 3)   scaled per load direction, Y_mean = 0
    │
    ▼  MODEL FORWARD PASS
  Y_pred_norm  (N, 3)
    │
    ▼  Step 6 — denormalise_Y_dir()     × Y_std[direction]
  Y_pred_mm  (N, 3)   physical displacements [U1, U2, U3] in mm

Why Y_mean = 0
--------------
Y_mean is set to zero instead of the global training mean.

If Y_mean were computed globally across all load directions, it would be
pulled upward by Z-bending cases (which produce large U3 displacements).
The model's natural resting state (output = 0 in normalised space) would
then denormalise to Y_mean ≠ 0 — a spurious displacement that looks like
Z-load contamination on X+/X- predictions.

With Y_mean = 0, the resting state maps to physical zero displacement,
which is correct for all load directions and all components.

Why direction-conditioned Y_std
--------------------------------
A single global Y_std is dominated by Z-bending cases, where U3 reaches
tens to hundreds of mm.  For X+/X- axial cases, U3 is only Poisson
contraction (~0.001–0.003 mm).  Dividing by the global U3_std (~30 mm)
makes the normalised axial-U3 labels indistinguishably close to zero, so
the MSE loss assigns almost zero gradient to U3 errors on axial cases.

With per-direction Y_std, the U3_std for X+/X- cases is ~0.002 mm, so
a 0.1 mm contamination error becomes ~50 in normalised units — a large
loss that forces the network to suppress spurious bending in axial cases.

The direction (fx, fy, fz) is already present as an input feature, so the
network has all the information it needs to learn direction-dependent scaling.

Why log(force_N)
----------------
Force spans several orders of magnitude in the training set (5 N to 200 kN).
Z-scoring raw force values gives the low-force range a z-score close to the
high-force range, compressing the signal. Log-transforming first spreads the
range uniformly before z-scoring.
"""

import numpy as np

# ── Column indices in the raw feature matrix X (N, 9) ─────────────────────────
# 0-2  : x, y, z          node coordinates  [mm]
# 3-5  : fx, fy, fz        force direction unit vector  [-]
# 6    : force_N           force magnitude   [N]   ← log-transformed before z-score
# 7    : E_MPa             Young's modulus   [MPa]
# 8    : nu                Poisson's ratio   [-]
LOG_COLS = [6]

# Force direction → unit vector (must match scripts/03_*.py and train.py)
DIR_MAP = {
    "Y":  np.array([ 0.0,   1.0,   0.0]),
    "Z":  np.array([ 0.0,   0.0,  -1.0]),   # transverse -Z (downward bending)
    "YZ": np.array([ 0.0,   0.707, 0.707]),
    "X":  np.array([-1.0,   0.0,   0.0]),   # axial compression (legacy key)
    "X+": np.array([ 1.0,   0.0,   0.0]),   # axial tension
    "X-": np.array([-1.0,   0.0,   0.0]),   # axial compression
}

# Mapping from direction label to npz storage key (avoids "+" / "-" in key names)
_DIR_TO_NPZ = {"Z": "Y_std_Z", "X+": "Y_std_Xp", "X-": "Y_std_Xm"}


# ── Direction extraction from raw X ───────────────────────────────────────────

def _dir_from_X_raw(X: np.ndarray) -> np.ndarray:
    """
    Recover the load direction label from raw feature column 3 (fx).

    Raw DIR_MAP values are exactly ±1 or 0, so a simple threshold is enough:
      fx >  0.5  →  "X+"
      fx < -0.5  →  "X-"
      otherwise  →  "Z"   (fz component carries the load; Y-only is treated as Z)

    Returns a str array of shape (N,).
    """
    fx = X[:, 3]
    return np.where(fx > 0.5, "X+", np.where(fx < -0.5, "X-", "Z"))


# ── Step 1 — assemble raw feature matrix ──────────────────────────────────────

def build_features(coords: np.ndarray, params: dict) -> np.ndarray:
    """
    Assemble the (N, 9) raw feature matrix for one simulation case.

    coords : (N, 3) node coordinates [mm]
    params : dict with keys  load_dir, force_N, E_MPa, nu
    """
    N = len(coords)
    force_dir = DIR_MAP[params["load_dir"]]
    case_feats = np.array(
        [*force_dir, params["force_N"], params["E_MPa"], params["nu"]],
        dtype=np.float32,
    )
    return np.hstack([coords.astype(np.float32), np.tile(case_feats, (N, 1))])


# ── Step 2 — log-transform (applied inside normalise_X, kept separate for clarity) ──

def log_transform_X(X: np.ndarray) -> np.ndarray:
    """
    Apply log to the force column (col 6) only.
    Returns a copy — does not modify the input.
    """
    X_log = X.copy()
    for c in LOG_COLS:
        X_log[:, c] = np.log(X_log[:, c])
    return X_log


# ── Step 3 — fit normalisation parameters (training split only) ───────────────

def compute_norm_params(X_train: np.ndarray, Y_train: np.ndarray) -> dict:
    """
    Compute and return normalisation statistics from the TRAINING split only.
    Never call this on validation or test data — that would be data leakage.

    Returns a dict with keys:
      X_mean, X_std           – feature z-score stats
      Y_mean                  – zero (intentional, see module docstring)
      Y_std                   – global displacement std (kept for backward compat)
      Y_std_Z, Y_std_Xp, Y_std_Xm  – per-direction displacement std

    Ready to be saved with np.savez(..., **norm_params).
    """
    X_log = log_transform_X(X_train)
    dir_labels = _dir_from_X_raw(X_train)

    # Per-direction Y_std (fallback to global if a direction has no training data)
    global_std = Y_train.std(axis=0) + 1e-8
    per_dir = {}
    for d, key in _DIR_TO_NPZ.items():
        mask = dir_labels == d
        per_dir[key] = Y_train[mask].std(axis=0) + 1e-8 if mask.any() else global_std

    return {
        "X_mean": X_log.mean(axis=0),
        "X_std":  X_log.std(axis=0) + 1e-8,
        # Y_mean is intentionally zero — see module docstring
        "Y_mean": np.zeros(Y_train.shape[1], dtype=np.float64),
        "Y_std":  global_std,   # kept for backward compatibility
        **per_dir,
    }


# ── Step 4 — normalise features ───────────────────────────────────────────────

def normalise_X(X: np.ndarray, norm: dict) -> np.ndarray:
    """
    Log-transform col 6, then z-score all columns.
    Returns float32 ready for the model.
    """
    X_log = log_transform_X(X)
    return ((X_log - norm["X_mean"]) / norm["X_std"]).astype(np.float32)


# ── Step 5 — normalise labels ─────────────────────────────────────────────────

def normalise_Y(Y: np.ndarray, norm: dict) -> np.ndarray:
    """
    Z-score displacement labels.
    Y_mean = 0, so this is just Y / Y_std.
    """
    return ((Y - norm["Y_mean"]) / norm["Y_std"]).astype(np.float32)


# ── Step 6 — denormalise predictions ──────────────────────────────────────────

def denormalise_Y(Y_norm: np.ndarray, norm: dict) -> np.ndarray:
    """
    Convert model output back to physical displacements [mm].
    Y_mean = 0, so this is just Y_norm * Y_std.
    Uses the global Y_std (kept for backward compatibility).
    """
    return Y_norm * norm["Y_std"] + norm["Y_mean"]


# ── Direction-conditioned variants ────────────────────────────────────────────

def _get_dir_Y_std(d: str, norm: dict) -> np.ndarray:
    """Return per-direction Y_std, falling back to global if key not found."""
    key = _DIR_TO_NPZ.get(d)
    if key and key in norm:
        return norm[key]
    return norm["Y_std"]


def normalise_Y_dir(Y: np.ndarray, X_raw: np.ndarray, norm: dict) -> np.ndarray:
    """
    Normalise displacement labels using per-direction Y_std.

    Each sample is divided by the Y_std computed on training samples with the
    same load direction.  This gives the loss equal sensitivity to the true
    signal magnitude in each direction, preventing large-displacement Z-bending
    cases from drowning out the tiny Poisson-contraction signal in axial cases.

    Y     : (N, 3) raw displacements [mm]
    X_raw : (N, 9) raw feature matrix (before any log-transform or z-score)
    norm  : dict from compute_norm_params()
    """
    dirs = _dir_from_X_raw(X_raw)
    Y_norm = np.empty_like(Y, dtype=np.float32)
    for d in np.unique(dirs):
        mask = dirs == d
        Y_norm[mask] = (Y[mask] / _get_dir_Y_std(d, norm)).astype(np.float32)
    return Y_norm


def denormalise_Y_dir(Y_norm: np.ndarray, X_raw: np.ndarray, norm: dict) -> np.ndarray:
    """
    Convert direction-conditioned normalised model output to physical mm.

    Y_norm : (N, 3) normalised model output
    X_raw  : (N, 9) raw feature matrix (used only to extract direction labels)
    norm   : dict from compute_norm_params()
    """
    dirs = _dir_from_X_raw(X_raw)
    Y_mm = np.empty_like(Y_norm, dtype=np.float64)
    for d in np.unique(dirs):
        mask = dirs == d
        Y_mm[mask] = Y_norm[mask] * _get_dir_Y_std(d, norm)
    return Y_mm
