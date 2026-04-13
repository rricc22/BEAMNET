"""
Train a Physics-Informed Neural Network (PINN) for beam steady-state thermal response.

Physics: steady-state heat conduction Laplacian
  ∇²T = 0   (no internal heat generation)

Data flow
---------
  elmer_cases/thermal_ccx_beam/*/case.vtk          – node coordinates + temperature field
  elmer_cases/thermal_ccx_beam/*/case_params.json  – q_total_mW, k, T_fix_C, material

Saves
-----
  saves/thermal_pinn.pt    – trained PINN weights
  saves/norm_params.npz    – normalisation statistics (for inference)

Run with:  python3 src/train.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import meshio
except ImportError:
    print("ERROR: meshio not installed — pip install meshio")
    sys.exit(1)

try:
    import wandb as _wandb

    WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from arch import ThermalNet
from losses import physics_loss, dirichlet_loss, neumann_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ELMER_CASES = ROOT / "elmer_cases" / "thermal_ccx_beam"
SAVES = ROOT / "saves"
VTK_MANIFEST = ELMER_CASES / "vtk_manifest.json"

# ── Conductivity lookup (matches script 03) ───────────────────────────────────
MATERIAL_K = {
    "Steel_A36": 50.0,
    "Steel_S355": 48.0,
    "Aluminium_6061": 167.0,
    "Titanium_Ti6Al4V": 6.7,
    "Concrete_C30": 1.8,
}

# ── PINN hyper-parameters ─────────────────────────────────────────────────────
CONFIG = {
    "hidden": 512,
    "epochs": 50,
    "batch_size": 16384,
    "phys_batch_size": 4096,
    "lr": 1e-3,
    "lambda_physics": 1.0,
    "lambda_dir": 1.0,
    "lambda_neu": 1.0,
    "step_size": 10,
    "gamma": 0.5,
    "weight_decay": 1e-4,
    "data_only": False,
}

# Column 3 (q_total_mW) spans orders of magnitude → log-transform
LOG_COLS = [3]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_val_case_ids(vtk_manifest: dict) -> set:
    """
    From the 500 train cases (100 Q × 5 materials), pick every 5th Q per
    material (sorted by Q) as validation → 20 val cases per material = 100 total.
    Remaining 80 per material = 400 train. Test cases are untouched.
    """
    # Collect successful train cases grouped by material, sorted by Q
    from collections import defaultdict
    by_material = defaultdict(list)
    for entry in vtk_manifest["cases"]:
        if entry.get("success") and entry.get("split") == "train":
            by_material[entry["material"]].append(
                (entry["q_total_mW"], entry["case_id"])
            )
    val_ids = set()
    for cases in by_material.values():
        cases.sort(key=lambda x: x[0])  # sort by Q
        for i, (_, case_id) in enumerate(cases):
            if (i + 1) % 5 == 0:  # every 5th → val
                val_ids.add(case_id)
    return val_ids


def load_dataset():
    """
    Read every successful VTK file and assemble the dataset.

    Feature vector (6 columns):
      0-2  x, y, z        node coordinates               [mm]
      3    q_total_mW      total applied heat flux        [mW]
      4    k               thermal conductivity       [mW/mm/°C]
      5    T_fix_C         Dirichlet temperature          [°C]

    Split strategy
    --------------
    The original manifest has 500 train + 25 test cases.
    Every 5th Q value per material (sorted) is reassigned to 'val':
      → 400 train  (80 Q/material × 5 materials)
      → 100 val    (20 Q/material × 5 materials, spread across Q range)
      →  25 test   (untouched — extrapolation benchmark)

    Returns
    -------
    X         : float32 (N_total, 6)
    Y         : float32 (N_total, 1)  temperature [°C]
    splits    : str array (N_total,)  'train' | 'val' | 'test'
    materials : str array (N_total,)  material name per node
    """
    vtk_manifest = json.loads(VTK_MANIFEST.read_text())
    val_case_ids = _build_val_case_ids(vtk_manifest)

    X_list, Y_list, splits, materials = [], [], [], []

    for entry in vtk_manifest["cases"]:
        if not entry.get("success"):
            continue

        vtk_rel = entry.get("vtk", "")
        if not vtk_rel:
            continue

        vtk_path = ELMER_CASES / vtk_rel
        params_path = vtk_path.parent / "case_params.json"
        if not vtk_path.exists() or not params_path.exists():
            continue

        params = json.loads(params_path.read_text())

        # Reassign val cases (overrides manifest split for selected train cases)
        split = "val" if params["case_id"] in val_case_ids else params["split"]

        mesh = meshio.read(str(vtk_path))
        coords = mesh.points.astype(np.float32)  # (N, 3)

        # Temperature field key (normalised to lower-case in script 04)
        temp_key = None
        for k in mesh.point_data:
            if k.lower() == "temperature":
                temp_key = k
                break
        if temp_key is None:
            continue

        T_field = mesh.point_data[temp_key].astype(np.float32).reshape(-1, 1)
        N = len(coords)

        k_val = MATERIAL_K.get(params["material"], params.get("k_mW_mm_C", 1.0))
        case_feats = np.array(
            [params["q_total_mW"], k_val, params["T_fix_C"]],
            dtype=np.float32,
        )  # (3,)

        X = np.hstack([coords, np.tile(case_feats, (N, 1))])  # (N, 6)
        X_list.append(X)
        Y_list.append(T_field)
        splits.extend([split] * N)
        materials.extend([params["material"]] * N)

    if not X_list:
        print("ERROR: no data loaded — run scripts 01–04 first")
        sys.exit(1)

    return np.vstack(X_list), np.vstack(Y_list), np.array(splits), np.array(materials)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def compute_norm_params(X, Y):
    X_p = X.copy()
    for c in LOG_COLS:
        X_p[:, c] = np.log(X_p[:, c])
    return {
        "X_mean": X_p.mean(axis=0),
        "X_std": X_p.std(axis=0) + 1e-8,
        "Y_mean": Y.mean(axis=0),
        "Y_std": Y.std(axis=0) + 1e-8,
    }


def apply_norm(X, Y, norm):
    X_p = X.copy()
    for c in LOG_COLS:
        X_p[:, c] = np.log(X_p[:, c])
    X_n = (X_p - norm["X_mean"]) / norm["X_std"]
    Y_n = (Y - norm["Y_mean"]) / norm["Y_std"]
    return X_n.astype(np.float32), Y_n.astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    SAVES.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if CONFIG["data_only"]:
        run_name = f"h{CONFIG['hidden']}-data_only-e{CONFIG['epochs']}"
    else:
        run_name = (
            f"h{CONFIG['hidden']}"
            f"-lp{CONFIG['lambda_physics']}"
            f"-ld{CONFIG['lambda_dir']}"
            f"-ln{CONFIG['lambda_neu']}"
            f"-e{CONFIG['epochs']}"
        )
    MODEL_PATH = SAVES / f"thermal_pinn_{run_name}.pt"

    if WANDB_AVAILABLE and _wandb is not None:
        _wandb.init(
            project="beam-thermal-pinn",
            name=run_name,
            config=CONFIG,
        )

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading dataset …")
    X, Y, splits, materials = load_dataset()
    print(f"  Total nodes : {len(X):,}")
    print(f"  Train nodes : {(splits == 'train').sum():,}")
    print(f"  Val   nodes : {(splits == 'val').sum():,}")
    print(f"  Test  nodes : {(splits == 'test').sum():,}")

    X_tr, Y_tr = X[splits == "train"], Y[splits == "train"]
    X_va, Y_va = X[splits == "val"],   Y[splits == "val"]
    mat_va     = materials[splits == "val"]

    # ── Normalise ─────────────────────────────────────────────────────────
    norm = compute_norm_params(X_tr, Y_tr)
    X_tr_n, Y_tr_n = apply_norm(X_tr, Y_tr, norm)
    X_va_n, Y_va_n = apply_norm(X_va, Y_va, norm)
    print(f"  Y_mean={norm['Y_mean'].item():.2f}°C  Y_std={norm['Y_std'].item():.2f}°C")

    np.savez(SAVES / "norm_params.npz", **norm)
    print("  Norm params → saves/norm_params.npz")

    # ── DataLoaders ───────────────────────────────────────────────────────
    t = lambda a: torch.from_numpy(a)
    train_loader = DataLoader(
        TensorDataset(t(X_tr_n), t(Y_tr_n)),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    X_va_t = t(X_va_n).to(device)
    Y_va_t = t(Y_va_n).to(device)

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    model = ThermalNet(hidden=CONFIG["hidden"]).to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=CONFIG["step_size"], gamma=CONFIG["gamma"]
    )
    criterion = nn.MSELoss()

    phys_bs = CONFIG["phys_batch_size"]
    n_params = sum(q.numel() for q in model.parameters())
    mode_str = "data-only ablation" if CONFIG["data_only"] else "PINN"
    print(
        f"\nTraining ThermalNet {mode_str}  (in=6, hidden={CONFIG['hidden']}, "
        f"epochs={CONFIG['epochs']}, device={device})"
    )
    print(f"  Parameters  : {n_params:,}")
    if not CONFIG["data_only"]:
        print(f"  λ_physics   : {CONFIG['lambda_physics']}")
        print(f"  λ_dir       : {CONFIG['lambda_dir']}")
        print(f"  λ_neu       : {CONFIG['lambda_neu']}")
    if CONFIG["data_only"]:
        print(f"\n{'Epoch':>6}  {'Data MSE':>10}  {'Val MAE[°C]':>12}")
        print("-" * 34)
    else:
        print(
            f"\n{'Epoch':>6}  {'Data MSE':>10}  {'Phys':>10}  "
            f"{'Dir':>10}  {'Neu':>10}  {'Val MAE[°C]':>12}"
        )
        print("-" * 74)

    best_val = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_dir_loss  = 0.0
        epoch_neu_loss  = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss_data = criterion(pred, yb)

            if CONFIG["data_only"]:
                loss = loss_data
            else:
                loss_phys = physics_loss(model, xb[:phys_bs].detach())
                loss_dir  = dirichlet_loss(model, xb[:phys_bs].detach(), norm)
                loss_neu  = neumann_loss(model, xb[:phys_bs].detach(), norm)
                loss = (loss_data
                        + CONFIG["lambda_physics"] * loss_phys
                        + CONFIG["lambda_dir"]     * loss_dir
                        + CONFIG["lambda_neu"]     * loss_neu)
                epoch_phys_loss += loss_phys.item()
                epoch_dir_loss  += loss_dir.item()
                epoch_neu_loss  += loss_neu.item()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_data_loss += loss_data.item() * len(xb)

        epoch_data_loss /= len(X_tr_n)
        epoch_phys_loss /= len(train_loader)
        epoch_dir_loss  /= len(train_loader)
        epoch_neu_loss  /= len(train_loader)

        model.eval()
        with torch.no_grad():
            pred_va = model(X_va_t)
            pred_C  = pred_va.cpu().numpy() * norm["Y_std"] + norm["Y_mean"]
            true_C  = Y_va_t.cpu().numpy()  * norm["Y_std"] + norm["Y_mean"]
            abs_err = np.abs(pred_C - true_C).squeeze()
            val_mae = abs_err.mean()

            # per-material val MAE
            mat_mae = {}
            for mat in MATERIAL_K:
                mask = mat_va == mat
                if mask.any():
                    mat_mae[f"val_mae_{mat}"] = abs_err[mask].mean()

        scheduler.step()

        if WANDB_AVAILABLE and _wandb is not None:
            _wandb.log(
                {
                    "epoch":               epoch,
                    "data_loss":           epoch_data_loss,
                    "weighted_phys_loss":  CONFIG["lambda_physics"] * epoch_phys_loss,
                    "weighted_dir_loss":   CONFIG["lambda_dir"]     * epoch_dir_loss,
                    "weighted_neu_loss":   CONFIG["lambda_neu"]     * epoch_neu_loss,
                    "total_loss":         (epoch_data_loss
                                          + CONFIG["lambda_physics"] * epoch_phys_loss
                                          + CONFIG["lambda_dir"]     * epoch_dir_loss
                                          + CONFIG["lambda_neu"]     * epoch_neu_loss),
                    "val_mae_C":           val_mae,
                    "lr":                  scheduler.get_last_lr()[0],
                    **mat_mae,
                }
            )

        if epoch % 10 == 0 or epoch == 1:
            if CONFIG["data_only"]:
                print(f"{epoch:6d}  {epoch_data_loss:10.6f}  {val_mae:12.6f}")
            else:
                print(
                    f"{epoch:6d}  {epoch_data_loss:10.6f}  {epoch_phys_loss:10.6f}"
                    f"  {epoch_dir_loss:10.6f}  {epoch_neu_loss:10.6f}  {val_mae:12.6f}"
                )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": CONFIG["hidden"],
                    "config": CONFIG,
                },
                MODEL_PATH,
            )

    print(f"\nBest val MAE : {best_val:.6f} °C")
    print(f"Model saved  → {MODEL_PATH.relative_to(ROOT)}")

    if WANDB_AVAILABLE and _wandb is not None:
        _wandb.save(str(MODEL_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ThermalNet PINN")
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--lambda_physics",  type=float, default=None)
    parser.add_argument("--lambda_dir",      type=float, default=None)
    parser.add_argument("--lambda_neu",      type=float, default=None)
    parser.add_argument("--hidden",          type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--data_only",       action="store_true",
                        help="Ablation: train with data loss only (no physics/BC losses)")
    args = parser.parse_args()

    for key, val in vars(args).items():
        if val is not None:
            CONFIG[key] = val

    train()
