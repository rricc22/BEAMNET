"""
Train a Physics-Informed Neural Network (PINN) for beam structural response.

Physics: Navier-Cauchy linear elasticity equilibrium equations
  (λ + μ) ∇(∇·u) + μ ∇²u = 0   (no body forces in interior)

Data flow
---------
  ccx_cases/elasticity_axial_beam/*/job.vtu          – node coordinates + displacement fields
  ccx_cases/elasticity_axial_beam/*/case_params.json – force, direction, material (E_MPa, nu)

Saves
-----
  saves/beam_pinn.pt    – trained PINN weights
  saves/norm_params.npz – normalisation statistics (for inference)

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
from arch import BeamNet
from losses import physics_loss
from norm import (
    build_features, compute_norm_params,
    normalise_X, normalise_Y_dir, denormalise_Y_dir,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CCX_CASES = ROOT / "ccx_cases" / "elasticity_axial_beam"
SAVES = ROOT / "saves"
VTK_MANIFEST = CCX_CASES / "vtk_manifest.json"

# ── PINN hyper-parameters ─────────────────────────────────────────────────────
CONFIG = {
    "hidden": 512,
    "epochs": 20,
    "batch_size": 16384,
    "phys_batch_size": 4096,
    "lr": 1e-3,
    "lambda_physics": 10.0,
    "step_size": 5,
    "gamma": 0.5,
    "weight_decay": 1e-4,
    "data_only": False,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_val_case_ids(vtk_manifest: dict) -> set:
    """
    From the 1500 train cases (100 F × 5 materials × 3 directions), pick every
    5th force per material+direction (sorted by F) as validation:
      → 20 val cases per group × 15 groups = 300 val
      → 80 train cases per group            = 1200 train
    Test cases (75) are untouched.
    """
    from collections import defaultdict
    by_group = defaultdict(list)
    for entry in vtk_manifest["cases"]:
        if entry.get("success") and entry.get("split") == "train":
            params = json.loads(
                (CCX_CASES / entry["vtu"]).parent.joinpath("case_params.json").read_text()
            )
            key = (params["material"], params["load_dir"])
            by_group[key].append((params["force_N"], entry["case_id"]))

    val_ids = set()
    for cases in by_group.values():
        cases.sort(key=lambda x: x[0])
        for i, (_, case_id) in enumerate(cases):
            if (i + 1) % 5 == 0:
                val_ids.add(case_id)
    return val_ids


def load_dataset():
    """
    Read every successful VTU file and assemble the dataset.

    Feature vector (9 columns):
      0-2  x, y, z        node coordinates           [mm]
      3-5  fx, fy, fz     force direction unit vec   [-]
      6    F               total force magnitude      [N]
      7    E               Young's modulus            [MPa]
      8    nu              Poisson's ratio            [-]

    Split strategy
    --------------
      1200 train  (80 F/group × 15 groups)
       300 val    (20 F/group × 15 groups, spread across F range)
        75 test   (untouched — extrapolation benchmark)

    Returns
    -------
    X      : float32 (N_total, 9)
    Y      : float32 (N_total, 3)  displacements [U1, U2, U3] mm
    splits : str array (N_total,)  'train' | 'val' | 'test'
    """
    vtk_manifest = json.loads(VTK_MANIFEST.read_text())
    val_case_ids = _build_val_case_ids(vtk_manifest)

    X_list, Y_list, splits = [], [], []

    for entry in vtk_manifest["cases"]:
        if not entry.get("success"):
            continue

        vtu_path = CCX_CASES / entry["vtu"]
        params = json.loads((vtu_path.parent / "case_params.json").read_text())

        split = "val" if entry["case_id"] in val_case_ids else params["split"]

        mesh = meshio.read(str(vtu_path))
        coords = mesh.points.astype(np.float32)  # (N, 3)
        disp = mesh.point_data["displacement"].astype(np.float32)  # (N, 3)

        X = build_features(coords, params)  # (N, 9) — see norm.py
        X_list.append(X)
        Y_list.append(disp)
        splits.extend([split] * len(coords))

    if not X_list:
        print("ERROR: no data loaded — run scripts 01-05 first")
        sys.exit(1)

    return np.vstack(X_list), np.vstack(Y_list), np.array(splits)


# ---------------------------------------------------------------------------
# Normalisation helpers live in src/norm.py
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    SAVES.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if CONFIG["data_only"]:
        run_name = f"h{CONFIG['hidden']}-data_only-e{CONFIG['epochs']}"
        model_path = SAVES / f"beam_pinn_{run_name}.pt"
    else:
        run_name = f"h{CONFIG['hidden']}-lp{CONFIG['lambda_physics']}-e{CONFIG['epochs']}"
        model_path = SAVES / "beam_pinn.pt"

    if WANDB_AVAILABLE and _wandb is not None:
        _wandb.init(
            project="beam-axial-pinn",
            name=run_name,
            config=CONFIG,
        )

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading dataset …")
    X, Y, splits = load_dataset()
    print(f"  Total nodes : {len(X):,}")
    print(f"  Train nodes : {(splits == 'train').sum():,}")
    print(f"  Val   nodes : {(splits == 'val').sum():,}")
    print(f"  Test  nodes : {(splits == 'test').sum():,}")

    X_tr, Y_tr = X[splits == "train"], Y[splits == "train"]
    X_va, Y_va = X[splits == "val"],   Y[splits == "val"]

    # ── Normalise ─────────────────────────────────────────────────────────
    norm = compute_norm_params(X_tr, Y_tr)
    X_tr_n = normalise_X(X_tr, norm)
    Y_tr_n = normalise_Y_dir(Y_tr, X_tr, norm)   # direction-conditioned
    X_va_n = normalise_X(X_va, norm)

    np.savez(SAVES / "norm_params.npz", **norm)
    print("  Norm params → saves/norm_params.npz")
    print(f"  Y_std (global) : {norm['Y_std']}")
    print(f"  Y_std_Z        : {norm['Y_std_Z']}")
    print(f"  Y_std_Xp       : {norm['Y_std_Xp']}")
    print(f"  Y_std_Xm       : {norm['Y_std_Xm']}")


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

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    model = BeamNet(hidden=CONFIG["hidden"]).to(device)
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
        f"\nTraining BeamNet {mode_str}  (in=9, hidden={CONFIG['hidden']}, "
        f"epochs={CONFIG['epochs']}, device={device})"
    )
    print(f"  Parameters  : {n_params:,}")
    if not CONFIG["data_only"]:
        print(f"  λ_physics   : {CONFIG['lambda_physics']}")
    if CONFIG["data_only"]:
        print(
            f"\n{'Epoch':>6}  {'Data MSE':>10}  "
            f"{'Val MSE':>10}  {'MAE U1':>10}  {'MAE U2':>10}  {'MAE U3':>10}  [mm]"
        )
        print("-" * 68)
    else:
        print(
            f"\n{'Epoch':>6}  {'Data MSE':>10}  {'Phys loss':>10}  "
            f"{'Val MSE':>10}  {'MAE U1':>10}  {'MAE U2':>10}  {'MAE U3':>10}  [mm]"
        )
        print("-" * 80)

    best_val = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        n_batches = len(train_loader)
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss_data = criterion(pred, yb)
            loss_phys = torch.tensor(0.0)

            if CONFIG["data_only"]:
                loss = loss_data
            else:
                loss_phys = physics_loss(model, xb[:phys_bs].detach())
                loss = loss_data + CONFIG["lambda_physics"] * loss_phys
                epoch_phys_loss += loss_phys.item()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_data_loss += loss_data.item() * len(xb)

            if (batch_idx + 1) % 200 == 0:
                if CONFIG["data_only"]:
                    print(
                        f"  ep {epoch}  batch {batch_idx+1}/{n_batches}"
                        f"  data={loss_data.item():.5f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  ep {epoch}  batch {batch_idx+1}/{n_batches}"
                        f"  data={loss_data.item():.5f}  phys={loss_phys.item():.5f}",
                        flush=True,
                    )

        epoch_data_loss /= len(X_tr_n)
        epoch_phys_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            pred_va = model(X_va_t)
            # Denormalise using per-direction Y_std (X_va is raw features)
            pred_mm = denormalise_Y_dir(pred_va.cpu().numpy(), X_va, norm)
            val_mse = np.mean((pred_mm - Y_va) ** 2)
            mae_per_comp = np.abs(pred_mm - Y_va).mean(axis=0)  # [U1, U2, U3]
            val_mae = mae_per_comp.mean()

        scheduler.step()

        if WANDB_AVAILABLE and _wandb is not None:
            log_dict = {
                "epoch": epoch,
                "data_loss": epoch_data_loss,
                "val_mse": val_mse,
                "val_mae_mm": val_mae,
                "val_mae_U1_mm": mae_per_comp[0],
                "val_mae_U2_mm": mae_per_comp[1],
                "val_mae_U3_mm": mae_per_comp[2],
                "lr": scheduler.get_last_lr()[0],
            }
            if not CONFIG["data_only"]:
                log_dict["phys_loss"] = epoch_phys_loss
                log_dict["weighted_phys_loss"] = CONFIG["lambda_physics"] * epoch_phys_loss
                log_dict["total_loss"] = epoch_data_loss + CONFIG["lambda_physics"] * epoch_phys_loss
            _wandb.log(log_dict)

        if epoch % 10 == 0 or epoch == 1:
            if CONFIG["data_only"]:
                print(
                    f"{epoch:6d}  {epoch_data_loss:10.6f}"
                    f"  {val_mse:10.6f}  "
                    f"MAE U1={mae_per_comp[0]:.4f} U2={mae_per_comp[1]:.4f} U3={mae_per_comp[2]:.4f} mm"
                )
            else:
                print(
                    f"{epoch:6d}  {epoch_data_loss:10.6f}  {epoch_phys_loss:10.6f}"
                    f"  {val_mse:10.6f}  "
                    f"MAE U1={mae_per_comp[0]:.4f} U2={mae_per_comp[1]:.4f} U3={mae_per_comp[2]:.4f} mm"
                )

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": CONFIG["hidden"],
                    "config": CONFIG,
                },
                model_path,
            )

    print(f"\nBest val MSE : {best_val:.6f}")
    print(f"Model saved  → {model_path.relative_to(ROOT)}")

    if WANDB_AVAILABLE and _wandb is not None:
        _wandb.save(str(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BeamNet PINN")
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--lambda_physics", type=float, default=None)
    parser.add_argument("--hidden",         type=int,   default=None)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--data_only",      action="store_true",
                        help="Ablation: train with data loss only (no physics loss)")
    args = parser.parse_args()

    for key, val in vars(args).items():
        if val is not None:
            CONFIG[key] = val

    train()
