"""
Run the trained ThermalNet on test cases: print metrics, save PNG plots,
and optionally write comparison VTU files for ParaView.

Usage
-----
  python3 src/inference.py                        # metrics + PNG for all test cases
  python3 src/inference.py --vtu                  # also write comparison VTU files
  python3 src/inference.py --case case_0500       # single case, metrics + PNG
  python3 src/inference.py --case case_0500 --vtu # single case, all outputs

Outputs
-------
  saves/prediction_<case_id>.png       – 3-panel comparison figure
  saves/<case_id>_comparison.vtu       – ParaView file (only with --vtu)
  Prints MAE, RMSE, relative error to stdout
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import meshio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from arch import ThermalNet

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ELMER_CASES = ROOT / "elmer_cases" / "thermal_ccx_beam"
SAVES = ROOT / "saves"
VTK_MANIFEST = ELMER_CASES / "vtk_manifest.json"

# Column 3 = log(q) — must match train.py
LOG_COLS = [3]

# Conductivity lookup (matches train.py / script 03)
MATERIAL_K = {
    "Steel_A36": 50.0,
    "Steel_S355": 48.0,
    "Aluminium_6061": 167.0,
    "Titanium_Ti6Al4V": 6.7,
    "Concrete_C30": 1.8,
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_model(model_path=None):
    path = model_path or (SAVES / "thermal_pinn.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ThermalNet(hidden=ckpt["hidden"])
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_norm_params():
    # norm_params.npz always lives in the root saves/, not the per-model outdir
    return dict(np.load(ROOT / "saves" / "norm_params.npz"))


def build_features(coords: np.ndarray, params: dict) -> np.ndarray:
    """Assemble (N, 6) feature matrix for one case."""
    N = len(coords)
    k_val = MATERIAL_K.get(params["material"], params.get("k_mW_mm_C", 1.0))
    case_feats = np.array(
        [params["q_total_mW"], k_val, params["T_fix_C"]],
        dtype=np.float32,
    )
    return np.hstack([coords.astype(np.float32), np.tile(case_feats, (N, 1))])


def normalise_X(X: np.ndarray, p: dict) -> np.ndarray:
    X_p = X.copy()
    for c in LOG_COLS:
        X_p[:, c] = np.log(X_p[:, c])
    return ((X_p - p["X_mean"]) / p["X_std"]).astype(np.float32)


def predict(model, X_n: np.ndarray, p: dict) -> np.ndarray:
    """Run model and denormalise output to °C. Returns (N,) array."""
    with torch.no_grad():
        Y_n = model(torch.from_numpy(X_n)).numpy()  # (N, 1)
    T = (Y_n * p["Y_std"] + p["Y_mean"]).squeeze()  # (N,)
    return T


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(pred_C: np.ndarray, true_C: np.ndarray) -> dict:
    diff = pred_C - true_C
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff**2).mean())
    denom = np.abs(true_C - true_C.min()).mean() + 1e-12  # range-normalised
    rel_err = np.abs(diff).mean() / denom * 100
    return {"mae": mae, "rmse": rmse, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------


def save_png(coords, pred_C, true_C, case_id, params, out_path):
    err = np.abs(pred_C - true_C)
    vmin, vmax = true_C.min(), true_C.max()

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        f"{case_id}  |  Q={params['q_total_mW'] / 1000:.1f} W  "
        f"mat={params['material']}  k={MATERIAL_K.get(params['material'], '?')} W/m/K",
        fontsize=11,
    )

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    for col, (data, title, cmap, v0, v1) in enumerate(
        [
            (true_C, "FEM  (ground truth)", "plasma", vmin, vmax),
            (pred_C, "ThermalNet prediction", "plasma", vmin, vmax),
            (err, "Absolute error |ΔT|", "Reds", 0, err.max()),
        ]
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        sc = ax.scatter(x, y, z, c=data, cmap=cmap, vmin=v0, vmax=v1, s=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X [mm]", fontsize=7)
        ax.set_ylabel("Y [mm]", fontsize=7)
        ax.set_zlabel("Z [mm]", fontsize=7)
        label = "T [°C]" if col < 2 else "|ΔT| [°C]"
        plt.colorbar(sc, ax=ax, shrink=0.6, label=label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  PNG  → {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# VTU output
# ---------------------------------------------------------------------------


def save_vtu(fem_mesh, pred_C, true_C, case_id, out_path):
    err = pred_C - true_C  # signed error (°C)

    point_data = dict(fem_mesh.point_data)
    point_data["temperature_FEM"] = true_C
    point_data["temperature_pred"] = pred_C
    point_data["error_C"] = err
    # Remove the raw temperature fields to avoid confusion
    for k in list(point_data.keys()):
        if k.lower() == "temperature" and k not in (
            "temperature_FEM",
            "temperature_pred",
        ):
            point_data.pop(k)

    out_mesh = meshio.Mesh(
        points=fem_mesh.points,
        cells=fem_mesh.cells,
        point_data=point_data,
    )
    meshio.write(str(out_path), out_mesh)
    print(f"  VTU  → {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------


def run_case(case_id, model, norm_p, vtk_entries, write_vtu: bool = False):
    entry = vtk_entries[case_id]
    vtk_rel = entry.get("vtk", "")
    if not vtk_rel:
        print(f"  WARNING: no VTK path in manifest for {case_id}")
        return None

    vtk_path = ELMER_CASES / vtk_rel
    params_path = vtk_path.parent / "case_params.json"
    params = json.loads(params_path.read_text())

    fem_mesh = meshio.read(str(vtk_path))
    coords = fem_mesh.points

    # Temperature ground truth
    temp_key = None
    for k in fem_mesh.point_data:
        if k.lower() == "temperature":
            temp_key = k
            break
    if temp_key is None:
        print(f"  WARNING: no temperature field in {vtk_path.name}")
        return None
    true_C = fem_mesh.point_data[temp_key].squeeze()

    X_n = normalise_X(build_features(coords, params), norm_p)
    pred_C = predict(model, X_n, norm_p)

    m = compute_metrics(pred_C, true_C)

    k_val = MATERIAL_K.get(params["material"], params.get("k_mW_mm_C", "?"))
    print(f"\n  Case         : {case_id}  [{params['split']}]")
    print(f"  Q_total      : {params['q_total_mW'] / 1000:.2f} W")
    print(f"  Material     : {params['material']}  (k={k_val} W/m/K)")
    print(f"  T_fix        : {params['T_fix_C']:.1f} °C")
    print(f"  MAE          : {m['mae']:.4f} °C")
    print(f"  RMSE         : {m['rmse']:.4f} °C")
    print(f"  Relative err : {m['rel_err']:.2f} %")
    print(f"  Max T FEM    : {true_C.max():.4f} °C")
    print(f"  Max T pred   : {pred_C.max():.4f} °C")

    save_png(
        coords, pred_C, true_C, case_id, params, SAVES / f"prediction_{case_id}.png"
    )

    if write_vtu:
        save_vtu(fem_mesh, pred_C, true_C, case_id, SAVES / f"{case_id}_comparison.vtu")

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global SAVES
    parser = argparse.ArgumentParser(
        description="ThermalNet inference: metrics, PNG plots, and optional VTU output."
    )
    parser.add_argument("--case",    default=None, help="case_id to evaluate (default: all test cases)")
    parser.add_argument("--vtu",     action="store_true", help="also write comparison VTU files")
    parser.add_argument("--model",   default=None, help="path to .pt model file (default: saves/thermal_pinn.pt)")
    parser.add_argument("--outdir",  default=None, help="output directory for plots (default: saves/)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else SAVES / "thermal_pinn.pt"
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    if args.outdir:
        SAVES = Path(args.outdir)
    SAVES.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    norm_p = load_norm_params()

    vtk_manifest = json.loads(VTK_MANIFEST.read_text())
    vtk_entries = {e["case_id"]: e for e in vtk_manifest["cases"] if e.get("success")}

    if args.case:
        target_cases = [args.case]
    else:
        target_cases = [
            e["case_id"] for e in vtk_manifest["cases"] if e.get("split") == "test"
        ]

    print("=" * 55)
    print("ThermalNet inference")
    print("=" * 55)

    vtu_files = []
    for cid in target_cases:
        if cid not in vtk_entries:
            print(f"  WARNING: {cid} not found in vtk_manifest")
            continue
        run_case(cid, model, norm_p, vtk_entries, write_vtu=args.vtu)
        if args.vtu:
            vtu_files.append(SAVES / f"{cid}_comparison.vtu")

    if args.vtu and vtu_files:
        print(f"\nOpen in ParaView:")
        print(f"  paraview {vtu_files[0]}")
        print("\n  → Colour by 'temperature_pred' : model prediction")
        print("  → Colour by 'temperature_FEM'  : FEM ground truth")
        print("  → Colour by 'error_C'           : signed error (pred − FEM)")

    print("\nDone.")


if __name__ == "__main__":
    main()
