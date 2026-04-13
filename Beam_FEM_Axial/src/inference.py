"""
Run the trained BeamNet on test cases: print metrics, save PNG plots,
and optionally write comparison VTU files for ParaView.

Usage
-----
  python3 src/inference.py                        # metrics + PNG for all test cases
  python3 src/inference.py --vtu                  # also write comparison VTU files
  python3 src/inference.py --case case_0100       # single case, metrics + PNG
  python3 src/inference.py --case case_0100 --vtu # single case, all outputs

Outputs
-------
  saves/prediction_<case_id>.png       – 3-panel comparison figure
  saves/<case_id>_comparison.vtu       – ParaView file (only with --vtu)
  Prints MAE, RMSE, relative error to stdout
"""

import argparse
import json
import sys
from datetime import datetime
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
from arch import BeamNet
from norm import build_features, normalise_X, denormalise_Y_dir  # normalise_X used inside predict()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CCX_CASES = ROOT / "ccx_cases" / "elasticity_axial_beam"
SAVES = ROOT / "saves"
VTK_MANIFEST = CCX_CASES / "vtk_manifest.json"


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_model(model_path: Path = None):
    ckpt = torch.load(model_path or SAVES / "beam_pinn.pt", map_location="cpu", weights_only=True)
    model = BeamNet(hidden=ckpt["hidden"])
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_norm_params():
    return dict(np.load(SAVES / "norm_params.npz"))


def predict(model, X_raw: np.ndarray, norm: dict) -> np.ndarray:
    """Normalise X, run model, and denormalise output to physical mm."""
    X_n = normalise_X(X_raw, norm)
    with torch.no_grad():
        Y_n = model(torch.from_numpy(X_n)).numpy()
    return denormalise_Y_dir(Y_n, X_raw, norm)  # (N, 3) [mm]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(pred_mm: np.ndarray, true_mm: np.ndarray) -> dict:
    mae = np.abs(pred_mm - true_mm).mean()
    rmse = np.sqrt(((pred_mm - true_mm) ** 2).mean())
    rel_err = (
        np.linalg.norm(pred_mm - true_mm, axis=1).mean()
        / (np.linalg.norm(true_mm, axis=1).mean() + 1e-12)
        * 100
    )
    return {"mae": mae, "rmse": rmse, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------


def save_png(coords, pred_mm, true_mm, case_id, params, out_path):
    pred_mag = np.linalg.norm(pred_mm, axis=1)
    true_mag = np.linalg.norm(true_mm, axis=1)
    err_mag = np.abs(pred_mag - true_mag)

    vmin, vmax = true_mag.min(), true_mag.max()

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        f"{case_id}  |  F={params['force_N'] / 1e3:.1f} kN  "
        f"dir={params['load_dir']}  mat={params['material']}",
        fontsize=11,
    )

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    for col, (data, title, cmap) in enumerate(
        [
            (true_mag, "FEM  (ground truth)", "viridis"),
            (pred_mag, "BeamNet prediction", "viridis"),
            (err_mag, "Absolute error |Δ|", "Reds"),
        ]
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        sc = ax.scatter(
            x,
            y,
            z,
            c=data,
            cmap=cmap,
            vmin=(vmin if col < 2 else 0),
            vmax=(vmax if col < 2 else err_mag.max()),
            s=4,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X [mm]", fontsize=7)
        ax.set_ylabel("Y [mm]", fontsize=7)
        ax.set_zlabel("Z [mm]", fontsize=7)
        plt.colorbar(sc, ax=ax, shrink=0.6, label="|U| [mm]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  PNG  → {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# VTU output
# ---------------------------------------------------------------------------


def save_vtu(fem_mesh, pred_mm, true_mm, case_id, out_path):
    err_mag = np.linalg.norm(pred_mm, axis=1) - np.linalg.norm(true_mm, axis=1)

    point_data = dict(fem_mesh.point_data)
    point_data["displacement_FEM"] = true_mm
    point_data["displacement_pred"] = pred_mm
    point_data["error_magnitude"] = err_mag
    point_data.pop("displacement", None)

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


def run_case(case_id, model, norm_p, vtu_entries, write_vtu: bool = False, out_dir: Path = SAVES):
    entry = vtu_entries[case_id]
    vtu_path = CCX_CASES / entry["vtu"]
    params = json.loads((vtu_path.parent / "case_params.json").read_text())

    fem_mesh = meshio.read(str(vtu_path))
    coords = fem_mesh.points
    true_mm = fem_mesh.point_data["displacement"]

    X_raw = build_features(coords, params)
    pred_mm = predict(model, X_raw, norm_p)

    m = compute_metrics(pred_mm, true_mm)

    print(f"\n  Case         : {case_id}  [{params['split']}]")
    print(
        f"  Force        : {params['force_N'] / 1e3:.1f} kN  dir={params['load_dir']}"
    )
    print(
        f"  Material     : {params['material']}  (E={params['E_MPa']:.0f} MPa, nu={params['nu']})"
    )
    print(f"  MAE          : {m['mae']:.4f} mm")
    print(f"  RMSE         : {m['rmse']:.4f} mm")
    print(f"  Relative err : {m['rel_err']:.2f} %")
    print(f"  Max |U| FEM  : {np.linalg.norm(true_mm, axis=1).max():.4f} mm")
    print(f"  Max |U| pred : {np.linalg.norm(pred_mm, axis=1).max():.4f} mm")

    save_png(
        coords, pred_mm, true_mm, case_id, params, out_dir / f"prediction_{case_id}.png"
    )

    if write_vtu:
        save_vtu(
            fem_mesh, pred_mm, true_mm, case_id, out_dir / f"{case_id}_comparison.vtu"
        )

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="BeamNet inference: metrics, PNG plots, and optional VTU output."
    )
    parser.add_argument(
        "--case",
        default=None,
        help="case_id to evaluate (default: all test cases)",
    )
    parser.add_argument(
        "--vtu",
        action="store_true",
        help="also write <case_id>_comparison.vtu files for ParaView",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="output directory for PNGs and VTUs (default: saves/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="path to .pt model file (default: saves/beam_pinn.pt)",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else SAVES / "beam_pinn.pt"
    if not model_path.exists():
        print(f"ERROR: no trained model found at {model_path}")
        sys.exit(1)

    out_dir = Path(args.outdir) if (args.outdir and Path(args.outdir).is_absolute()) else \
              (ROOT / args.outdir) if args.outdir else \
              SAVES / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    model = load_model(model_path)
    norm_p = load_norm_params()
    SAVES.mkdir(exist_ok=True)

    vtk_manifest = json.loads(VTK_MANIFEST.read_text())
    vtu_entries = {e["case_id"]: e for e in vtk_manifest["cases"] if e.get("success")}

    if args.case:
        target_cases = [args.case]
    else:
        target_cases = [
            e["case_id"] for e in vtk_manifest["cases"] if e.get("split") == "test"
        ]

    print("=" * 55)
    print("BeamNet inference")
    print("=" * 55)

    vtu_files = []
    for cid in target_cases:
        if cid not in vtu_entries:
            print(f"  WARNING: {cid} not found in vtk_manifest")
            continue
        run_case(cid, model, norm_p, vtu_entries, write_vtu=args.vtu, out_dir=out_dir)
        if args.vtu:
            vtu_files.append(out_dir / f"{cid}_comparison.vtu")

    if args.vtu and vtu_files:
        print(f"\nOpen in ParaView:")
        print(f"  paraview {vtu_files[0]}")
        print("\n  → Colour by 'displacement_pred' : model prediction")
        print("  → Colour by 'displacement_FEM'  : FEM ground truth")
        print("  → Colour by 'error_magnitude'   : where errors are largest")

    print("\nDone.")


if __name__ == "__main__":
    main()
