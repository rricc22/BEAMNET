"""
Visualize BeamNet inference results on the test set.

Does NOT modify inference.py — runs its own forward pass.

Produces (all saved to saves/):
  results_summary.txt     – ASCII table per case + aggregate stats by regime/direction
  error_vs_force.png      – MAE [mm] & rel-err [%] vs Force [kN], one line per material,
                            separated panels for each load direction, training-range band
  scatter_maxU.png        – predicted vs FEM max-displacement scatter, coloured by material,
                            marker shape = load direction
  heatmap_rel_err.png     – material × force-level heatmap, one panel per load direction

Usage:
  python3 scripts/visualize_results.py
"""

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
    import matplotlib.colors as mcolors
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arch import BeamNet
from norm import build_features, normalise_X, denormalise_Y_dir

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CCX_CASES = ROOT / "ccx_cases" / "elasticity_axial_beam"
SAVES = ROOT / "saves"
VTK_MANIFEST = CCX_CASES / "vtk_manifest.json"

MATERIAL_ORDER = [
    "Steel_A36",
    "Steel_S355",
    "Aluminium_6061",
    "Titanium_Ti6Al4V",
    "Concrete_C30",
]

DIR_ORDER = ["Z", "X+", "X-"]

# Training force range (N)
F_TRAIN_MIN_N = 5_000.0
F_TRAIN_MAX_N = 200_000.0


# ── Model helpers ──────────────────────────────────────────────────────────────


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


def predict(model, X_raw, norm):
    X_n = normalise_X(X_raw, norm)
    with torch.no_grad():
        Y_n = model(torch.from_numpy(X_n)).numpy()
    return denormalise_Y_dir(Y_n, X_raw, norm)  # (N, 3) mm


def compute_metrics(pred_mm, true_mm):
    mae = float(np.abs(pred_mm - true_mm).mean())
    rmse = float(np.sqrt(((pred_mm - true_mm) ** 2).mean()))
    rel = (
        np.linalg.norm(pred_mm - true_mm, axis=1).mean()
        / (np.linalg.norm(true_mm, axis=1).mean() + 1e-12)
        * 100.0
    )
    max_pred = float(np.linalg.norm(pred_mm, axis=1).max())
    max_true = float(np.linalg.norm(true_mm, axis=1).max())
    return dict(
        mae=mae, rmse=rmse, rel_err=float(rel), max_pred=max_pred, max_true=max_true
    )


# ── Collect results ────────────────────────────────────────────────────────────


def collect_results(model_path: Path = None):
    model = load_model(model_path)
    norm_p = load_norm_params()
    manifest = json.loads(VTK_MANIFEST.read_text())
    test_entries = [
        e for e in manifest["cases"] if e.get("split") == "test" and e.get("success")
    ]

    rows = []
    for entry in test_entries:
        vtu_path = CCX_CASES / entry["vtu"]
        params_path = vtu_path.parent / "case_params.json"
        params = json.loads(params_path.read_text())

        fem_mesh = meshio.read(str(vtu_path))
        coords = fem_mesh.points
        true_mm = fem_mesh.point_data["displacement"]

        X_raw = build_features(coords, params)
        pred_mm = predict(model, X_raw, norm_p)
        m = compute_metrics(pred_mm, true_mm)

        f_N = params["force_N"]
        f_kN = f_N / 1000.0
        if f_N < F_TRAIN_MIN_N:
            regime = "extrap-below"
        elif f_N > F_TRAIN_MAX_N:
            regime = "extrap-above"
        else:
            regime = "interp"

        rows.append(
            dict(
                case_id=entry["case_id"],
                material=params["material"],
                load_dir=params["load_dir"],
                force_N=f_N,
                force_kN=f_kN,
                regime=regime,
                E_MPa=params["E_MPa"],
                nu=params["nu"],
                **m,
            )
        )
        print(
            f"  {entry['case_id']}  {params['material']:20s}  "
            f"dir={params['load_dir']:3s}  F={f_kN:7.1f} kN  "
            f"MAE={m['mae']:.4f} mm  rel={m['rel_err']:8.2f} %  [{regime}]"
        )

    return rows


# ── Plot 1: error vs force ─────────────────────────────────────────────────────


def plot_error_vs_force(rows, out_dir):
    colors = plt.cm.tab10.colors
    mat_colors = {m: colors[i] for i, m in enumerate(MATERIAL_ORDER)}

    fig, axes = plt.subplots(2, len(DIR_ORDER), figsize=(15, 8), sharey="row")
    fig.suptitle("BeamNet – Error vs Force  (test set)", fontsize=13)

    for col, d in enumerate(DIR_ORDER):
        for row_idx, (key, ylabel) in enumerate(
            [("mae", "MAE  [mm]"), ("rel_err", "Relative error  [%]")]
        ):
            ax = axes[row_idx][col]
            ax.axvspan(
                F_TRAIN_MIN_N / 1e3,
                F_TRAIN_MAX_N / 1e3,
                alpha=0.10,
                color="green",
                label="train range",
            )

            for mat in MATERIAL_ORDER:
                sub = sorted(
                    [r for r in rows if r["material"] == mat and r["load_dir"] == d],
                    key=lambda r: r["force_kN"],
                )
                if not sub:
                    continue
                xs = [r["force_kN"] for r in sub]
                ys = [r[key] for r in sub]
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=mat_colors[mat],
                    label=mat if col == 0 else "_nolegend_",
                    linewidth=1.8,
                    markersize=6,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Force  [kN]", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"dir = {d}", fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            if col == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out = out_dir / "error_vs_force.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Plot 2: predicted vs FEM max-|U| scatter ──────────────────────────────────


def plot_scatter_maxU(rows, out_dir):
    colors = plt.cm.tab10.colors
    mat_colors = {m: colors[i] for i, m in enumerate(MATERIAL_ORDER)}
    dir_markers = {"Z": "o", "X+": "^", "X-": "v"}

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("BeamNet – Max |U|: Predicted vs FEM  (test set)", fontsize=12)

    all_vals = [r["max_true"] for r in rows] + [r["max_pred"] for r in rows]
    lo = max(min(all_vals) * 0.8, 1e-6)
    hi = max(all_vals) * 1.2

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect", zorder=1)

    for mat in MATERIAL_ORDER:
        for d, mkr in dir_markers.items():
            sub = [r for r in rows if r["material"] == mat and r["load_dir"] == d]
            if not sub:
                continue
            ax.scatter(
                [r["max_true"] for r in sub],
                [r["max_pred"] for r in sub],
                color=mat_colors[mat],
                marker=mkr,
                s=70,
                zorder=3,
                label=f"{mat} | {d}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("FEM max |U|  [mm]", fontsize=10)
    ax.set_ylabel("Predicted max |U|  [mm]", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = out_dir / "scatter_maxU.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Plot 3: heatmap relative error ─────────────────────────────────────────────


def plot_heatmap(rows, out_dir):
    force_levels = sorted(set(r["force_kN"] for r in rows))
    f_labels = [f"{f:.4g} kN" for f in force_levels]

    fig, axes = plt.subplots(1, len(DIR_ORDER), figsize=(14, 4.5), sharey=True)
    fig.suptitle("BeamNet – Relative Error [%]  (test set)", fontsize=13)

    all_rel = [r["rel_err"] for r in rows]
    vmin = max(min(all_rel), 0.1)
    vmax = max(all_rel)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for ax, d in zip(axes, DIR_ORDER):
        data = np.full((len(MATERIAL_ORDER), len(force_levels)), np.nan)
        for r in rows:
            if r["load_dir"] != d:
                continue
            mi = MATERIAL_ORDER.index(r["material"])
            fi = force_levels.index(r["force_kN"])
            data[mi, fi] = r["rel_err"]

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", norm=norm)
        ax.set_title(f"dir = {d}", fontsize=11)
        ax.set_xticks(range(len(f_labels)))
        ax.set_xticklabels(f_labels, fontsize=8, rotation=15)
        ax.set_yticks(range(len(MATERIAL_ORDER)))
        ax.set_yticklabels(MATERIAL_ORDER, fontsize=9)
        ax.set_xlabel("Force", fontsize=9)

        # annotate cells
        for mi in range(len(MATERIAL_ORDER)):
            for fi in range(len(force_levels)):
                v = data[mi, fi]
                if not np.isnan(v):
                    txt = f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"
                    ax.text(
                        fi,
                        mi,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if v > 500 else "black",
                    )

        # vertical separator between regimes
        regime_at_f = {}
        for r in rows:
            if r["load_dir"] == d:
                regime_at_f[r["force_kN"]] = r["regime"]
        prev = None
        for fi, f in enumerate(force_levels):
            reg = regime_at_f.get(f)
            if prev is not None and prev != reg:
                ax.axvline(fi - 0.5, color="navy", linewidth=2.0, linestyle="--")
            prev = reg

    plt.colorbar(im, ax=axes[-1], label="Relative error  [%]", shrink=0.9)
    plt.tight_layout()
    out = out_dir / "heatmap_rel_err.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Summary table ──────────────────────────────────────────────────────────────


def write_summary(rows, out_dir):
    lines = []
    lines.append("=" * 110)
    lines.append("BeamNet – Test-set Results Summary")
    lines.append(
        f"  Training force range: {F_TRAIN_MIN_N / 1e3:.1f} – {F_TRAIN_MAX_N / 1e3:.0f} kN"
    )
    lines.append("=" * 110)
    hdr = (
        f"{'Case':12s}  {'Material':22s}  {'Dir':4s}  {'F [kN]':>8}  "
        f"{'Regime':14s}  {'MAE [mm]':>10}  {'RMSE [mm]':>10}  "
        f"{'Rel.err %':>10}  {'MaxU_FEM':>10}  {'MaxU_pred':>10}"
    )
    lines.append(hdr)
    lines.append("-" * 110)

    for d in DIR_ORDER:
        for mat in MATERIAL_ORDER:
            sub = sorted(
                [r for r in rows if r["material"] == mat and r["load_dir"] == d],
                key=lambda r: r["force_kN"],
            )
            for r in sub:
                lines.append(
                    f"{r['case_id']:12s}  {r['material']:22s}  {r['load_dir']:4s}  "
                    f"{r['force_kN']:8.1f}  {r['regime']:14s}  "
                    f"{r['mae']:10.4f}  {r['rmse']:10.4f}  "
                    f"{r['rel_err']:10.2f}  {r['max_true']:10.4f}  {r['max_pred']:10.4f}"
                )
        lines.append("")

    # aggregate by direction × regime
    lines.append("-" * 110)
    lines.append("Aggregate stats:")
    for d in DIR_ORDER:
        for regime in ["interp", "extrap-below", "extrap-above"]:
            sub = [r for r in rows if r["load_dir"] == d and r["regime"] == regime]
            if not sub:
                continue
            maes = [r["mae"] for r in sub]
            rels = [r["rel_err"] for r in sub]
            lines.append(
                f"  dir={d:3s}  [{regime:14s}]  n={len(sub):2d}  "
                f"mean-MAE={np.mean(maes):8.4f} mm  "
                f"mean-rel={np.mean(rels):8.2f} %  "
                f"max-rel={np.max(rels):8.2f} %"
            )
    lines.append("=" * 110)

    txt = "\n".join(lines)
    print("\n" + txt + "\n")
    out = out_dir / "results_summary.txt"
    out.write_text(txt)
    print(f"  → {out.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="BeamNet visualisation on the test set.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="output directory (default: saves/run_YYYYMMDD_HHMMSS)",
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

    print("Collecting inference results …")
    rows = collect_results(model_path)

    print("\nGenerating plots …")
    plot_error_vs_force(rows, out_dir)
    plot_scatter_maxU(rows, out_dir)
    plot_heatmap(rows, out_dir)
    write_summary(rows, out_dir)

    print(f"\nDone. Outputs in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
