"""
Visualize ThermalNet inference results on the test set.

Produces (all saved to saves/):
  results_summary.txt       – ASCII table (MAE, RMSE, rel-err per case)
  error_vs_q.png            – MAE [°C] vs Q [W], one line per material (log-x)
  scatter_maxT.png          – predicted vs FEM max-T scatter, coloured by material
  heatmap_rel_err.png       – 5×5 heatmap (materials × Q levels), relative error %

Usage:
  python3 scripts/visualize_results.py
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
    import matplotlib.colors as mcolors
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arch import ThermalNet

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
ELMER_CASES = ROOT / "elmer_cases" / "thermal_ccx_beam"
SAVES       = ROOT / "saves"
VTK_MANIFEST = ELMER_CASES / "vtk_manifest.json"

# ── Constants (match train.py) ─────────────────────────────────────────────────
LOG_COLS = [3]
MATERIAL_K = {
    "Steel_A36":       50.0,
    "Steel_S355":      48.0,
    "Aluminium_6061": 167.0,
    "Titanium_Ti6Al4V": 6.7,
    "Concrete_C30":    1.8,
}
MATERIAL_ORDER = list(MATERIAL_K.keys())

# Training Q range (mW) – used to label extrapolation zones
Q_TRAIN_MIN_mW = 500.0
Q_TRAIN_MAX_mW = 100_000.0


# ── Model helpers ──────────────────────────────────────────────────────────────

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
    return dict(np.load(ROOT / "saves" / "norm_params.npz"))


def build_features(coords, params):
    N = len(coords)
    k_val = MATERIAL_K.get(params["material"], params.get("k_mW_mm_C", 1.0))
    feats = np.array([params["q_total_mW"], k_val, params["T_fix_C"]], dtype=np.float32)
    return np.hstack([coords.astype(np.float32), np.tile(feats, (N, 1))])


def normalise_X(X, p):
    X_p = X.copy()
    for c in LOG_COLS:
        X_p[:, c] = np.log(X_p[:, c])
    return ((X_p - p["X_mean"]) / p["X_std"]).astype(np.float32)


def predict(model, X_n, p):
    with torch.no_grad():
        Y_n = model(torch.from_numpy(X_n)).numpy()
    return (Y_n * p["Y_std"] + p["Y_mean"]).squeeze()


def compute_metrics(pred_C, true_C):
    diff  = pred_C - true_C
    mae   = float(np.abs(diff).mean())
    rmse  = float(np.sqrt((diff ** 2).mean()))
    denom = float(np.abs(true_C - true_C.min()).mean()) + 1e-12
    rel   = mae / denom * 100.0
    max_pred = float(pred_C.max())
    max_true = float(true_C.max())
    return dict(mae=mae, rmse=rmse, rel_err=rel, max_pred=max_pred, max_true=max_true)


# ── Collect results ────────────────────────────────────────────────────────────

def collect_results(model_path=None):
    model  = load_model(model_path)
    norm_p = load_norm_params()
    manifest = json.loads(VTK_MANIFEST.read_text())

    test_entries = [e for e in manifest["cases"]
                    if e.get("split") == "test" and e.get("success")]

    rows = []
    for entry in test_entries:
        vtk_path    = ELMER_CASES / entry["vtk"]
        params_path = vtk_path.parent / "case_params.json"
        params      = json.loads(params_path.read_text())

        fem_mesh = meshio.read(str(vtk_path))
        coords   = fem_mesh.points

        temp_key = next((k for k in fem_mesh.point_data if k.lower() == "temperature"), None)
        if temp_key is None:
            continue
        true_C = fem_mesh.point_data[temp_key].squeeze()

        X_n    = normalise_X(build_features(coords, params), norm_p)
        pred_C = predict(model, X_n, norm_p)
        m      = compute_metrics(pred_C, true_C)

        q_W = params["q_total_mW"] / 1000.0
        if params["q_total_mW"] < Q_TRAIN_MIN_mW:
            regime = "extrap-below"
        elif params["q_total_mW"] > Q_TRAIN_MAX_mW:
            regime = "extrap-above"
        else:
            regime = "interp"

        rows.append(dict(
            case_id  = entry["case_id"],
            material = params["material"],
            q_mW     = params["q_total_mW"],
            q_W      = q_W,
            T_fix    = params["T_fix_C"],
            regime   = regime,
            **m,
        ))
        print(f"  {entry['case_id']}  {params['material']:20s}  Q={q_W:8.2f} W"
              f"  MAE={m['mae']:8.2f} °C  rel={m['rel_err']:7.2f} %  [{regime}]")

    return rows


# ── Plot 1: error vs Q ─────────────────────────────────────────────────────────

def plot_error_vs_q(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ThermalNet – Error vs Heat Input  (test set)", fontsize=13)

    colors = plt.cm.tab10.colors
    mat_colors = {m: colors[i] for i, m in enumerate(MATERIAL_ORDER)}

    for ax, ylabel, key in zip(
        axes,
        ["MAE  [°C]", "Relative error  [%]"],
        ["mae",        "rel_err"],
    ):
        # training range band
        ax.axvspan(Q_TRAIN_MIN_mW / 1000, Q_TRAIN_MAX_mW / 1000,
                   alpha=0.10, color="green", label="training range")

        for mat in MATERIAL_ORDER:
            sub = sorted([r for r in rows if r["material"] == mat], key=lambda r: r["q_W"])
            qs  = [r["q_W"] for r in sub]
            vs  = [r[key]   for r in sub]
            ax.plot(qs, vs, "o-", color=mat_colors[mat], label=mat, linewidth=1.8,
                    markersize=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Q_total  [W]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = SAVES / "error_vs_q.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Plot 2: predicted vs FEM max-T scatter ─────────────────────────────────────

def plot_scatter_maxT(rows):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ThermalNet – Max Temperature: Predicted vs FEM  (test set)", fontsize=12)

    colors   = plt.cm.tab10.colors
    markers  = {"extrap-below": "v", "interp": "o", "extrap-above": "^"}
    mat_colors = {m: colors[i] for i, m in enumerate(MATERIAL_ORDER)}

    all_vals = [r["max_true"] for r in rows] + [r["max_pred"] for r in rows]
    lo, hi   = min(all_vals) * 0.9, max(all_vals) * 1.1

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction", zorder=1)

    for mat in MATERIAL_ORDER:
        for regime, mkr in markers.items():
            sub = [r for r in rows if r["material"] == mat and r["regime"] == regime]
            if not sub:
                continue
            ax.scatter(
                [r["max_true"] for r in sub],
                [r["max_pred"] for r in sub],
                color=mat_colors[mat], marker=mkr, s=70, zorder=3,
                label=f"{mat} [{regime}]",
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FEM max T  [°C]", fontsize=10)
    ax.set_ylabel("Predicted max T  [°C]", fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = SAVES / "scatter_maxT.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Plot 3: heatmap relative error ─────────────────────────────────────────────

def plot_heatmap(rows):
    # collect unique Q levels in order
    q_levels = sorted(set(r["q_W"] for r in rows))
    q_labels = [f"{q:.4g} W" for q in q_levels]
    mat_labels = MATERIAL_ORDER

    data = np.full((len(mat_labels), len(q_levels)), np.nan)
    for r in rows:
        mi = mat_labels.index(r["material"])
        qi = q_levels.index(r["q_W"])
        data[mi, qi] = r["rel_err"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("ThermalNet – Relative Error [%]  (test set)", fontsize=13)

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   norm=mcolors.LogNorm(vmin=max(data[~np.isnan(data)].min(), 0.1),
                                        vmax=data[~np.isnan(data)].max()))
    plt.colorbar(im, ax=ax, label="Relative error  [%]")

    ax.set_xticks(range(len(q_labels))); ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_yticks(range(len(mat_labels))); ax.set_yticklabels(mat_labels, fontsize=9)
    ax.set_xlabel("Q_total", fontsize=10)
    ax.set_ylabel("Material", fontsize=10)

    # annotate cells
    for mi in range(len(mat_labels)):
        for qi in range(len(q_levels)):
            v = data[mi, qi]
            if not np.isnan(v):
                ax.text(qi, mi, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if v > 200 else "black")

    # vertical lines separating regimes
    # extrap-below | interp | extrap-above
    # find boundary indices
    for r in rows:
        pass
    regime_at_q = {}
    for r in rows:
        regime_at_q[r["q_W"]] = r["regime"]
    prev = None
    for qi, q in enumerate(q_levels):
        reg = regime_at_q[q]
        if prev is not None and prev != reg:
            ax.axvline(qi - 0.5, color="navy", linewidth=2.0, linestyle="--")
        prev = reg

    plt.tight_layout()
    out = SAVES / "heatmap_rel_err.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# ── Summary table ──────────────────────────────────────────────────────────────

def write_summary(rows):
    lines = []
    lines.append("=" * 100)
    lines.append("ThermalNet – Test-set Results Summary")
    lines.append(f"  Training Q range: {Q_TRAIN_MIN_mW/1000:.2f} – {Q_TRAIN_MAX_mW/1000:.0f} W")
    lines.append("=" * 100)
    hdr = (f"{'Case':12s}  {'Material':22s}  {'Q [W]':>10}  "
           f"{'Regime':14s}  {'MAE [°C]':>10}  {'RMSE [°C]':>10}  "
           f"{'Rel.err %':>10}  {'MaxT_FEM':>10}  {'MaxT_pred':>10}")
    lines.append(hdr)
    lines.append("-" * 100)

    for mat in MATERIAL_ORDER:
        sub = sorted([r for r in rows if r["material"] == mat], key=lambda r: r["q_W"])
        for r in sub:
            lines.append(
                f"{r['case_id']:12s}  {r['material']:22s}  {r['q_W']:10.4g}  "
                f"{r['regime']:14s}  {r['mae']:10.3f}  {r['rmse']:10.3f}  "
                f"{r['rel_err']:10.2f}  {r['max_true']:10.2f}  {r['max_pred']:10.2f}"
            )
        lines.append("")

    # aggregate stats by regime
    for regime in ["interp", "extrap-below", "extrap-above"]:
        sub = [r for r in rows if r["regime"] == regime]
        if not sub:
            continue
        maes = [r["mae"] for r in sub]
        rels = [r["rel_err"] for r in sub]
        lines.append(f"[{regime:14s}]  n={len(sub):2d}  "
                     f"mean-MAE={np.mean(maes):8.3f} °C  "
                     f"mean-rel={np.mean(rels):8.2f} %  "
                     f"max-rel={np.max(rels):8.2f} %")
    lines.append("=" * 100)

    txt = "\n".join(lines)
    print("\n" + txt + "\n")

    out = SAVES / "results_summary.txt"
    out.write_text(txt)
    print(f"  → {out.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global SAVES
    parser = argparse.ArgumentParser(description="ThermalNet visualisation")
    parser.add_argument("--model",  default=None, help="path to .pt model file")
    parser.add_argument("--outdir", default=None, help="output directory for plots")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else SAVES / "thermal_pinn.pt"
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    if args.outdir:
        SAVES = Path(args.outdir).resolve()
    SAVES.mkdir(parents=True, exist_ok=True)

    print("Collecting inference results …")
    rows = collect_results(model_path)

    print("\nGenerating plots …")
    plot_error_vs_q(rows)
    plot_scatter_maxT(rows)
    plot_heatmap(rows)
    write_summary(rows)

    print(f"\nDone. Outputs in {SAVES}")


if __name__ == "__main__":
    main()
