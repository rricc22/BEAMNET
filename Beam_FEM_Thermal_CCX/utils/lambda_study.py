"""
Lambda hyperparameter study for ThermalNet.
Reads all results_summary.txt files under saves/ and produces:
  - ASCII table printed to stdout
  - saves/lambda_study.png  (grouped bar chart)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT  = Path(__file__).parent.parent
SAVES = ROOT / "saves"

# ── Collect results ────────────────────────────────────────────────────────────

REGIME_RE = re.compile(
    r"\[(interp|extrap-below|extrap-above)\s*\].*?mean-MAE=\s*([\d.]+).*?mean-rel=\s*([\d.]+)"
)

def parse_summary(path: Path) -> dict | None:
    text = path.read_text()
    result = {}
    for m in REGIME_RE.finditer(text):
        regime, mae, rel = m.group(1), float(m.group(2)), float(m.group(3))
        result[regime] = {"rel": rel, "mae": mae}
    return result if len(result) == 3 else None


def model_label(folder: str) -> tuple[str, str]:
    """Return (short_label, loss_family) from folder name."""
    # data-only ablation
    if "data_only" in folder:
        return "data only", "ablation"
    # parse lp / ld / ln or lg
    lp = re.search(r"lp([\d.]+)", folder)
    ld = re.search(r"ld([\d.]+)", folder)
    ln = re.search(r"ln([\d.]+)", folder)
    lg = re.search(r"lg([\d.]+)", folder)
    lp_v = lp.group(1) if lp else "?"
    ld_v = ld.group(1) if ld else "?"
    if ln:
        family = "Neumann BC"
        ln_v = ln.group(1)
        label = f"lp{lp_v} ld{ld_v} ln{ln_v}"
    elif lg:
        family = "Gradient BC"
        lg_v = lg.group(1)
        label = f"lp{lp_v} ld{ld_v} lg{lg_v}"
    else:
        family = "unknown"
        label = folder
    return label, family


entries = []

# saved model subdirectories (skip the ablation folder — handled separately)
for d in sorted(SAVES.iterdir()):
    if d.name == "ablation":
        continue
    summary = d / "results_summary.txt"
    if not summary.exists():
        continue
    parsed = parse_summary(summary)
    if parsed is None:
        continue
    label, family = model_label(d.name)
    if family == "Gradient BC":
        continue
    entries.append({"label": label, "family": family, "folder": d.name, **parsed})

# ablation model (data-only, results in saves/ablation/)
ablation_summary = SAVES / "ablation" / "results_summary.txt"
if ablation_summary.exists():
    parsed = parse_summary(ablation_summary)
    if parsed:
        entries.append({"label": "data only", "family": "ablation", "folder": "ablation", **parsed})

if not entries:
    print("No results_summary.txt files found.")
    raise SystemExit(1)

# ── ASCII table ────────────────────────────────────────────────────────────────

HDR = f"{'Model':<22}  {'Family':<12}  {'Interp rel%':>11}  {'Interp MAE°C':>12}  {'Ext-above rel%':>14}  {'Ext-above MAE°C':>15}  {'Ext-below rel%':>14}"
print("\n" + "=" * len(HDR))
print("ThermalNet – Lambda Study")
print("=" * len(HDR))
print(HDR)
print("-" * len(HDR))

# sort by interpolation rel%
entries.sort(key=lambda e: e["interp"]["rel"])

for e in entries:
    i  = e["interp"]
    ea = e["extrap-above"]
    eb = e["extrap-below"]
    print(
        f"{e['label']:<22}  {e['family']:<12}  "
        f"{i['rel']:>11.2f}  {i['mae']:>12.1f}  "
        f"{ea['rel']:>14.2f}  {ea['mae']:>15.1f}  "
        f"{eb['rel']:>14.1f}"
    )
print("=" * len(HDR) + "\n")

# ── Plot ───────────────────────────────────────────────────────────────────────

FAMILY_COLOR = {
    "Neumann BC":  "#2196F3",   # blue
    "Gradient BC": "#FF9800",   # orange
    "ablation":    "#9E9E9E",   # grey
}

labels      = [e["label"]    for e in entries]
families    = [e["family"]   for e in entries]
interp_rel  = [e["interp"]["rel"]         for e in entries]
above_rel   = [e["extrap-above"]["rel"]   for e in entries]
below_mae   = [e["extrap-below"]["mae"]   for e in entries]
interp_mae  = [e["interp"]["mae"]         for e in entries]

x     = np.arange(len(entries))
width = 0.35
colors = [FAMILY_COLOR.get(f, "#888") for f in families]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ThermalNet – Lambda Hyperparameter Study", fontsize=13, fontweight="bold")

def bar_plot(ax, values, title, ylabel, log=False):
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # value labels on top
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * (1.04 if not log else 1.15),
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=6.5,
        )
    return bars

bar_plot(axes[0], interp_rel,  "Interpolation — mean rel. error [%]",    "mean rel. err. [%]")
bar_plot(axes[1], above_rel,   "Extrap-above — mean rel. error [%]",      "mean rel. err. [%]")
bar_plot(axes[2], interp_mae,  "Interpolation — mean MAE [°C]",           "mean MAE [°C]", log=True)

# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLOR.items()]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = SAVES / "lambda_study.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out.relative_to(ROOT)}")
