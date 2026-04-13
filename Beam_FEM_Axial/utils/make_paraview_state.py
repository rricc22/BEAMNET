"""
Generate a ParaView state file (.pvsm) showing 8 beam VTU files in a 4×2 layout.

Run with pvpython (NOT python3):
    pvpython scripts/make_paraview_state.py
    pvpython scripts/make_paraview_state.py --field displacement_pred
    pvpython scripts/make_paraview_state.py --field error_magnitude
    pvpython scripts/make_paraview_state.py --cases case_1502 case_1517 case_1532 case_1562 case_1500 case_1503 case_1507 case_1537

Then open the result:
    paraview saves/beam_comparison.pvsm

Available --field values:
    displacement_FEM    FEM ground-truth |U| [mm]      (default)
    displacement_pred   BeamNet prediction |U| [mm]
    error_magnitude     signed error pred-FEM [mm]
    stress              von-Mises stress [MPa]
    strain              equivalent strain [-]
"""

import argparse
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CCX_CASES = ROOT / "ccx_cases" / "elasticity_axial_beam"
SAVES = ROOT / "saves"
MANIFEST = CCX_CASES / "vtk_manifest.json"

# ── Default 8 cases (4×2 grid, row-major) ─────────────────────────────────────
# Row 0 – bending Z, 4 materials, interpolation range (102.5 kN)
# Row 1 – mixed: extrap-below Z, extrap-above Z, axial X+ interp, axial X- interp
DEFAULT_CASES = [
    "case_1502",  # Steel_A36        Z   102.5 kN  interp
    "case_1517",  # Steel_S355       Z   102.5 kN  interp
    "case_1532",  # Aluminium_6061   Z   102.5 kN  interp
    "case_1562",  # Concrete_C30     Z   102.5 kN  interp
    "case_1500",  # Steel_A36        Z     2.0 kN  extrap-below
    "case_1503",  # Steel_A36        Z   250.0 kN  extrap-above
    "case_1507",  # Steel_A36        X+  102.5 kN  interp  (axial)
    "case_1512",  # Steel_A36        X-  102.5 kN  interp  (axial)
]

DEFAULT_FIELD = "displacement_FEM"
WINDOW_W, WINDOW_H = 1920, 1080


# ── Argument parsing ───────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ParaView .pvsm with 8 beam VTU panels."
    )
    p.add_argument(
        "--cases",
        nargs=8,
        default=DEFAULT_CASES,
        metavar="ID",
        help="8 case_ids to display (row-major, 4 per row)",
    )
    p.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        choices=[
            "displacement_FEM",
            "displacement_pred",
            "error_magnitude",
            "stress",
            "strain",
        ],
        help="field to colour by (default: displacement_FEM)",
    )
    p.add_argument(
        "--out", default=str(SAVES / "beam_comparison.pvsm"), help="output .pvsm path"
    )
    p.add_argument("--no-scalar-bar", action="store_true", help="hide colour bar")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_manifest():
    return {e["case_id"]: e for e in json.loads(MANIFEST.read_text())["cases"]}


def case_title(cid, manifest):
    """Short label for the annotation text in each panel."""
    e = manifest.get(cid, {})
    f_kN = e.get("force_N", 0) / 1000.0
    return (
        f"{cid}  {e.get('material', '?')}\n"
        f"dir={e.get('load_dir', '?')}  F={f_kN:.1f} kN"
    )


def vtu_path(cid):
    return str(SAVES / f"{cid}_comparison.vtu")


def scalar_range(src, field):
    """Return (min, max) magnitude range for a field on a source."""
    src.UpdatePipeline()
    pdi = src.GetDataInformation().GetPointDataInformation()
    arr = pdi.GetArrayInformation(field)
    if arr is None:
        return 0.0, 1.0
    nc = arr.GetNumberOfComponents()
    return arr.GetComponentRange(-1 if nc > 1 else 0)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    try:
        import paraview.simple as pvs
    except ImportError:
        print("ERROR: run this script with pvpython, not python3")
        print("  pvpython scripts/make_paraview_state.py")
        sys.exit(1)

    pvs._DisableFirstRenderCameraReset()

    manifest = load_manifest()

    # Validate VTU files exist
    for cid in args.cases:
        p = vtu_path(cid)
        if not Path(p).exists():
            print(f"ERROR: {p} not found.")
            print("  Run:  python3 src/inference.py --vtu")
            sys.exit(1)

    SAVES.mkdir(exist_ok=True)

    # ── Build 4×2 layout ──────────────────────────────────────────────────────
    layout = pvs.CreateLayout("BeamComparison")
    layout.SetSize(WINDOW_W, WINDOW_H)

    views = []

    # Assign first view to root cell (hint=0)
    v0 = pvs.CreateRenderView()
    pvs.AssignViewToLayout(view=v0, layout=layout, hint=0)
    views.append(v0)

    # Split top row into 4 columns (always split the rightmost view)
    for frac in [0.25, 1.0 / 3.0, 0.5]:
        c = layout.SplitViewHorizontal(views[-1], frac)
        v = pvs.CreateRenderView()
        pvs.AssignViewToLayout(view=v, layout=layout, hint=c)
        views.append(v)

    top_views = views[:4]

    # Split each top view vertically to get bottom row
    for tv in top_views:
        c = layout.SplitViewVertical(tv, 0.5)
        v = pvs.CreateRenderView()
        pvs.AssignViewToLayout(view=v, layout=layout, hint=c)
        views.append(v)

    # views[0..3] = top row, views[4..7] = bottom row
    # reorder to row-major: top-left→top-right→bot-left→bot-right
    ordered_views = [
        views[0],
        views[1],
        views[2],
        views[3],
        views[4],
        views[5],
        views[6],
        views[7],
    ]

    # ── Load sources & compute shared colour range ─────────────────────────────
    print(f"Loading {len(args.cases)} VTU files …")
    sources = []
    all_min, all_max = [], []

    for cid in args.cases:
        src = pvs.OpenDataFile(vtu_path(cid))
        pvs.RenameSource(case_title(cid, manifest), src)
        lo, hi = scalar_range(src, args.field)
        all_min.append(lo)
        all_max.append(hi)
        sources.append(src)
        print(f"  {cid}  range=[{lo:.3f}, {hi:.3f}]")

    global_min = min(all_min)
    global_max = max(all_max)
    print(f"  Shared range: [{global_min:.3f}, {global_max:.3f}]")

    # ── Assign sources to views ────────────────────────────────────────────────
    is_vector = args.field in (
        "displacement_FEM",
        "displacement_pred",
        "reaction_force",
    )

    lut = pvs.GetColorTransferFunction(args.field)
    lut.RescaleTransferFunction(global_min, global_max)

    for src, view in zip(sources, ordered_views):
        pvs.SetActiveView(view)
        disp = pvs.Show(src, view)
        disp.Representation = "Surface"

        if is_vector:
            pvs.ColorBy(disp, ("POINTS", args.field, "Magnitude"))
        else:
            pvs.ColorBy(disp, ("POINTS", args.field))

        lut.RescaleTransferFunction(global_min, global_max)

        # Camera: isometric view of the beam (1000 mm long along X)
        view.CameraPosition = [1900, -900, 700]
        view.CameraFocalPoint = [500, 0, 25]
        view.CameraViewUp = [0, 0, 1]
        pvs.ResetCamera(view)
        disp.SetScalarBarVisibility(view, False)

    # ── Colour bar in first panel only ────────────────────────────────────────
    if not args.no_scalar_bar:
        pvs.SetActiveView(ordered_views[0])
        disp0 = pvs.Show(sources[0], ordered_views[0])
        sb = pvs.GetScalarBar(lut, ordered_views[0])
        sb.Title = args.field.replace("_", " ")
        sb.ComponentTitle = "Magnitude [mm]" if is_vector else ""
        sb.Visibility = 1
        sb.WindowLocation = "Upper Right Corner"
        sb.ScalarBarLength = 0.4

    # ── Render all & save ─────────────────────────────────────────────────────
    for view in ordered_views:
        pvs.SetActiveView(view)
        pvs.Render(view)

    pvs.SaveState(args.out)
    print(f"\nState file → {args.out}")
    print(f'Open with:   paraview "{args.out}"')


if __name__ == "__main__":
    main()
