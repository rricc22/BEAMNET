#!/usr/bin/env python3
"""
Generate the reference beam geometry: 1 × 1 × 10 cube ratio.

  Cross-section : 100 × 100 mm  (Y-Z plane)
  Length        : 1000 mm        (along X)
  Nfix          : face at X = 0  (fixed-temperature end)
  Nload         : face at X = L  (heat-flux end)

Creates
-------
  CAD/reference/reference_beam.step   — STEP geometry for Gmsh/FreeCAD
  CAD/reference/reference_beam.json   — geometry metadata

Run with:  freecadcmd scripts/01_GENERATE_REFERENCE_BEAM.py
"""

import sys
import json
from pathlib import Path

try:
    import FreeCAD
    import Part
except ImportError:
    print("ERROR: Run with freecadcmd!")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Reference beam dimensions  (mm)
# ---------------------------------------------------------------------------
BEAM_SIDE = 100.0  # cross-section side  a
BEAM_LENGTH = 1000.0  # beam length         L = 10 * a


def main():
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "CAD" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    a = BEAM_SIDE
    L = BEAM_LENGTH

    print("=" * 55)
    print("Reference Beam  (1 × 1 × 10 cube ratio)  — Thermal CCX")
    print(f"  Cross-section : {a:.0f} × {a:.0f} mm  (Y-Z plane)")
    print(f"  Length        : {L:.0f} mm  (along X-axis)")
    print(f"  Nfix          : X = 0  (fixed temperature end)")
    print(f"  Nload         : X = {L:.0f}  (heat flux end)")
    print("=" * 55)

    # Box from (0, 0, 0) to (L, a, a) — beam axis along X
    shape = Part.makeBox(L, a, a)

    meta = {
        "name": "reference_beam",
        "description": (
            f"Square-section prismatic beam {L:.0f}x{a:.0f}x{a:.0f} mm, "
            "axis X, thermal (Elmer)"
        ),
        "a_mm": a,
        "L_mm": L,
        "ratio": [1, 1, 10],
        "axis": "X",
        "x_fix": 0.0,
        "x_load": L,
        "volume_mm3": shape.Volume,
        "surface_mm2": shape.Area,
        "bbox_mm": [
            shape.BoundBox.XLength,
            shape.BoundBox.YLength,
            shape.BoundBox.ZLength,
        ],
    }

    step_file = out_dir / "reference_beam.step"
    shape.exportStep(str(step_file))
    print(f"\n  Saved STEP : {step_file}")

    json_file = out_dir / "reference_beam.json"
    with open(json_file, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved JSON : {json_file}")

    print(f"\n  Volume     : {meta['volume_mm3']:,.0f} mm³")
    print(f"  Surface    : {meta['surface_mm2']:,.0f} mm²")
    print("\nDone.")


if __name__ == "__main__":
    main()
