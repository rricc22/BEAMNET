#!/usr/bin/env python3
"""
Mesh the reference beam (STEP file) and write Elmer mesh files.

Produces 4 Elmer mesh files in elmer_mesh/reference/reference_beam/:
  mesh.header   — node/element counts and type codes
  mesh.nodes    — node coordinates
  mesh.elements — volume elements (body 1)
  mesh.boundary — boundary faces with BC IDs:
                    1 = Nfix  (X = 0,    fixed-temperature end)
                    2 = Nload (X = Lmax, heat-flux end)
                    3 = walls (adiabatic, no BC needed in SIF)

Target mesh density
-------------------
  ~10 000 nodes  (finer than the 2 000-node Elmer Thermal mesh, closer to
  the ~10 000-node CalculiX meshes used in Beam_FEM_Axial)

Coordinate convention
---------------------
  X  : beam axis  (0 → L = 1000 mm)
  Y, Z : cross-section  (0 → a = 100 mm)

Run with:  freecadcmd scripts/02_MESH_REFERENCE_BEAM.py
"""

import sys
import json
from collections import Counter
from pathlib import Path

try:
    import FreeCAD
    import Part
    import ObjectsFem
    import femmesh.gmshtools as gmshtools
except ImportError:
    print("ERROR: Run with freecadcmd!")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
END_FACE_TOL = 0.02  # fraction of beam length for end-face detection
TARGET_NODES = 10_000  # aim for ~10 000 nodes (≈ Beam_FEM_Axial density)
NODE_TOL = 0.15  # accept if within ±15 %
MAX_ATTEMPTS = 6
CHAR_LEN_MIN = 5.0  # mm


def compute_char_length(volume_mm3: float, n_target: int = TARGET_NODES) -> float:
    k = 0.04
    L_c = (volume_mm3 / (n_target * k)) ** (1.0 / 3.0)
    return max(L_c, CHAR_LEN_MIN)


# ---------------------------------------------------------------------------
# Elmer mesh writer
# ---------------------------------------------------------------------------


def write_elmer_mesh(fem_mesh, output_dir: Path, geom_name: str) -> dict:
    """
    Write 4 Elmer mesh files for the reference beam.

    Boundary condition IDs
    ----------------------
      1  Nfix  (X = x_min)   fixed-temperature end
      2  Nload (X = x_max)   heat-flux end
      3  walls               adiabatic lateral faces
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = fem_mesh.Nodes
    vol_ids = fem_mesh.getIdByElementType("Volume")
    face_ids = fem_mesh.getIdByElementType("Face")

    # Identify beam ends along X
    x_coords = [c[0] for c in nodes.values()]
    x_min = min(x_coords)
    x_max = max(x_coords)
    x_tol = max(1.0, (x_max - x_min) * END_FACE_TOL)

    nfix_nodes = frozenset(
        nid for nid, c in nodes.items() if abs(c[0] - x_min) <= x_tol
    )
    nload_nodes = frozenset(
        nid for nid, c in nodes.items() if abs(c[0] - x_max) <= x_tol
    )

    def get_bc_id(face_node_ids):
        if all(nid in nfix_nodes for nid in face_node_ids):
            return 1  # fixed-temperature end
        if all(nid in nload_nodes for nid in face_node_ids):
            return 2  # heat-flux end
        return 3  # adiabatic lateral wall

    # Count element types for header
    vol_types = Counter()
    for vid in vol_ids:
        n = len(fem_mesh.getElementNodes(vid))
        vol_types[504 if n == 4 else 510] += 1  # linear / quadratic tet

    bnd_types = Counter()
    for fid in face_ids:
        n = len(fem_mesh.getElementNodes(fid))
        bnd_types[303 if n == 3 else 306] += 1  # linear / quadratic tri

    all_types = list(vol_types.items()) + list(bnd_types.items())

    # ── mesh.header ──────────────────────────────────────────────────────────
    with open(output_dir / "mesh.header", "w") as f:
        f.write(f"{len(nodes)} {len(vol_ids)} {len(face_ids)}\n")
        f.write(f"{len(all_types)}\n")
        for et, cnt in all_types:
            f.write(f"{et} {cnt}\n")

    # ── mesh.nodes ───────────────────────────────────────────────────────────
    with open(output_dir / "mesh.nodes", "w") as f:
        for nid, (x, y, z) in sorted(nodes.items()):
            f.write(f"{nid} -1 {x:.10f} {y:.10f} {z:.10f}\n")

    # ── mesh.elements ─────────────────────────────────────────────────────────
    with open(output_dir / "mesh.elements", "w") as f:
        for idx, vid in enumerate(vol_ids, start=1):
            en = fem_mesh.getElementNodes(vid)
            et = 504 if len(en) == 4 else 510
            f.write(f"{idx} 1 {et} {' '.join(str(n) for n in en)}\n")

    # ── mesh.boundary ─────────────────────────────────────────────────────────
    n_nfix = n_nload = 0
    with open(output_dir / "mesh.boundary", "w") as f:
        for idx, fid in enumerate(face_ids, start=1):
            fn = fem_mesh.getElementNodes(fid)
            et = 303 if len(fn) == 3 else 306
            bc = get_bc_id(fn)
            if bc == 1:
                n_nfix += 1
            elif bc == 2:
                n_nload += 1
            f.write(f"{idx} {bc} 0 0 {et} {' '.join(str(n) for n in fn)}\n")

    dom_vol_type = 510 if 510 in vol_types else 504
    return {
        "n_nodes": len(nodes),
        "n_elements": len(vol_ids),
        "n_faces": len(face_ids),
        "elem_type": f"{'C3D10' if dom_vol_type == 510 else 'C3D4'} → Elmer {dom_vol_type}",
        "n_nfix": n_nfix,
        "n_nload": n_nload,
        "x_min": x_min,
        "x_max": x_max,
    }


# ---------------------------------------------------------------------------
# Mesh builder (FreeCAD / Gmsh)
# ---------------------------------------------------------------------------


def mesh_step(
    step_file: Path, out_dir: Path, volume_mm3: float, second_order: bool = True
) -> tuple:
    doc = FreeCAD.newDocument("ref_mesh")

    shape = Part.Shape()
    shape.read(str(step_file))
    part = doc.addObject("Part::Feature", "Part")
    part.Shape = shape

    mesh_obj = ObjectsFem.makeMeshGmsh(doc, "FEMmesh")
    mesh_obj.Shape = part

    if hasattr(mesh_obj, "SecondOrderLinear"):
        mesh_obj.SecondOrderLinear = False
    if hasattr(mesh_obj, "Order"):
        mesh_obj.Order = 2 if second_order else 1

    doc.recompute()

    geom_name = step_file.stem
    char_len = compute_char_length(volume_mm3, TARGET_NODES)
    stats: dict = {}

    lo = TARGET_NODES * (1 - NODE_TOL)
    hi = TARGET_NODES * (1 + NODE_TOL)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        mesh_obj.CharacteristicLengthMax = char_len
        mesh_obj.CharacteristicLengthMin = char_len / 5.0

        gmsh_mesh = gmshtools.GmshTools(mesh_obj)
        gmsh_mesh.create_mesh()

        n_nodes = len(mesh_obj.FemMesh.Nodes)
        is_last = attempt == MAX_ATTEMPTS

        print(
            f"  [attempt {attempt}/{MAX_ATTEMPTS}]  "
            f"L_c = {char_len:.2f} mm  →  {n_nodes} nodes",
            end="",
        )

        in_range = lo <= n_nodes <= hi

        if in_range or is_last:
            if not in_range:
                print(f"  (best result after {MAX_ATTEMPTS} attempts — accepting)")
            else:
                print(f"  ✓ within target range [{lo:.0f}–{hi:.0f}]")
            stats = write_elmer_mesh(mesh_obj.FemMesh, out_dir, geom_name)
            break

        scale = (n_nodes / TARGET_NODES) ** (1.0 / 3.0)
        new_len = max(char_len * scale, CHAR_LEN_MIN)
        direction = "coarser" if n_nodes > hi else "finer"
        print(f"  → {direction}: L_c {char_len:.2f} → {new_len:.2f} mm")
        char_len = new_len

    FreeCAD.closeDocument(doc.Name)
    return stats, out_dir, char_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent

    step_file = project_root / "CAD" / "reference" / "reference_beam.step"
    json_file = project_root / "CAD" / "reference" / "reference_beam.json"
    out_dir = project_root / "elmer_mesh" / "reference" / "reference_beam"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not step_file.exists():
        print(f"ERROR: {step_file} not found!")
        print("Run 01_GENERATE_REFERENCE_BEAM.py first.")
        sys.exit(1)

    volume_mm3 = 100.0 * 100.0 * 1000.0
    if json_file.exists():
        with open(json_file) as fh:
            meta = json.load(fh)
        volume_mm3 = meta.get("volume_mm3", volume_mm3)

    print("=" * 60)
    print("Meshing Reference Beam  (quadratic tet)  — Elmer format")
    print(f"  STEP    : {step_file.name}")
    print(f"  Volume  : {volume_mm3:,.0f} mm³")
    lo = int(TARGET_NODES * (1 - NODE_TOL))
    hi = int(TARGET_NODES * (1 + NODE_TOL))
    print(f"  Target  : ~{TARGET_NODES:,} nodes  (accept range {lo:,}–{hi:,})")
    print(f"  Output  : {out_dir}")
    print("=" * 60)

    stats, out_dir, char_len = mesh_step(
        step_file,
        out_dir,
        volume_mm3=volume_mm3,
        second_order=True,
    )

    print("\n" + "=" * 60)
    print(f"  Nodes      : {stats['n_nodes']:,}")
    print(f"  Elements   : {stats['n_elements']:,}  ({stats['elem_type']})")
    print(f"  Faces      : {stats['n_faces']:,}  (boundary elements)")
    print(
        f"  BC 1 Nfix  : {stats['n_nfix']} faces  (X = {stats['x_min']:.1f})  ← fixed T"
    )
    print(
        f"  BC 2 Nload : {stats['n_nload']} faces  (X = {stats['x_max']:.1f})  ← heat flux"
    )
    print(
        f"  BC 3 walls : {stats['n_faces'] - stats['n_nfix'] - stats['n_nload']} faces  ← adiabatic"
    )
    print(f"  L_c used   : {char_len:.2f} mm")
    print(f"  Saved in   : {out_dir}")
    print("=" * 60)

    stats["mesh_dir"] = str(out_dir)
    stats["char_length_mm"] = char_len
    stats_path = out_dir / "mesh_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Stats      : {stats_path}")


if __name__ == "__main__":
    main()
