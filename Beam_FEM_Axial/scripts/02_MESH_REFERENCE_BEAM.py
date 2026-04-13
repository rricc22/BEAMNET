#!/usr/bin/env python3
"""
Mesh the reference beam (STEP file) for CalculiX mechanical FEM.

Produces a single CalculiX .inp file containing:
  *Node               — all mesh nodes
  *Element            — C3D10 quadratic tet elements (C3D4 linear fallback)
  *Nset Nall          — all nodes
  *Elset Eall         — all elements
  *Nset Nfix          — nodes on X=0 face  (clamped / encastre end)
  *Nset Nload         — nodes on X=Lmax face  (free / loaded end)
  *Surface Sfix/Sload — element-face surfaces for distributed loads / contact

Coordinate convention
---------------------
  X  : beam axis  (0 → L = 1000 mm)
  Y, Z : cross-section  (0 → a = 100 mm)
  Nfix  : X = 0    (clamped end)
  Nload : X = L    (loaded end)

Mesh target: minimum 10 000 nodes  (fine mesh for axial load study)

Run with:  freecadcmd scripts/02_MESH_REFERENCE_BEAM.py
"""

import sys
import json
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
# CalculiX element type maps
# ---------------------------------------------------------------------------
VOL_TYPE = {4: "C3D4", 10: "C3D10", 8: "C3D8", 20: "C3D20"}
FACE_TYPE = {3: "S3", 6: "S6", 4: "S4", 8: "S8"}

# Tolerance (fraction of X-range) used to identify the two end faces
END_FACE_TOL = 0.02

# ---------------------------------------------------------------------------
# Mesh size targets  —  INCREASED to minimum 10 000 nodes
# ---------------------------------------------------------------------------
TARGET_NODES = 10_000  # desired node count (minimum 10 000 for fine mesh)
NODE_TOL = 0.15  # accept if within ±15 % of TARGET_NODES  (8500–11500)
MAX_ATTEMPTS = 8  # max iterations (more attempts given finer mesh)
CHAR_LEN_MIN = 2.0  # mm — never go finer than this


# ---------------------------------------------------------------------------
# Characteristic-length estimator  (tet10 empirical formula)
# ---------------------------------------------------------------------------


def compute_char_length(volume_mm3: float, n_target: int = TARGET_NODES) -> float:
    """
    Estimate Gmsh CharacteristicLengthMax to hit n_target nodes.

    Empirical relation for Gmsh tet10:
      N_nodes ≈ V / (L_c³ × 0.04)
      ⟹  L_c = (V / (N × 0.04))^(1/3)
    """
    k = 0.04
    L_c = (volume_mm3 / (n_target * k)) ** (1.0 / 3.0)
    return max(L_c, CHAR_LEN_MIN)


# ---------------------------------------------------------------------------
# Helper: write a comma-separated ID list (N ids per line)
# ---------------------------------------------------------------------------


def _write_id_list(f, id_list, per_line: int = 8):
    for i in range(0, len(id_list), per_line):
        chunk = id_list[i : i + per_line]
        f.write(", ".join(str(x) for x in chunk) + "\n")


# ---------------------------------------------------------------------------
# CalculiX .inp writer
# ---------------------------------------------------------------------------


def write_calculix_inp(fem_mesh, output_file: Path, geom_name: str) -> dict:
    """
    Write a CalculiX .inp file from a FreeCAD FEM mesh object.

    End faces are identified along the X axis (beam axis).
      Nfix  → X = X_min  (clamped end)
      Nload → X = X_max  (loaded end)

    Returns a stats dict.
    """
    nodes = fem_mesh.Nodes  # {id: (x, y, z)}
    vol_ids = fem_mesh.getIdByElementType("Volume")  # list of element IDs
    face_ids = fem_mesh.getIdByElementType("Face")  # boundary face IDs

    # Dominant volume element type
    node_counts = [len(fem_mesh.getElementNodes(eid)) for eid in vol_ids]
    dominant_n = max(set(node_counts), key=node_counts.count) if node_counts else 4
    calculix_vol_type = VOL_TYPE.get(dominant_n, "C3D10")

    # End-face identification along X
    x_coords = [c[0] for c in nodes.values()]
    x_min = min(x_coords)
    x_max = max(x_coords)
    x_tol = max(1.0, (x_max - x_min) * END_FACE_TOL)

    nfix_ids = [nid for nid, c in nodes.items() if abs(c[0] - x_min) <= x_tol]
    nload_ids = [nid for nid, c in nodes.items() if abs(c[0] - x_max) <= x_tol]

    # Boundary faces on each end
    def face_on_end(face_node_ids, target_x, tol):
        return all(abs(nodes[n][0] - target_x) <= tol for n in face_node_ids)

    sfix_ids = [
        fid
        for fid in face_ids
        if face_on_end(fem_mesh.getElementNodes(fid), x_min, x_tol)
    ]
    sload_ids = [
        fid
        for fid in face_ids
        if face_on_end(fem_mesh.getElementNodes(fid), x_max, x_tol)
    ]

    # Node-connectivity winding fix for Gmsh → CalculiX
    def fix_orientation(en):
        n = list(en)
        if len(n) == 4:
            n[2], n[3] = n[3], n[2]
        elif len(n) == 10:
            n[2], n[3] = n[3], n[2]
            n[4], n[5] = n[5], n[4]
            n[6], n[7] = n[7], n[6]
            n[8], n[9] = n[9], n[8]
        return n

    # Group volume elements by type
    type_groups: dict = {}
    for eid in vol_ids:
        en = fix_orientation(fem_mesh.getElementNodes(eid))
        etype = VOL_TYPE.get(len(en), "C3D4")
        type_groups.setdefault(etype, []).append((eid, en))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        # Heading
        f.write(f"** CalculiX input deck — mesh only\n")
        f.write(f"** Geometry : {geom_name}\n")
        f.write(f"** Elements : {calculix_vol_type}\n")
        f.write(f"** Nodes    : {len(nodes)}   Volume elements: {len(vol_ids)}\n")
        f.write(f"** Nfix  (X={x_min:.3f}) : {len(nfix_ids)} nodes\n")
        f.write(f"** Nload (X={x_max:.3f}) : {len(nload_ids)} nodes\n")
        f.write("**\n")
        f.write("*Heading\n")
        f.write(f" {geom_name}\n")

        # Nodes
        f.write("**\n*Node\n")
        for nid, (x, y, z) in sorted(nodes.items()):
            f.write(f"{nid:>8}, {x:>18.10f}, {y:>18.10f}, {z:>18.10f}\n")

        # Volume elements
        for etype, elems in type_groups.items():
            f.write(f"**\n*Element, type={etype}, Elset=E_{etype}\n")
            for eid, en in elems:
                f.write(f"{eid:>8}, {', '.join(str(n) for n in en)}\n")

        # Node sets
        f.write("**\n** --- Node sets ---\n")
        f.write("*Nset, Nset=Nall\n")
        _write_id_list(f, sorted(nodes.keys()))

        f.write("**\n*Elset, Elset=Eall\n")
        _write_id_list(f, sorted(vol_ids))

        # Nfix
        f.write("**\n** Clamped end (X=Xmin) — encastre BC\n")
        f.write("*Nset, Nset=Nfix\n")
        _write_id_list(f, sorted(nfix_ids))

        # Nload
        f.write("**\n** Loaded end (X=Xmax) — apply forces here\n")
        f.write("*Nset, Nset=Nload\n")
        _write_id_list(f, sorted(nload_ids))

        # Surfaces
        if sfix_ids:
            f.write("**\n*Surface, Name=Sfix, Type=Node\n")
            f.write("Nfix\n")
        if sload_ids:
            f.write("**\n*Surface, Name=Sload, Type=Node\n")
            f.write("Nload\n")

        f.write("**\n** --- End of mesh definition ---\n")
        f.write("** *Material, *Step, *Boundary, *Cload defined in job.inp\n")

    return {
        "n_nodes": len(nodes),
        "n_elements": len(vol_ids),
        "elem_type": calculix_vol_type,
        "n_nfix": len(nfix_ids),
        "n_nload": len(nload_ids),
        "x_min": x_min,
        "x_max": x_max,
    }


# ---------------------------------------------------------------------------
# Mesh the STEP file with automatic retry to hit TARGET_NODES
# ---------------------------------------------------------------------------


def mesh_step(
    step_file: Path, out_dir: Path, volume_mm3: float, second_order: bool = True
) -> tuple:
    """
    Load a STEP file, Gmsh-mesh it, write a CalculiX .inp.
    Retries with adjusted CharacteristicLengthMax to stay near TARGET_NODES.

    Returns (stats, Path to .inp, char_length_used).
    """
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
    out_file = out_dir / f"{geom_name}.inp"

    char_len = compute_char_length(volume_mm3, TARGET_NODES)
    stats: dict = {}

    lo = TARGET_NODES * (1 - NODE_TOL)  # 8 500
    hi = TARGET_NODES * (1 + NODE_TOL)  # 11 500

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
            stats = write_calculix_inp(mesh_obj.FemMesh, out_file, geom_name)
            break

        # Rescale L_c toward TARGET_NODES in both directions  (N ∝ 1/L_c³)
        scale = (n_nodes / TARGET_NODES) ** (1.0 / 3.0)
        new_len = max(char_len * scale, CHAR_LEN_MIN)
        direction = "coarser" if n_nodes > hi else "finer"
        print(f"  → {direction}: L_c {char_len:.2f} → {new_len:.2f} mm")
        char_len = new_len

    FreeCAD.closeDocument(doc.Name)
    return stats, out_file, char_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent

    step_file = project_root / "CAD" / "reference" / "reference_beam.step"
    json_file = project_root / "CAD" / "reference" / "reference_beam.json"
    out_dir = project_root / "calculix_mesh" / "reference" / "reference_beam"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not step_file.exists():
        print(f"ERROR: {step_file} not found!")
        print("Run 01_GENERATE_REFERENCE_BEAM.py first.")
        sys.exit(1)

    # Load metadata for volume-based mesh sizing
    volume_mm3 = 100.0 * 100.0 * 1000.0  # default: 10^7 mm³
    if json_file.exists():
        with open(json_file) as fh:
            meta = json.load(fh)
        volume_mm3 = meta.get("volume_mm3", volume_mm3)

    print("=" * 60)
    print("Meshing Reference Beam  (C3D10 quadratic tet)")
    print(f"  STEP    : {step_file.name}")
    print(f"  Volume  : {volume_mm3:,.0f} mm³")
    lo = int(TARGET_NODES * (1 - NODE_TOL))
    hi = int(TARGET_NODES * (1 + NODE_TOL))
    print(f"  Target  : ~{TARGET_NODES} nodes  (accept range {lo}–{hi})")
    est_L_c = compute_char_length(volume_mm3)
    print(f"  Initial L_c estimate : {est_L_c:.2f} mm")
    print("=" * 60)

    stats, out_file, char_len = mesh_step(
        step_file,
        out_dir,
        volume_mm3=volume_mm3,
        second_order=True,
    )

    print("\n" + "=" * 60)
    print(f"  Nodes    : {stats['n_nodes']:,}")
    print(f"  Elements : {stats['n_elements']:,}  ({stats['elem_type']})")
    print(f"  Nfix     : {stats['n_nfix']} nodes  (X = {stats['x_min']:.3f})")
    print(f"  Nload    : {stats['n_nload']} nodes  (X = {stats['x_max']:.3f})")
    print(f"  L_c used : {char_len:.2f} mm")
    print(f"  Saved    : {out_file}")
    print("=" * 60)

    # Save mesh stats
    stats["inp_file"] = str(out_file.relative_to(out_dir))
    stats["char_length_mm"] = char_len
    stats_path = out_dir / "mesh_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Stats    : {stats_path}")


if __name__ == "__main__":
    main()
