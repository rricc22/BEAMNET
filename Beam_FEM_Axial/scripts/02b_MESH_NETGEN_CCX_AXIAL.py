#!/usr/bin/env python3
"""
Mesh reference_beam.step with NETGEN and write a self-contained CalculiX job.

Mesh settings
-------------
  Engine  : NETGEN (via netgen.occ)
  Order   : first-order  (C3D4 linear tet)
  Max h   : 10 mm

CalculiX job
------------
  BC      : encastre at Nfix  (X = 0)
  Load    : 100 kN in +X direction  (axial tension)
  Output  : displacements (U), reaction forces (RF), stress (S), strain (E)

Coordinate convention (same as rest of project)
------------------------------------------------
  X : beam axis  (0 → 1000 mm)
  Y, Z : cross-section
  Nfix  : X = 0     (clamped end)
  Nload : X = Xmax  (loaded end)

Units: mm | N | MPa | t

Run with:  python3 scripts/02b_MESH_NETGEN_CCX_AXIAL.py
"""

import sys
from pathlib import Path

try:
    from netgen.occ import OCCGeometry
    from netgen.libngpy._meshing import PointId as NGPointId
except ImportError:
    print("ERROR: netgen not found.  Install with:  pip install netgen-mesher")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STEP_FILE = Path(__file__).parent.parent / "CAD" / "reference" / "reference_beam.step"

OUT_DIR = Path(__file__).parent.parent / "calculix_mesh" / "reference" / "reference_beam_netgen"

MAX_H = 10.0          # mm — maximum element edge length
END_FACE_TOL = 0.5    # mm — tolerance to identify X=0 / X=Xmax end faces

# Load
FORCE_TOTAL = 100_000.0  # N  (100 kN)
LOAD_DOF    = 1           # DOF 1 = X direction

# Material: Steel A36
MATERIAL = "Steel_A36"
E_MPA    = 200_000.0
NU       = 0.26
RHO      = 7.85e-9        # t/mm³

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_id_list(f, id_list, per_line: int = 8):
    for i in range(0, len(id_list), per_line):
        chunk = id_list[i : i + per_line]
        f.write(", ".join(str(x) for x in chunk) + "\n")


# ---------------------------------------------------------------------------
# Mesh with NETGEN
# ---------------------------------------------------------------------------

def mesh_with_netgen(step_file: Path) -> tuple[dict, dict, list]:
    """
    Load STEP and generate a first-order tet mesh with NETGEN.

    Returns
    -------
    nodes    : {node_id (1-based): (x, y, z)}
    elements : {elem_id (1-based): [n1, n2, n3, n4]}  (C3D4)
    tris_2d  : list of [n1, n2, n3] for all surface triangles
    """
    print(f"  Loading STEP: {step_file}")
    geo    = OCCGeometry(str(step_file))

    print(f"  Generating mesh  (maxh = {MAX_H} mm, first order) ...")
    ngmesh = geo.GenerateMesh(maxh=MAX_H)

    # --- extract nodes ---
    # CRITICAL: ngmesh.Points() does NOT iterate in PointId order.
    # Element vertices reference actual PointId.nr values (1-based integers).
    # Must access each point by its PointId explicitly so node keys match
    # the element connectivity.
    nodes = {}
    npts = len(list(ngmesh.Points()))
    for i in range(1, npts + 1):
        p = ngmesh[NGPointId(i)]
        nodes[i] = (p.p[0], p.p[1], p.p[2])

    # --- extract volume elements (C3D4) ---
    elements = {}
    for eid, el in enumerate(ngmesh.Elements3D(), start=1):
        verts = [v.nr for v in el.vertices]
        # NETGEN→CalculiX winding fix: swap nodes[2] and nodes[3]
        verts[2], verts[3] = verts[3], verts[2]
        elements[eid] = verts

    # --- extract surface triangles (for area-weighted nodal forces) ---
    tris_2d = []
    for el in ngmesh.Elements2D():
        verts = [v.nr for v in el.vertices]
        if len(verts) == 3:
            tris_2d.append(verts)

    print(f"  Nodes   : {len(nodes):,}")
    print(f"  C3D4 el : {len(elements):,}")
    print(f"  Surf tri: {len(tris_2d):,}")
    return nodes, elements, tris_2d


# ---------------------------------------------------------------------------
# Area-weighted nodal forces on the loaded face
# ---------------------------------------------------------------------------

def compute_nodal_forces(nodes: dict, tris_2d: list, nload: list,
                         force_total: float, dof: int) -> dict:
    """
    Distribute force_total over nload nodes weighted by each node's
    tributary area (1/3 of each adjacent surface triangle on the face).
    This places the resultant exactly at the geometric centroid of the face,
    eliminating any eccentric-load bending artefact.

    Returns {node_id: force_value} for *Cload.
    """
    import math
    nload_set = set(nload)

    # Area contribution per node from triangles whose all 3 vertices are in nload
    area_contrib = {nid: 0.0 for nid in nload}
    for tri in tris_2d:
        if all(v in nload_set for v in tri):
            n1, n2, n3 = tri
            p1 = nodes[n1]; p2 = nodes[n2]; p3 = nodes[n3]
            # triangle area via cross product
            ax, ay, az = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
            bx, by, bz = p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]
            area = 0.5 * math.sqrt((ay*bz-az*by)**2 + (az*bx-ax*bz)**2 + (ax*by-ay*bx)**2)
            share = area / 3.0
            area_contrib[n1] += share
            area_contrib[n2] += share
            area_contrib[n3] += share

    total_area = sum(area_contrib.values())
    if total_area <= 0:
        # fallback: uniform distribution
        f = force_total / len(nload)
        return {nid: f for nid in nload}

    return {nid: force_total * area_contrib[nid] / total_area for nid in nload}


# ---------------------------------------------------------------------------
# Identify boundary node sets along the beam axis (X)
# ---------------------------------------------------------------------------

def identify_end_sets(nodes: dict) -> tuple[list, list, float, float]:
    x_coords = [c[0] for c in nodes.values()]
    x_min = min(x_coords)
    x_max = max(x_coords)

    nfix  = sorted(nid for nid, c in nodes.items() if abs(c[0] - x_min) <= END_FACE_TOL)
    nload = sorted(nid for nid, c in nodes.items() if abs(c[0] - x_max) <= END_FACE_TOL)

    return nfix, nload, x_min, x_max


# ---------------------------------------------------------------------------
# Write self-contained CalculiX .inp  (mesh + material + step)
# ---------------------------------------------------------------------------

def write_inp(out_file: Path, nodes: dict, elements: dict,
              nfix: list, nload: list, x_min: float, x_max: float,
              nodal_forces: dict):

    n_nload = len(nload)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        # ── Header ────────────────────────────────────────────────────────
        f.write("** ============================================================\n")
        f.write("** CalculiX job — reference beam, NETGEN mesh, axial +X load\n")
        f.write(f"** Mesh      : first-order C3D4  maxh={MAX_H} mm\n")
        f.write(f"** Nodes     : {len(nodes):,}     Elements: {len(elements):,}\n")
        f.write(f"** Nfix      : {len(nfix)} nodes  (X = {x_min:.3f} mm)\n")
        f.write(f"** Nload     : {len(nload)} nodes  (X = {x_max:.3f} mm)\n")
        f.write(f"** Load      : {FORCE_TOTAL/1000:.0f} kN  +X  (area-weighted over {n_nload} nodes)\n")
        f.write(f"** Material  : {MATERIAL}  E={E_MPA:.0f} MPa  nu={NU}\n")
        f.write("** Units     : mm | N | MPa | t\n")
        f.write("** ============================================================\n")
        f.write("*Heading\n")
        f.write(" reference_beam — NETGEN C3D4 — 100kN +X axial\n")

        # ── Nodes ─────────────────────────────────────────────────────────
        f.write("**\n*Node\n")
        for nid, (x, y, z) in sorted(nodes.items()):
            f.write(f"{nid:>8}, {x:>18.10f}, {y:>18.10f}, {z:>18.10f}\n")

        # ── Elements ──────────────────────────────────────────────────────
        f.write("**\n*Element, type=C3D4, Elset=Eall\n")
        for eid, en in sorted(elements.items()):
            f.write(f"{eid:>8}, {', '.join(str(n) for n in en)}\n")

        # ── Node sets ─────────────────────────────────────────────────────
        f.write("**\n** --- Node sets ---\n")
        f.write("*Nset, Nset=Nall\n")
        _write_id_list(f, sorted(nodes.keys()))

        f.write("**\n** Clamped end (X=0)\n")
        f.write("*Nset, Nset=Nfix\n")
        _write_id_list(f, nfix)

        f.write("**\n** Loaded end (X=Xmax)\n")
        f.write("*Nset, Nset=Nload\n")
        _write_id_list(f, nload)

        # ── Material ──────────────────────────────────────────────────────
        f.write("**\n** --- Material ---\n")
        f.write(f"*Material, Name={MATERIAL}\n")
        f.write("*Elastic\n")
        f.write(f"{E_MPA:.1f}, {NU:.3f}\n")
        f.write("*Density\n")
        f.write(f"{RHO:.4e}\n")

        # ── Section ───────────────────────────────────────────────────────
        f.write("**\n** --- Section ---\n")
        f.write(f"*Solid Section, Elset=Eall, Material={MATERIAL}\n")

        # ── Step ──────────────────────────────────────────────────────────
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Step: Linear static — 100 kN axial tension (+X)\n")
        f.write("** ============================================================\n")
        f.write("*Step, Nlgeom=No\n")
        f.write("*Static\n")
        f.write("**\n")
        f.write("** Boundary conditions — encastre at X=0\n")
        f.write("*Boundary\n")
        f.write("Nfix, 1, 1\n")
        f.write("Nfix, 2, 2\n")
        f.write("Nfix, 3, 3\n")
        f.write("**\n")
        f.write(f"** Applied load — {FORCE_TOTAL/1000:.0f} kN in +X, area-weighted over {n_nload} nodes\n")
        f.write("*Cload\n")
        for nid in sorted(nodal_forces):
            f.write(f"{nid:>8}, {LOAD_DOF}, {nodal_forces[nid]:.6g}\n")
        f.write("**\n")
        f.write("** Output\n")
        f.write("*El File\n")
        f.write("S, E\n")
        f.write("*Node File\n")
        f.write("U, RF\n")
        f.write("**\n")
        f.write("*El Print, Elset=Eall, Totals=Yes\n")
        f.write("EVOL\n")
        f.write("*Node Print, Nset=Nfix, Totals=Yes\n")
        f.write("RF\n")
        f.write("*Node Print, Nset=Nload, Totals=Yes\n")
        f.write("U\n")
        f.write("**\n")
        f.write("*End Step\n")

    print(f"  Written : {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not STEP_FILE.exists():
        print(f"ERROR: STEP file not found: {STEP_FILE}")
        sys.exit(1)

    print("=" * 60)
    print("NETGEN mesh  →  CalculiX job  (C3D4, maxh=10 mm)")
    print(f"  STEP   : {STEP_FILE.name}")
    print(f"  Max h  : {MAX_H} mm  (first order)")
    print(f"  Load   : {FORCE_TOTAL/1000:.0f} kN  +X axial")
    print("=" * 60)

    # 1. Mesh
    nodes, elements, tris_2d = mesh_with_netgen(STEP_FILE)

    # 2. End-face node sets
    nfix, nload, x_min, x_max = identify_end_sets(nodes)
    print(f"  Nfix  : {len(nfix)} nodes  at X = {x_min:.3f} mm")
    print(f"  Nload : {len(nload)} nodes  at X = {x_max:.3f} mm")

    # 3. Area-weighted nodal forces (avoids eccentric load from non-uniform node spacing)
    nodal_forces = compute_nodal_forces(nodes, tris_2d, nload, FORCE_TOTAL, LOAD_DOF)
    y_centroid = sum(nodes[n][1] * nodal_forces[n] for n in nload) / FORCE_TOTAL
    z_centroid = sum(nodes[n][2] * nodal_forces[n] for n in nload) / FORCE_TOTAL
    print(f"  Load centroid: Y={y_centroid:.4f} mm, Z={z_centroid:.4f} mm  (ideal: 50.0, 50.0)")

    # 4. Write .inp
    out_file = OUT_DIR / "reference_beam_netgen_100kN_X.inp"
    write_inp(out_file, nodes, elements, nfix, nload, x_min, x_max, nodal_forces)

    print("=" * 60)
    print(f"  Nodes    : {len(nodes):,}")
    print(f"  Elements : {len(elements):,}  (C3D4)")
    print(f"  Nfix     : {len(nfix)}")
    print(f"  Nload    : {len(nload)}")
    print(f"  Output   : {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
