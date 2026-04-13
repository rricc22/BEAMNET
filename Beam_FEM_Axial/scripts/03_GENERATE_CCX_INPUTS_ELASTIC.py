#!/usr/bin/env python3
"""
Generate CalculiX job .inp files for the reference elastic cantilever.
Axial + transverse load study with multiple materials.

  Geometry : 1 × 1 × 10 cube-ratio beam  (100 × 100 × 1000 mm)
  Physics  : linear elasticity, static
  BC       : encastre at Nfix (X=0)
  Loads    : point forces at Nload (X=1000 mm)

Load directions
---------------
  -Z   transverse downward bending  (DOF 3, factor -1)
  +X   axial tension                (DOF 1, factor +1)
  -X   axial compression            (DOF 1, factor -1)

Case structure
--------------
  5 materials × 3 load directions × 100 linearly-spaced forces = 1500 train cases
  + 5 materials × 3 load directions × 5 test forces            =   75 test  cases
  Total : 1575 cases

Output
------
  ccx_cases/elasticity_axial_beam/
    manifest.json
    case_0000/
      job.inp           — full CalculiX job (includes mesh via *Include)
      case_params.json  — parameters for this case

Coordinate convention (from mesher)
------------------------------------
  X : beam axis  (0 → 1000 mm)
  Y, Z : cross-section
  DOF 1=X, 2=Y, 3=Z

Units: mm | N | MPa | t

Run with:  python3 scripts/03_GENERATE_CCX_INPUTS_ELASTIC.py
"""

import json
import numpy as np
from pathlib import Path

try:
    from scipy.spatial import Delaunay
except ImportError:
    print("ERROR: scipy not installed — pip install scipy")
    raise


# ---------------------------------------------------------------------------
# Material library  (E in MPa, nu dimensionless, rho in t/mm³)
# ---------------------------------------------------------------------------

MATERIALS = {
    "Steel_A36": {
        "E": 200_000.0,
        "nu": 0.26,
        "rho": 7.85e-9,
        "description": "Structural steel A36 / S235",
    },
    "Steel_S355": {
        "E": 210_000.0,
        "nu": 0.30,
        "rho": 7.85e-9,
        "description": "High-strength structural steel S355",
    },
    "Aluminium_6061": {
        "E": 69_000.0,
        "nu": 0.33,
        "rho": 2.70e-9,
        "description": "Aluminium alloy 6061-T6",
    },
    "Titanium_Ti6Al4V": {
        "E": 114_000.0,
        "nu": 0.34,
        "rho": 4.43e-9,
        "description": "Titanium alloy Ti-6Al-4V",
    },
    "Concrete_C30": {
        "E": 33_000.0,
        "nu": 0.20,
        "rho": 2.40e-9,
        "description": "Concrete C30/37 (elastic only)",
    },
}

# ---------------------------------------------------------------------------
# Load directions
# DOF 1=X (axial), 2=Y (transverse), 3=Z (transverse)
# ---------------------------------------------------------------------------

LOAD_DIRS = {
    "Z": [(3, -1.000)],  # transverse -Z  (downward bending)
    "X+": [(1, +1.000)],  # axial tension   (+X)
    "X-": [(1, -1.000)],  # axial compression (-X)
}

# ---------------------------------------------------------------------------
# Force ranges (N)
# ---------------------------------------------------------------------------

FORCE_MIN = 5_000.0  #   5 kN
FORCE_MAX = 200_000.0  # 200 kN

N_FORCES = 100  # linearly-spaced force values (training)

# 5 test forces: 2 below range, 1 mid interpolation, 2 above range
FORCE_TEST = [
    2_000.0,  # below training min
    3_500.0,  # just below training min
    102_500.0,  # mid-range (between linspace steps)
    250_000.0,  # above training max
    350_000.0,  # well above training max
]


# ---------------------------------------------------------------------------
# Area-weighted consistent nodal force helpers
# ---------------------------------------------------------------------------


def read_nload_nodes(mesh_inp_path: Path) -> dict:
    """
    Parse a CalculiX mesh .inp and return the YZ coordinates of all nodes
    in the Nload set.

    Returns {node_id: (y, z)}.
    """
    all_nodes: dict = {}
    nload_ids: list = []

    with open(mesh_inp_path) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()

        if upper.startswith("*NODE") and not upper.startswith("*NODE P"):
            # Read node coordinates until next keyword
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                parts = lines[i].strip().split(",")
                if len(parts) >= 4:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    all_nodes[nid] = (x, y, z)
                i += 1
        elif upper.startswith("*NSET") and "NLOAD" in upper:
            # Read node IDs until next keyword
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                parts = lines[i].strip().split(",")
                nload_ids.extend(int(p) for p in parts if p.strip())
                i += 1
        else:
            i += 1

    return {nid: (all_nodes[nid][1], all_nodes[nid][2]) for nid in nload_ids if nid in all_nodes}


def area_weighted_cload(nload_yz: dict, force_total: float, load_dir: str) -> list:
    """
    Compute consistent nodal forces for the loaded face using Delaunay
    triangulation of the node YZ positions.

    Each node receives force proportional to its tributary area (1/3 of each
    adjacent triangle), so the load resultant passes exactly through the
    geometric centroid of the face regardless of node spacing.

    Returns list of (node_id, dof, force_value) tuples.
    """
    node_ids = list(nload_yz.keys())
    coords = np.array([nload_yz[nid] for nid in node_ids])  # (N, 2)

    tri = Delaunay(coords)
    weights = np.zeros(len(node_ids))

    for simplex in tri.simplices:
        a, b, c = coords[simplex[0]], coords[simplex[1]], coords[simplex[2]]
        area = 0.5 * abs(
            (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        )
        for idx in simplex:
            weights[idx] += area / 3.0

    weights /= weights.sum()  # normalise → sum to 1

    entries = []
    for nid, w in zip(node_ids, weights):
        node_force = force_total * w
        for dof, frac in LOAD_DIRS[load_dir]:
            entries.append((nid, dof, node_force * frac))

    return entries


# ---------------------------------------------------------------------------
# CalculiX job .inp template
# ---------------------------------------------------------------------------


def generate_job_inp(
    mesh_rel_path: str,
    material_name: str,
    mat: dict,
    force_total: float,
    load_dir: str,
    cload_entries: list,
    case_id: str,
) -> str:
    """Return a complete CalculiX job .inp file as a string.

    cload_entries : list of (node_id, dof, force_value) from area_weighted_cload().
    Using per-node area-weighted forces places the load resultant at the exact
    face centroid, eliminating spurious bending from eccentric nodal loads.
    """
    cload_lines = "\n".join(
        f"{nid}, {dof}, {val:.6g}" for nid, dof, val in cload_entries
    )

    return f"""\
** ============================================================
** CalculiX Mechanical FEM — Reference Cantilever  (elastic)
** Case     : {case_id}
** Material : {material_name}  ({mat["description"]})
** Load     : {force_total:.0f} N  dir={load_dir}
**            (area-weighted consistent nodal forces, {len(cload_entries)} entries)
** Units    : mm | N | MPa | t
** ============================================================
**
** --- Mesh (nodes + elements + node sets) ---
*Include, Input={mesh_rel_path}
**
** --- Material ---
*Material, Name={material_name}
*Elastic
{mat["E"]:.1f}, {mat["nu"]:.3f}
*Density
{mat["rho"]:.4e}
**
** --- Section assignment ---
*Solid Section, Elset=Eall, Material={material_name}
**
** ============================================================
** Step: Linear static analysis
** ============================================================
*Step, Nlgeom=No
*Static
**
** --- Boundary conditions ---
** Clamped end (Nfix = nodes at X=0): all translations fixed (DOF 1-3)
*Boundary
Nfix, 1, 1
Nfix, 2, 2
Nfix, 3, 3
**
** --- Applied load ---
** Total force {force_total:.0f} N distributed over Nload (X=Xmax, free end)
*Cload
{cload_lines}
**
** --- Output: .frd (full field results) ---
*El File
S, E
*Node File
U, RF
**
** --- Output: .dat (text summary) ---
*El Print, Elset=Eall, Totals=Yes
EVOL
*Node Print, Nset=Nfix, Totals=Yes
RF
*Node Print, Nset=Nload, Totals=Yes
U
**
*End Step
"""


# ---------------------------------------------------------------------------
# Case generator
# ---------------------------------------------------------------------------


class ElasticCaseGenerator:
    """Generate CalculiX cases for the reference elastic cantilever."""

    MESH_REL = "../../../calculix_mesh/reference/reference_beam/reference_beam.inp"

    def __init__(self, base_dir: Path, mesh_inp_path: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Parse Nload node positions once — used for area-weighted forces
        print(f"  Reading Nload node positions from: {mesh_inp_path.name}")
        self._nload_yz = read_nload_nodes(mesh_inp_path)
        print(f"  Nload nodes found : {len(self._nload_yz)}")

    # ------------------------------------------------------------------
    def _case_params(self, case_id, force_total, load_dir, mat_name, split):
        mat = MATERIALS[mat_name]
        return {
            "case_id": case_id,
            "geometry": "reference_beam",
            "force_N": force_total,
            "load_dir": load_dir,
            "material": mat_name,
            "E_MPa": mat["E"],
            "nu": mat["nu"],
            "rho_t_mm3": mat["rho"],
            "n_nload": len(self._nload_yz),
            "split": split,
        }

    # ------------------------------------------------------------------
    def _write_case(self, params: dict) -> Path:
        case_dir = self.base_dir / params["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        mat = MATERIALS[params["material"]]
        cload_entries = area_weighted_cload(
            self._nload_yz, params["force_N"], params["load_dir"]
        )
        job_text = generate_job_inp(
            mesh_rel_path=self.MESH_REL,
            material_name=params["material"],
            mat=mat,
            force_total=params["force_N"],
            load_dir=params["load_dir"],
            cload_entries=cload_entries,
            case_id=params["case_id"],
        )

        job_path = case_dir / "job.inp"
        job_path.write_text(job_text)

        params_path = case_dir / "case_params.json"
        params_path.write_text(json.dumps(params, indent=2))

        return job_path

    # ------------------------------------------------------------------
    def generate_all(self) -> dict:
        forces = np.linspace(FORCE_MIN, FORCE_MAX, N_FORCES)
        cases = []
        case_idx = 0

        # ── Training cases  (5 mat × 3 dir × 100 forces = 1500) ──────
        for mat_name in MATERIALS:
            for load_dir in LOAD_DIRS:
                for f in forces:
                    params = self._case_params(
                        case_id=f"case_{case_idx:04d}",
                        force_total=float(f),
                        load_dir=load_dir,
                        mat_name=mat_name,
                        split="train",
                    )
                    cases.append(params)
                    case_idx += 1

        # ── Test cases  (5 mat × 3 dir × 5 forces = 75) ──────────────
        for mat_name in MATERIALS:
            for load_dir in LOAD_DIRS:
                for f in FORCE_TEST:
                    params = self._case_params(
                        case_id=f"case_{case_idx:04d}",
                        force_total=float(f),
                        load_dir=load_dir,
                        mat_name=mat_name,
                        split="test",
                    )
                    cases.append(params)
                    case_idx += 1

        # ── Write files ───────────────────────────────────────────────
        n_train = sum(1 for c in cases if c["split"] == "train")
        n_test = sum(1 for c in cases if c["split"] == "test")

        print(f"\n{'=' * 65}")
        print(f"Generating {len(cases)} Elastic Reference Cases")
        print(f"  Materials  : {list(MATERIALS.keys())}")
        print(f"  Directions : {list(LOAD_DIRS.keys())}")
        print(
            f"  Train      : {n_train}  ({len(MATERIALS)} mat × {len(LOAD_DIRS)} dir × {N_FORCES} forces)"
        )
        print(f"               Force range {FORCE_MIN:.0f} N → {FORCE_MAX:.0f} N")
        print(
            f"  Test       : {n_test}  ({len(MATERIALS)} mat × {len(LOAD_DIRS)} dir × {len(FORCE_TEST)} forces)"
        )
        print(f"               {[f'{f / 1000:.1f}kN' for f in FORCE_TEST]}")
        print(f"{'=' * 65}\n")

        manifest_cases = []

        for params in cases:
            job_path = self._write_case(params)
            split_sym = {"train": "✓", "test": "▲"}.get(params["split"], "?")
            print(
                f"{split_sym} {params['case_id']}  {params['material']:<22}"
                f"  dir={params['load_dir']:<4}  F={params['force_N']:>9.0f} N"
                f"  [{params['split']}]"
            )
            manifest_cases.append(
                {
                    "case_id": params["case_id"],
                    "material": params["material"],
                    "force_N": params["force_N"],
                    "load_dir": params["load_dir"],
                    "split": params["split"],
                    "job_inp": str(job_path.relative_to(self.base_dir)),
                }
            )

        manifest = {
            "type": "elastic",
            "geometry": "reference_beam",
            "n_cases": len(cases),
            "n_train": n_train,
            "n_test": n_test,
            "materials": list(MATERIALS.keys()),
            "load_directions": list(LOAD_DIRS.keys()),
            "n_forces": N_FORCES,
            "train_force_N": [FORCE_MIN, FORCE_MAX],
            "test_force_N": FORCE_TEST,
            "cases": manifest_cases,
        }

        manifest_path = self.base_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"\n{'=' * 65}")
        print(f"✓ {len(cases)} cases written to {self.base_dir}")
        print(f"✓ Manifest: {manifest_path}")
        print(f"{'=' * 65}")

        return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent

    mesh_inp = (
        project_root
        / "calculix_mesh"
        / "reference"
        / "reference_beam"
        / "reference_beam.inp"
    )

    if not mesh_inp.exists():
        print(f"ERROR: {mesh_inp} not found — run 02_MESH_REFERENCE_BEAM.py first.")
        raise SystemExit(1)

    generator = ElasticCaseGenerator(
        base_dir=project_root / "ccx_cases" / "elasticity_axial_beam",
        mesh_inp_path=mesh_inp,
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
