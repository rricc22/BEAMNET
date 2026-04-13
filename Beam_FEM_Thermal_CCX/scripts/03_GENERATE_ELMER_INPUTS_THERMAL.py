#!/usr/bin/env python3
"""
Generate Elmer SIF files for the reference beam multi-material thermal cases.

Physics: steady-state heat conduction
  ∇·(k ∇T) = 0

Boundary conditions (match BC IDs written by script 02)
---------------------------------------------------------
  BC 1  (X = 0,    Nfix)  : T = T_fix = 20 °C  (Dirichlet — fixed temperature)
  BC 2  (X = L,    Nload) : Heat Flux = q_total / A  (Neumann — normal heat flux)
  BC 3  (side walls)      : natural BC = zero flux (adiabatic) — no SIF entry

Load sweep
----------
  5 materials × 100 log-spaced flux levels (train) + 5 test levels = 525 cases
    → 500 train  (100 Q values × 5 materials)
    →  25 test   (  5 Q values × 5 materials)

  Log-spacing is used because conductivities span ~50× across materials;
  a linear sweep would be trivial for steel and extreme for concrete.

Units (consistent mm / t / s / °C system)
-------------------------------------------
  Length      : mm
  Temperature : °C
  Heat power  : mW  = mJ/s
  Conductivity: mW/mm/°C  (same numerical value as W/m/K)
  Heat flux   : mW/mm²
  Density     : t/mm³
  Spec. heat  : mm²/s²/°C  (cp J/kg/K × 1e6)

Run with:  python3 scripts/03_GENERATE_ELMER_INPUTS_THERMAL.py
"""

import json
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Multi-material thermal library  (same 5 materials as Beam_FEM_Axial)
# ---------------------------------------------------------------------------

MATERIALS = {
    "Steel_A36": {
        "k": 50.0,  # mW/mm/°C  (= W/m/K)
        "cp": 490e6,  # mm²/s²/°C
        "rho": 7.85e-9,  # t/mm³
        "alpha_th": 12.0e-6,  # 1/°C
        "description": "Structural steel A36 — k=50 W/m/K",
    },
    "Steel_S355": {
        "k": 48.0,
        "cp": 490e6,
        "rho": 7.85e-9,
        "alpha_th": 12.0e-6,
        "description": "Structural steel S355 — k=48 W/m/K",
    },
    "Aluminium_6061": {
        "k": 167.0,
        "cp": 896e6,
        "rho": 2.70e-9,
        "alpha_th": 23.6e-6,
        "description": "Aluminium alloy 6061-T6 — k=167 W/m/K",
    },
    "Titanium_Ti6Al4V": {
        "k": 6.7,
        "cp": 526e6,
        "rho": 4.43e-9,
        "alpha_th": 8.6e-6,
        "description": "Titanium alloy Ti-6Al-4V — k=6.7 W/m/K",
    },
    "Concrete_C30": {
        "k": 1.8,
        "cp": 880e6,
        "rho": 2.40e-9,
        "alpha_th": 10.0e-6,
        "description": "Concrete C30/37 — k=1.8 W/m/K",
    },
}

# Fixed-temperature BC  (°C) — Dirichlet at X=0
T_FIX = 20.0
T_INIT = 20.0

# Heat-flux range [mW] — log-spaced so all materials see interesting gradients
Q_MIN = 500.0
Q_MAX = 100_000.0

N_Q = 100  # log-spaced training Q values per material

# 5 test Q values: bracketing and interpolating the training range
Q_TEST = [
    100.0,  # below training min
    300.0,  # just below training min
    52_500.0,  # mid-range interpolation
    200_000.0,  # above training max
    500_000.0,  # well above training max
]

# Reference beam cross-section area [mm²]
BEAM_AREA_MM2 = 100.0 * 100.0  # 100 × 100 mm


# ---------------------------------------------------------------------------
# SIF template
# ---------------------------------------------------------------------------


def generate_sif(
    material_name: str,
    mat: dict,
    q_total_mW: float,
    T_fix_C: float,
    T_init_C: float,
    case_id: str,
) -> str:
    """Return an Elmer SIF for steady-state heat conduction."""
    q_flux = q_total_mW / BEAM_AREA_MM2  # mW/mm² at BC 2
    L_mm = 1000.0
    delta_T = q_total_mW * L_mm / (mat["k"] * BEAM_AREA_MM2)  # 1D estimate

    return f"""\
! ============================================================
! Elmer SIF — Steady-State Heat Conduction
! Case     : {case_id}
! Material : {material_name}  ({mat["description"]})
! Q_total  : {q_total_mW:.2f} mW  →  q_flux = {q_flux:.6g} mW/mm²
! T_fix    : {T_fix_C:.1f} °C  (BC 1 — X=0)
! T_init   : {T_init_C:.1f} °C
! ΔT_est   : {delta_T:.2f} °C  (1D bar estimate)
! Units    : mm | t | s | °C  →  k in mW/mm/°C, Q in mW
! ============================================================

Header
  CHECK KEYWORDS Warn
  Mesh DB "." "."
  Include Path ""
  Results Directory ""
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian
  Simulation Type = Steady state
  Steady State Max Iterations = 1
  Output Intervals = 1
End

Body 1
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
  Initial condition = 1
End

Solver 1
  Equation = Heat Equation
  Procedure = "HeatSolve" "HeatSolver"
  Variable = Temperature
  Exec Solver = Always
  Stabilize = True
  Optimize Bandwidth = True
  Steady State Convergence Tolerance = 1.0e-8
  Nonlinear System Convergence Tolerance = 1.0e-7
  Nonlinear System Max Iterations = 20
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-10
  Linear System Preconditioning = ILU0
End

Solver 2
  Equation = Result Output
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "case"
  Vtu Format = True
  Binary Output = False
  Exec Solver = After Simulation
End

Equation 1
  Name = "Heat"
  Active Solvers(2) = 1 2
End

Material 1
  Name = "{material_name}"
  Heat Conductivity = {mat["k"]:.4g}
  Heat Capacity = {mat["cp"]:.6g}
  Density = {mat["rho"]:.4e}
End

Initial Condition 1
  Temperature = {T_init_C:.2f}
End

! BC 1 — fixed temperature at X = 0 (Nfix)
Boundary Condition 1
  Target Boundaries(1) = 1
  Name = "Nfix"
  Temperature = {T_fix_C:.2f}
End

! BC 2 — normal heat flux at X = L (Nload)
! q_flux = Q_total / A = {q_total_mW:.2f} / {BEAM_AREA_MM2:.0f} = {q_flux:.6g} mW/mm²
Boundary Condition 2
  Target Boundaries(1) = 2
  Name = "Nload"
  Heat Flux = {q_flux:.6g}
End

! BC 3 (adiabatic walls) — natural BC = zero flux, no entry required
"""


# ---------------------------------------------------------------------------
# Case generator
# ---------------------------------------------------------------------------


class ThermalCaseGenerator:
    """Generate Elmer SIF files for the reference beam."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _make_case_params(
        self,
        case_id: str,
        q_total_mW: float,
        T_fix_C: float,
        T_init_C: float,
        material_name: str,
        split: str,
    ) -> dict:
        mat = MATERIALS[material_name]
        return {
            "case_id": case_id,
            "q_total_mW": q_total_mW,
            "T_fix_C": T_fix_C,
            "T_init_C": T_init_C,
            "material": material_name,
            "k_mW_mm_C": mat["k"],
            "alpha_th_per_C": mat["alpha_th"],
            "split": split,
        }

    def generate_cases(self) -> list:
        """
        100 log-spaced training Q values × 5 materials = 500 train
        +  5 test Q values            × 5 materials =  25 test
        = 525 total cases
        """
        cases = []
        case_idx = 0

        # Log-spaced Q values for training
        q_train = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), N_Q)

        for mat_name in MATERIALS:
            for q in q_train:
                cases.append(
                    self._make_case_params(
                        case_id=f"case_{case_idx:04d}",
                        q_total_mW=float(q),
                        T_fix_C=T_FIX,
                        T_init_C=T_INIT,
                        material_name=mat_name,
                        split="train",
                    )
                )
                case_idx += 1

            for q in Q_TEST:
                cases.append(
                    self._make_case_params(
                        case_id=f"case_{case_idx:04d}",
                        q_total_mW=float(q),
                        T_fix_C=T_FIX,
                        T_init_C=T_INIT,
                        material_name=mat_name,
                        split="test",
                    )
                )
                case_idx += 1

        return cases

    def write_case(self, params: dict) -> Path:
        """Write case.sif and case_params.json for one case."""
        case_dir = self.base_dir / params["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        mat = MATERIALS[params["material"]]
        sif_text = generate_sif(
            material_name=params["material"],
            mat=mat,
            q_total_mW=params["q_total_mW"],
            T_fix_C=params["T_fix_C"],
            T_init_C=params["T_init_C"],
            case_id=params["case_id"],
        )

        (case_dir / "case.sif").write_text(sif_text)
        (case_dir / "case_params.json").write_text(json.dumps(params, indent=2))

        return case_dir / "case.sif"

    def generate_all_cases(self, mesh_dir: str) -> dict:
        """Generate all SIF files and write manifest.json."""
        cases = self.generate_cases()
        n_train = sum(1 for c in cases if c["split"] == "train")
        n_test = sum(1 for c in cases if c["split"] == "test")

        print(f"\n{'=' * 65}")
        print(f"Generating {len(cases)} Elmer Thermal Cases")
        print(f"  Materials : {list(MATERIALS.keys())}")
        print(f"  T_fix     : {T_FIX} °C")
        print(
            f"  Train     : {n_train}  ({N_Q} log-spaced Q/mat,  "
            f"{Q_MIN:.0f}–{Q_MAX:.0f} mW)"
        )
        print(
            f"  Test      : {n_test}   "
            f"({[f'{q / 1000:.2g}kW' for q in Q_TEST]} × 5 mat)"
        )
        print(f"  Mesh      : {mesh_dir}")
        print(f"{'=' * 65}\n")

        manifest = {
            "type": "thermal_elmer_ccx",
            "n_cases": len(cases),
            "n_train": n_train,
            "n_test": n_test,
            "materials": list(MATERIALS.keys()),
            "T_fix_C": T_FIX,
            "n_q": N_Q,
            "q_spacing": "log",
            "train_q_mW": [Q_MIN, Q_MAX],
            "test_q_mW": Q_TEST,
            "mesh_dir": mesh_dir,
            "cases": [],
        }

        for params in cases:
            sif_path = self.write_case(params)

            manifest["cases"].append(
                {
                    "case_id": params["case_id"],
                    "material": params["material"],
                    "q_total_mW": params["q_total_mW"],
                    "T_fix_C": params["T_fix_C"],
                    "T_init_C": params["T_init_C"],
                    "split": params["split"],
                    "sif_file": str(sif_path.relative_to(self.base_dir)),
                }
            )

            tag = {"train": "✓", "test": "▶"}.get(params["split"], "?")
            print(
                f"{tag} {params['case_id']}  {params['material']:<22}  "
                f"Q={params['q_total_mW']:>10.2f} mW  [{params['split']}]"
            )

        manifest_path = self.base_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"\n{'=' * 65}")
        print(f"Generated {len(cases)} cases in {self.base_dir}")
        print(f"Manifest : {manifest_path}")
        print(f"{'=' * 65}")

        return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent
    mesh_dir = project_root / "elmer_mesh" / "reference" / "reference_beam"

    if not mesh_dir.exists():
        print(f"ERROR: Elmer mesh not found at {mesh_dir}")
        print("Run  freecadcmd scripts/02_MESH_REFERENCE_BEAM.py  first.")
        import sys

        sys.exit(1)

    required = ["mesh.header", "mesh.nodes", "mesh.elements", "mesh.boundary"]
    missing = [f for f in required if not (mesh_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing mesh files: {missing}")
        import sys

        sys.exit(1)

    generator = ThermalCaseGenerator(
        base_dir=project_root / "elmer_cases" / "thermal_ccx_beam"
    )
    generator.generate_all_cases(mesh_dir=str(mesh_dir.resolve()))


if __name__ == "__main__":
    main()
