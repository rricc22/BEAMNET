#!/usr/bin/env python3
"""
Quick validation test for 03_GENERATE_CCX_INPUTS_ELASTIC.py

Generates a single X- axial compression case at 102,000 N,
verifies the area-weighted load centroid is at the beam centre (Y=50, Z=50),
then runs CalculiX and reports tip displacement vs. analytical solution.

Run with:  python3 scripts/test_03_single_case.py
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

import numpy as np

# ── Load the generator module by path ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "gen",
    Path(__file__).parent / "03_GENERATE_CCX_INPUTS_ELASTIC.py",
)
gen = importlib.util.module_from_spec(_spec)   # type: ignore[arg-type]
_spec.loader.exec_module(gen)                  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
MESH_INP = ROOT / "calculix_mesh" / "reference" / "reference_beam" / "reference_beam.inp"
TEST_DIR = ROOT / "ccx_cases" / "_test_single_case"
FORCE    = 102_000.0  # N
LOAD_DIR = "X-"
MAT_NAME = "Steel_A36"

print("=" * 60)
print(f"Test case : {LOAD_DIR}  F = {FORCE/1000:.0f} kN  mat = {MAT_NAME}")
print("=" * 60)

# ── 1. Read Nload nodes ────────────────────────────────────────────────────
nload_yz = gen.read_nload_nodes(MESH_INP)
yz       = np.array(list(nload_yz.values()))
print(f"\n[1] Nload nodes : {len(nload_yz)}")
print(f"    Node-set centroid : Y = {yz[:,0].mean():.4f}  Z = {yz[:,1].mean():.4f} mm")
print(f"    Expected          : Y = 50.0000  Z = 50.0000 mm")

# ── 2. Area-weighted forces ────────────────────────────────────────────────
cload_entries = gen.area_weighted_cload(nload_yz, FORCE, LOAD_DIR)
forces   = np.array([v for _, _, v in cload_entries])
node_ids = [n for n, _, _ in cload_entries]
weights  = np.abs(forces) / np.abs(forces).sum()

w_y = sum(w * nload_yz[nid][0] for nid, w in zip(node_ids, weights))
w_z = sum(w * nload_yz[nid][1] for nid, w in zip(node_ids, weights))
ecc = max(abs(w_y - 50), abs(w_z - 50))

print(f"\n[2] Cload entries : {len(cload_entries)}")
print(f"    Total force     : {forces.sum():.2f} N  (expected {-FORCE:.2f} N)")
print(f"    Load centroid   : Y = {w_y:.4f}  Z = {w_z:.4f} mm")
print(f"    Eccentricity    : ΔY = {abs(w_y-50):.4f}  ΔZ = {abs(w_z-50):.4f} mm", end="  ")
if ecc < 0.01:
    print("✓ < 0.01 mm")
else:
    print(f"✗ WARNING {ecc:.4f} mm — spurious bending NOT eliminated")

# ── 3. Write .inp ─────────────────────────────────────────────────────────
TEST_DIR.mkdir(parents=True, exist_ok=True)
mat = gen.MATERIALS[MAT_NAME]

# Use absolute mesh path so the standalone .inp can *Include it directly
job_text = gen.generate_job_inp(
    mesh_rel_path=str(MESH_INP.resolve()),
    material_name=MAT_NAME,
    mat=mat,
    force_total=FORCE,
    load_dir=LOAD_DIR,
    cload_entries=cload_entries,
    case_id="test_X-_102kN",
)

job_path = TEST_DIR / "job.inp"
job_path.write_text(job_text)
print(f"\n[3] Job written : {job_path}")

# ── 4. Run CalculiX ───────────────────────────────────────────────────────
print(f"\n[4] Running CalculiX …", flush=True)
result = subprocess.run(
    ["ccx", "-i", "job"],
    cwd=TEST_DIR,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("    ✗ CalculiX FAILED — stdout tail:")
    print(result.stdout[-2000:])
    print(result.stderr[-500:])
    sys.exit(1)

print("    ✓ CalculiX finished")

# ── 5. Check tip displacement vs. analytical ──────────────────────────────
dat_path = TEST_DIR / "job.dat"
u_vals = []
if dat_path.exists():
    in_block = False
    for line in dat_path.read_text().splitlines():
        if "displacements (nset=nload)" in line.lower():
            in_block = True
            continue
        if in_block:
            parts = line.split()
            if len(parts) == 4:
                try:
                    u_vals.append([float(p) for p in parts[1:]])
                except ValueError:
                    pass
            elif u_vals and parts:
                break  # end of block

if u_vals:
    u_mean = np.array(u_vals).mean(axis=0)
    L, A, E = 1000.0, 100.0 * 100.0, mat["E"]
    u1_analytical = -FORCE * L / (A * E)

    print(f"\n[5] Mean tip displacement (Nload nodes):")
    print(f"    U1 (axial X) = {u_mean[0]:+.6f} mm   analytical = {u1_analytical:+.6f} mm", end="  ")
    err = abs(u_mean[0] - u1_analytical) / abs(u1_analytical) * 100
    print(f"err = {err:.2f} %", end="  ")
    print("✓" if err < 2.0 else "✗")
    print(f"    U2 (Y)       = {u_mean[1]:+.6f} mm   (should be near-zero Poisson contraction)")
    print(f"    U3 (Z)       = {u_mean[2]:+.6f} mm   (should be near-zero Poisson contraction)")

    nu  = mat["nu"]
    u23_poisson = nu * FORCE / (A * E) * 50  # rough Poisson estimate at mid cross-section
    print(f"\n    Rough Poisson estimate |U2|,|U3| ~ {u23_poisson:.6f} mm")
    ratio = max(abs(u_mean[1]), abs(u_mean[2])) / abs(u1_analytical)
    if ratio < 0.05:
        print("    ✓ Transverse displacements are small relative to axial — no spurious bending")
    else:
        print(f"    ✗ Transverse/axial ratio = {ratio:.3f} — spurious bending may still be present")
else:
    print("\n[5] Could not parse .dat — check manually")

# ── 6. ParaView ───────────────────────────────────────────────────────────
frd = TEST_DIR / "job.frd"
print(f"\n[6] ParaView: File → Open → {frd}")
print("=" * 60)
