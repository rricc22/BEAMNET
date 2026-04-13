#!/usr/bin/env python3
"""
Patch *Cload in all existing job.inp files to use area-weighted consistent
nodal forces instead of the old uniform-per-node distribution.

Only the *Cload block is rewritten — mesh, material, BCs and output
requests are left untouched. No CalculiX runs are performed here.

After patching, re-run CalculiX + VTU conversion:
    python3 scripts/04_RUN_AND_CONVERT.py

Run with:  python3 scripts/patch_cload.py
"""

import importlib.util
import json
import sys
from pathlib import Path

# ── Load the generator module by path ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "gen",
    Path(__file__).parent / "03_GENERATE_CCX_INPUTS_ELASTIC.py",
)
gen = importlib.util.module_from_spec(_spec)   # type: ignore[arg-type]
_spec.loader.exec_module(gen)                  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
MESH_INP  = ROOT / "calculix_mesh" / "reference" / "reference_beam" / "reference_beam.inp"
CASES_DIR = ROOT / "ccx_cases" / "elasticity_axial_beam"

if not MESH_INP.exists():
    print(f"ERROR: mesh not found at {MESH_INP}")
    sys.exit(1)

# ── Read Nload positions once ──────────────────────────────────────────────
print(f"Reading Nload nodes from {MESH_INP.name} …")
nload_yz = gen.read_nload_nodes(MESH_INP)
print(f"  {len(nload_yz)} Nload nodes found\n")

# ── Pre-compute cload entries per (load_dir, force) — cache by key ─────────
# (same force value appears in multiple materials → reuse)
_cload_cache: dict = {}

def get_cload(load_dir: str, force: float) -> list:
    key = (load_dir, round(force, 4))
    if key not in _cload_cache:
        _cload_cache[key] = gen.area_weighted_cload(nload_yz, force, load_dir)
    return _cload_cache[key]


def patch_job_inp(job_path: Path, load_dir: str, force: float) -> bool:
    """
    Replace the *Cload block in job_path in-place.
    Returns True if patched, False if already patched or not found.
    """
    text = job_path.read_text()

    # Skip if already patched (per-node lines, not Nset lines)
    if "*Cload\nNload," not in text and "area-weighted" in text:
        return False  # already patched

    # Build new cload block
    entries = get_cload(load_dir, force)
    new_cload_lines = "\n".join(f"{nid}, {dof}, {val:.6g}" for nid, dof, val in entries)
    n = len(entries)

    new_block = (
        f"** --- Applied load ---\n"
        f"** Total force {force:.0f} N  area-weighted consistent nodal forces ({n} entries)\n"
        f"*Cload\n"
        f"{new_cload_lines}"
    )

    # Replace from "** --- Applied load ---" up to (not including) the next "**\n"
    import re
    pattern = re.compile(
        r"\*\* --- Applied load ---.*?\*Cload\n.*?(?=\n\*\*\n)",
        re.DOTALL,
    )
    new_text, count = pattern.subn(new_block, text)

    if count == 0:
        # Fallback: replace just the *Cload block (older format)
        pattern2 = re.compile(r"\*Cload\nNload,.*?(?=\n\*\*)", re.DOTALL)
        new_text, count = pattern2.subn(f"*Cload\n{new_cload_lines}", text)

    if count == 0:
        print(f"  WARNING: could not find *Cload block in {job_path}")
        return False

    job_path.write_text(new_text)
    return True


# ── Walk all case directories ──────────────────────────────────────────────
case_dirs = sorted(CASES_DIR.glob("case_*"))
if not case_dirs:
    print(f"ERROR: no case directories found in {CASES_DIR}")
    sys.exit(1)

print(f"Found {len(case_dirs)} cases — patching *Cload …\n")

n_patched   = 0
n_skipped   = 0
n_error     = 0

for i, case_dir in enumerate(case_dirs):
    params_path = case_dir / "case_params.json"
    job_path    = case_dir / "job.inp"

    if not params_path.exists() or not job_path.exists():
        print(f"  SKIP {case_dir.name} — missing files")
        n_skipped += 1
        continue

    params = json.loads(params_path.read_text())
    load_dir = params["load_dir"]
    force    = params["force_N"]

    try:
        patched = patch_job_inp(job_path, load_dir, force)
        if patched:
            n_patched += 1
        else:
            n_skipped += 1
    except Exception as e:
        print(f"  ERROR {case_dir.name}: {e}")
        n_error += 1

    # Progress every 100 cases
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(case_dirs)}  patched={n_patched}  skipped={n_skipped}  errors={n_error}")

print(f"\n{'='*50}")
print(f"Done.")
print(f"  Patched : {n_patched}")
print(f"  Skipped : {n_skipped}  (already patched or missing)")
print(f"  Errors  : {n_error}")
print(f"\nNext step: re-run CalculiX + VTU conversion")
print(f"  python3 scripts/04_RUN_AND_CONVERT.py")
