#!/usr/bin/env python3
"""
Run Elmer simulations and immediately convert each result to .vtk.

For each case:
  1. Copy shared mesh files into the case directory
  2. Run  ElmerSolver case.sif  (writes case0001.vtu)
  3. Convert case0001.vtu → case.vtk  (legacy VTK for downstream scripts)

Results are openable in ParaView as soon as each case finishes,
even while the script is still running.

Logs to: elmer_cases/thermal_ccx_beam/results_manifest.json
         elmer_cases/thermal_ccx_beam/vtk_manifest.json  (alias for train.py)

Run with:  python3 scripts/04_RUN_AND_CONVERT.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import meshio
except ImportError:
    print("ERROR: meshio not installed — pip install meshio")
    sys.exit(1)

ELMER_TIMEOUT = 600  # seconds per case
ELMER_CANDIDATES = ["ElmerSolver", "elmersolver"]

MESH_FILES = ["mesh.header", "mesh.nodes", "mesh.elements", "mesh.boundary"]


# ---------------------------------------------------------------------------
# Elmer runner
# ---------------------------------------------------------------------------


def find_elmer() -> str | None:
    for name in ELMER_CANDIDATES:
        if shutil.which(name):
            return name
    return None


def run_elmer(case_dir: Path, elmer_bin: str) -> tuple[bool, str]:
    """Run `ElmerSolver case.sif` in case_dir. Returns (success, message)."""
    if not (case_dir / "case.sif").exists():
        return False, "case.sif not found"

    try:
        result = subprocess.run(
            [elmer_bin, "case.sif"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=ELMER_TIMEOUT,
        )

        if result.returncode != 0:
            return (
                False,
                f"exit code {result.returncode}: {result.stderr[:200].strip()}",
            )

        vtu_files = list(case_dir.glob("case*.vtu"))
        if not vtu_files:
            return False, "no case*.vtu produced"

        stdout = result.stdout
        if "FATAL ERROR" in stdout.upper():
            for line in stdout.splitlines():
                if "fatal" in line.lower():
                    return False, f"Elmer fatal: {line.strip()[:200]}"

        return True, "OK"

    except subprocess.TimeoutExpired:
        return False, f"timeout after {ELMER_TIMEOUT}s"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# VTU → VTK
# ---------------------------------------------------------------------------


def vtu_to_vtk(vtu_path: Path, vtk_path: Path) -> tuple[bool, str]:
    """Convert Elmer .vtu → legacy .vtk. Returns (success, message)."""
    try:
        mesh = meshio.read(str(vtu_path))

        # Elmer writes temperature as "Temperature" — normalise to lower-case
        # key expected by train.py / inference.py
        pd = {}
        for k, v in mesh.point_data.items():
            pd[k.lower() if k == "Temperature" else k] = v
        mesh.point_data = pd

        meshio.write(str(vtk_path), mesh, file_format="vtk")
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent
    cases_dir = project_root / "elmer_cases" / "thermal_ccx_beam"
    manifest_path = cases_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found!")
        print("Run scripts/03_GENERATE_ELMER_INPUTS_THERMAL.py first.")
        sys.exit(1)

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    cases = manifest["cases"]

    # Resolve shared mesh directory
    mesh_dir = Path(manifest.get("mesh_dir", ""))
    if not mesh_dir.exists():
        print(f"ERROR: Mesh directory not found: {mesh_dir}")
        print("Run  freecadcmd scripts/02_MESH_REFERENCE_BEAM.py  first.")
        sys.exit(1)

    missing = [f for f in MESH_FILES if not (mesh_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing mesh files in {mesh_dir}: {missing}")
        sys.exit(1)

    elmer_bin = find_elmer()
    if elmer_bin is None:
        print("ERROR: ElmerSolver not found on PATH!")
        print(f"  Tried: {ELMER_CANDIDATES}")
        print("  Install: sudo apt install elmer")
        sys.exit(1)

    try:
        import meshio as _mio

        mio_ver = _mio.__version__
    except Exception:
        mio_ver = "?"

    print(f"ElmerSolver : {elmer_bin}  ({shutil.which(elmer_bin)})")
    print(f"meshio      : {mio_ver}")
    print("=" * 65)
    print(f"Running {len(cases)} cases  —  Elmer + VTU→VTK per case")
    print(
        f"  {manifest.get('n_train', 0)} train  |  {manifest.get('n_test', 0)} test  "
        f"|  timeout: {ELMER_TIMEOUT}s/case"
    )
    print(f"  Materials : {manifest.get('materials', [])}")
    print("=" * 65)

    results = []
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    tag_sym = {"train": "✓", "test": "▲"}

    for i, case_info in enumerate(cases, start=1):
        case_id = case_info["case_id"]
        case_dir = cases_dir / case_id
        mat = case_info.get("material", "?")
        q_mw = case_info.get("q_total_mW", 0)
        split = case_info.get("split", "train")

        t = tag_sym.get(split, "?")
        print(f"\n{t} [{i:03d}/{len(cases)}] {case_id}")
        print(f"  {mat:<25}  Q={q_mw:>12.2f} mW  [{split}]")

        if not case_dir.exists():
            print(f"  ✗ Directory not found: {case_dir}")
            n_fail += 1
            results.append(
                {
                    "case_id": case_id,
                    "success": False,
                    "message": "directory not found",
                    "elapsed_s": 0,
                    "vtu_file": "",
                    "vtk": "",
                }
            )
            continue

        # ── 0. Copy shared mesh files ──────────────────────────────────────
        for mf in MESH_FILES:
            shutil.copy2(mesh_dir / mf, case_dir / mf)

        # ── 1. Run ElmerSolver ─────────────────────────────────────────────
        t0 = time.time()
        elmer_ok, elmer_msg = run_elmer(case_dir, elmer_bin)
        elapsed_elmer = round(time.time() - t0, 2)

        if not elmer_ok:
            print(f"  ✗ Elmer failed {elapsed_elmer}s  —  {elmer_msg}")
            n_fail += 1
            results.append(
                {
                    "case_id": case_id,
                    "material": mat,
                    "q_total_mW": q_mw,
                    "T_fix_C": case_info.get("T_fix_C", 20.0),
                    "split": split,
                    "success": False,
                    "message": elmer_msg,
                    "elapsed_s": elapsed_elmer,
                    "vtu_file": "",
                    "vtk": "",
                }
            )
            # Write partial manifest so progress isn't lost
            _write_manifests(cases_dir, manifest, results, i, n_ok, n_fail, t_start)
            continue

        # Pick the latest VTU produced by Elmer (usually case0001.vtu)
        vtu_files = sorted(case_dir.glob("case*.vtu"))
        vtu_path = vtu_files[-1]
        vtu_size = vtu_path.stat().st_size // 1024
        vtu_rel = str(vtu_path.relative_to(cases_dir))
        print(f"  ✓ Elmer {elapsed_elmer}s  —  {vtu_path.name}  {vtu_size} KB")

        # ── 2. VTU → VTK ──────────────────────────────────────────────────
        vtk_path = case_dir / "case.vtk"
        vtk_ok, vtk_msg = vtu_to_vtk(vtu_path, vtk_path)
        vtk_rel = ""

        if vtk_ok:
            vtk_size = vtk_path.stat().st_size // 1024
            vtk_rel = str(vtk_path.relative_to(cases_dir))
            print(f"  ✓ VTK  {vtk_size} KB  →  {vtk_path.name}")
        else:
            print(f"  ✗ VTK failed  —  {vtk_msg}")

        elapsed_total = round(time.time() - t0, 2)
        n_ok += 1

        results.append(
            {
                "case_id": case_id,
                "material": mat,
                "q_total_mW": q_mw,
                "T_fix_C": case_info.get("T_fix_C", 20.0),
                "split": split,
                "success": True,
                "message": "OK",
                "elapsed_s": elapsed_total,
                "vtu_file": vtu_rel,
                "vtk": vtk_rel,
            }
        )

        # Write incremental manifest after every case
        _write_manifests(cases_dir, manifest, results, i, n_ok, n_fail, t_start)

    total_elapsed = round(time.time() - t_start, 1)

    # Final manifests
    final = {
        "type": manifest.get("type", "thermal_elmer_ccx"),
        "n_total": len(cases),
        "n_success": n_ok,
        "n_failed": n_fail,
        "total_time_s": total_elapsed,
        "avg_time_s": round(total_elapsed / max(len(cases), 1), 2),
        "cases": results,
    }
    (cases_dir / "results_manifest.json").write_text(json.dumps(final, indent=2))
    (cases_dir / "vtk_manifest.json").write_text(json.dumps(final, indent=2))

    print("\n" + "=" * 65)
    print(f"✓ Succeeded : {n_ok}/{len(cases)}")
    if n_fail:
        print(f"✗ Failed    : {n_fail}")
    print(
        f"✓ Total time: {total_elapsed}s  ({total_elapsed / max(len(cases), 1):.1f}s avg)"
    )
    print("=" * 65)

    splits: dict = {}
    for r in results:
        s = r["split"]
        splits.setdefault(s, [0, 0])
        splits[s][0 if r["success"] else 1] += 1

    print("\nSplit summary:")
    for s, (ok, fail) in splits.items():
        print(f"  {s:<18} {ok}/{ok + fail} succeeded")


def _write_manifests(cases_dir, manifest, results, n_processed, n_ok, n_fail, t_start):
    """Write incremental results_manifest.json and vtk_manifest.json."""
    partial = {
        "type": manifest.get("type", "thermal_elmer_ccx"),
        "n_total": len(manifest["cases"]),
        "n_processed": n_processed,
        "n_success": n_ok,
        "n_failed": n_fail,
        "elapsed_s": round(time.time() - t_start, 1),
        "cases": results,
    }
    (cases_dir / "results_manifest.json").write_text(json.dumps(partial, indent=2))
    (cases_dir / "vtk_manifest.json").write_text(json.dumps(partial, indent=2))


if __name__ == "__main__":
    main()
