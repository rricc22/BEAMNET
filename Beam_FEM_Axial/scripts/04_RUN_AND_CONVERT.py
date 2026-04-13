#!/usr/bin/env python3
"""
Run CalculiX simulations and immediately convert each result to .vtu and .vtk.

For each case:
  1. Run  ccx job       (reads job.inp → writes job.frd)
  2. Convert job.frd  → job.vtu  (ParaView XML)
  3. Convert job.vtu  → job.vtk  (legacy ParaView)

Results are openable in ParaView as soon as each case finishes,
even while the script is still running.

Logs to: ccx_cases/elasticity_axial_beam/results_manifest.json

Run with:  python3 scripts/04_RUN_AND_CONVERT.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import meshio
except ImportError:
    print("ERROR: meshio not installed — pip install meshio")
    sys.exit(1)

CCX_TIMEOUT = 600  # seconds per case
CCX_CANDIDATES = ["ccx", "ccx2.x", "ccx_2.x", "CalculiX", "calculix"]

# CalculiX FRD element type code → meshio cell type
# CalculiX FRD element type codes (t[2] in the -1 record):
#   3 = C3D4  (4-node linear tet)
#   6 = C3D10 (10-node quadratic tet)
FRD_ELEM_TYPE = {
    3: "tetra",
    6: "tetra10",
    1: "hexahedron",
    2: "wedge",
}

# FRD result field name → (output_name, n_components_to_keep)
FIELD_MAP = {
    "DISP": ("displacement", 3),
    "STRESS": ("stress", 6),
    "TOSTRAIN": ("strain", 6),
    "FORC": ("reaction_force", 3),
    "ERROR": ("error_indicator", 1),
}


# ---------------------------------------------------------------------------
# CCX runner
# ---------------------------------------------------------------------------


def find_ccx() -> str | None:
    for name in CCX_CANDIDATES:
        if shutil.which(name):
            return name
    return None


def run_ccx(case_dir: Path, ccx_bin: str) -> tuple[bool, str]:
    """Run `ccx job` in case_dir. Returns (success, message)."""
    try:
        result = subprocess.run(
            [ccx_bin, "job"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=CCX_TIMEOUT,
        )

        frd_file = case_dir / "job.frd"
        if result.returncode != 0:
            return (
                False,
                f"exit code {result.returncode}: {result.stderr[:200].strip()}",
            )
        if not frd_file.exists():
            return False, "job.frd not produced"

        dat_file = case_dir / "job.dat"
        if dat_file.exists():
            for line in dat_file.read_text(errors="replace").splitlines():
                if "ERROR" in line.upper():
                    return False, f"error in .dat: {line.strip()}"

        return True, "OK"

    except subprocess.TimeoutExpired:
        return False, f"timeout after {CCX_TIMEOUT}s"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# FRD parser
# ---------------------------------------------------------------------------


def parse_frd(frd_path: Path) -> tuple[dict, dict, dict]:
    """Parse a CalculiX .frd file. Returns (nodes, elements, results)."""
    nodes: dict = {}
    elements: dict = {}
    results: dict = {}

    lines = frd_path.read_text(errors="replace").splitlines()
    i, n = 0, len(lines)

    while i < n:
        head = lines[i].strip()

        if head.startswith("2C"):
            i += 1
            while i < n:
                l = lines[i]
                if l.strip().startswith("-3"):
                    break
                if l.strip().startswith("-1"):
                    nid = int(l[3:13])
                    nodes[nid] = (float(l[13:25]), float(l[25:37]), float(l[37:49]))
                i += 1

        elif head.startswith("3C"):
            i += 1
            cur_eid = None
            cur_type = None
            cur_conn = []
            while i < n:
                l = lines[i]
                s = l.strip()
                if s.startswith("-3"):
                    if cur_eid is not None:
                        elements[cur_eid] = (cur_type, cur_conn)
                    break
                if s.startswith("-1"):
                    if cur_eid is not None:
                        elements[cur_eid] = (cur_type, cur_conn)
                    t = l.split()
                    cur_eid = int(t[1])
                    cur_type = int(t[2])
                    cur_conn = []
                elif s.startswith("-2"):
                    cur_conn.extend(int(x) for x in l.split()[1:])
                i += 1

        elif head.startswith("100C"):
            i += 1
            field_name = None
            while i < n:
                l = lines[i]
                if l.strip().startswith("-4"):
                    t = l.split()
                    field_name = t[1]
                    break
                i += 1
            i += 1
            while i < n and lines[i].strip().startswith("-5"):
                i += 1

            field_data: dict = {}
            while i < n:
                l = lines[i]
                s = l.strip()
                if s.startswith("-3"):
                    break
                if s.startswith("-1"):
                    nid = int(l[3:13])
                    vals = []
                    row = l.rstrip()
                    pos = 13
                    while pos + 12 <= len(row):
                        chunk = row[pos : pos + 12]
                        if chunk.strip():
                            vals.append(float(chunk))
                        pos += 12
                    field_data[nid] = vals
                i += 1

            if field_name and field_data:
                results[field_name] = field_data

        i += 1

    return nodes, elements, results


# ---------------------------------------------------------------------------
# FRD → VTU → VTK
# ---------------------------------------------------------------------------


def frd_to_vtu(frd_path: Path, vtu_path: Path) -> tuple[bool, str]:
    """Convert .frd → .vtu. Returns (success, message)."""
    try:
        nodes, elements, results = parse_frd(frd_path)

        if not nodes:
            return False, "no nodes in .frd"
        if not elements:
            return False, "no elements in .frd"

        node_ids = sorted(nodes.keys())
        nid_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        points = np.array([nodes[nid] for nid in node_ids], dtype=np.float64)

        cell_blocks: dict = {}
        for eid, (frd_type, conn) in elements.items():
            mio_type = FRD_ELEM_TYPE.get(frd_type, "tetra")
            conn_idx = [nid_to_idx[nid] for nid in conn if nid in nid_to_idx]
            cell_blocks.setdefault(mio_type, []).append(conn_idx)

        cells = [
            (ctype, np.array(conns, dtype=np.int32))
            for ctype, conns in cell_blocks.items()
        ]

        n_pts = len(node_ids)
        point_data: dict = {}

        for frd_name, field_dict in results.items():
            if frd_name not in FIELD_MAP:
                continue
            out_name, keep = FIELD_MAP[frd_name]
            sample = next(iter(field_dict.values()))
            n_write = min(keep, len(sample))

            arr = np.zeros((n_pts, n_write), dtype=np.float64)
            for nid, vals in field_dict.items():
                if nid in nid_to_idx:
                    arr[nid_to_idx[nid]] = vals[:n_write]

            point_data[out_name] = arr.squeeze() if n_write == 1 else arr

        mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
        meshio.write(str(vtu_path), mesh)
        return True, f"{n_pts} nodes"

    except Exception as e:
        return False, str(e)


def vtu_to_vtk(vtu_path: Path, vtk_path: Path) -> tuple[bool, str]:
    """Convert .vtu → legacy .vtk. Returns (success, message)."""
    try:
        mesh = meshio.read(str(vtu_path))
        meshio.write(str(vtk_path), mesh, file_format="vtk")
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).parent.parent
    cases_dir = project_root / "ccx_cases" / "elasticity_axial_beam"

    manifest_path = cases_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found!")
        print("Run scripts/03_GENERATE_CCX_INPUTS_ELASTIC.py first.")
        sys.exit(1)

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    cases = manifest["cases"]

    ccx_bin = find_ccx()
    if ccx_bin is None:
        print("ERROR: CalculiX not found on PATH!")
        print(f"  Tried: {CCX_CANDIDATES}")
        print("  Install: https://www.calculix.de  or  sudo apt install calculix")
        sys.exit(1)

    print(f"CalculiX : {ccx_bin}  ({shutil.which(ccx_bin)})")
    print(f"meshio   : {meshio.__version__}")
    print("=" * 65)
    print(f"Running {len(cases)} cases  —  CCX + FRD→VTU→VTK per case")
    print(
        f"  {manifest.get('n_train', 0)} train  |  {manifest.get('n_test', 0)} test  "
        f"|  timeout: {CCX_TIMEOUT}s/case"
    )
    print("=" * 65)

    results = []
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    tag_sym = {"train": "✓", "test": "▲"}

    for i, case_info in enumerate(cases, start=1):
        case_id = case_info["case_id"]
        case_dir = (cases_dir / case_info["job_inp"]).parent
        mat = case_info.get("material", "?")
        force = case_info.get("force_N", 0)
        load_dir = case_info.get("load_dir", "?")
        split = case_info.get("split", "train")

        t = tag_sym.get(split, "?")
        print(f"\n{t} [{i:04d}/{len(cases)}] {case_id}")
        print(f"  {mat:<25}  F={force:>9.0f} N  dir={load_dir}  [{split}]")

        if not case_dir.exists():
            print(f"  ✗ Directory not found: {case_dir}")
            n_fail += 1
            results.append(
                {
                    "case_id": case_id,
                    "success": False,
                    "message": "directory not found",
                    "elapsed_s": 0,
                    "vtu": "",
                    "vtk": "",
                }
            )
            continue

        # ── 1. Run CalculiX ───────────────────────────────────────────
        t0 = time.time()
        ccx_ok, ccx_msg = run_ccx(case_dir, ccx_bin)
        elapsed_ccx = round(time.time() - t0, 2)

        if not ccx_ok:
            print(f"  ✗ CCX failed {elapsed_ccx}s  —  {ccx_msg}")
            n_fail += 1
            results.append(
                {
                    "case_id": case_id,
                    "material": mat,
                    "force_N": force,
                    "load_dir": load_dir,
                    "split": split,
                    "success": False,
                    "message": ccx_msg,
                    "elapsed_s": elapsed_ccx,
                    "vtu": "",
                    "vtk": "",
                }
            )
            continue

        frd_size = (case_dir / "job.frd").stat().st_size // 1024
        print(f"  ✓ CCX {elapsed_ccx}s  —  job.frd {frd_size} KB")

        # ── 2. FRD → VTU ─────────────────────────────────────────────
        vtu_path = case_dir / "job.vtu"
        vtu_ok, vtu_msg = frd_to_vtu(case_dir / "job.frd", vtu_path)

        if vtu_ok:
            vtu_size = vtu_path.stat().st_size // 1024
            print(f"  ✓ VTU  {vtu_size} KB  →  {vtu_path.name}")
        else:
            print(f"  ✗ VTU failed  —  {vtu_msg}")

        # ── 3. VTU → VTK ─────────────────────────────────────────────
        vtk_path = case_dir / "job.vtk"
        vtk_ok = False
        if vtu_ok:
            vtk_ok, vtk_msg = vtu_to_vtk(vtu_path, vtk_path)
            if vtk_ok:
                vtk_size = vtk_path.stat().st_size // 1024
                print(f"  ✓ VTK  {vtk_size} KB  →  {vtk_path.name}")
            else:
                print(f"  ✗ VTK failed  —  {vtk_msg}")

        elapsed_total = round(time.time() - t0, 2)
        n_ok += 1

        results.append(
            {
                "case_id": case_id,
                "material": mat,
                "force_N": force,
                "load_dir": load_dir,
                "split": split,
                "success": True,
                "message": "OK",
                "elapsed_s": elapsed_total,
                "frd_file": str((case_dir / "job.frd").relative_to(cases_dir)),
                "vtu": str(vtu_path.relative_to(cases_dir)) if vtu_ok else "",
                "vtk": str(vtk_path.relative_to(cases_dir)) if vtk_ok else "",
            }
        )

        # ── Save incremental manifest after every case ─────────────────
        # (so you can inspect results while script is still running)
        partial = {
            "type": manifest.get("type", "elastic"),
            "n_total": len(cases),
            "n_processed": i,
            "n_success": n_ok,
            "n_failed": n_fail,
            "cases": results,
        }
        (cases_dir / "results_manifest.json").write_text(json.dumps(partial, indent=2))

    total_elapsed = round(time.time() - t_start, 1)

    final_manifest = {
        "type": manifest.get("type", "elastic"),
        "n_total": len(cases),
        "n_success": n_ok,
        "n_failed": n_fail,
        "total_time_s": total_elapsed,
        "avg_time_s": round(total_elapsed / max(len(cases), 1), 2),
        "cases": results,
    }
    (cases_dir / "results_manifest.json").write_text(
        json.dumps(final_manifest, indent=2)
    )
    # also write vtk_manifest.json so downstream scripts (train, inference) work as-is
    (cases_dir / "vtk_manifest.json").write_text(json.dumps(final_manifest, indent=2))

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


if __name__ == "__main__":
    main()
