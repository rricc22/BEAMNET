#!/usr/bin/env python3
"""
Generate a ~25s video of FEM temperature results.

Layout  : 32-panel grid (4 rows × 8 cols), each panel shows one case
Motion  : camera orbits a full 360° around the beam
Subset  : 5 materials × 10 heat-flux levels = 50 cases
          → 2 groups of 32 panels (last group padded with repeats)

Output  : saves/fem_overview.mp4

Run with:  python3 scripts/make_video.py
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pip install pyvista")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CASES_DIR = ROOT / "elmer_cases" / "thermal_ccx_beam"
SAVES = ROOT / "saves"
SAVES.mkdir(exist_ok=True)

MANIFEST = CASES_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Video settings
# ---------------------------------------------------------------------------
FPS = 30
SECS_PER_GROUP = 5  # seconds per 32-panel group
N_FRAMES_ROT = FPS * SECS_PER_GROUP  # frames per group (= 150)
COLS = 8  # columns in grid
ROWS = 4  # rows in grid
PANELS = COLS * ROWS  # 32 panels per frame
FRAME_W = 3840  # 4K-wide to keep per-panel resolution with 8 cols
FRAME_H = 2160
OUTPUT = SAVES / "fem_overview.mp4"

# 10 Q indices to sample from the 100-step sorted training sweep
Q_INDICES = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

# Camera
CAM_DIST = 1200  # distance from focal point (mm)
ELEV_DEG = 35  # elevation above XY plane
VIEW_ANGLE = 60  # field-of-view (degrees)


# ---------------------------------------------------------------------------
# Build case subset
# ---------------------------------------------------------------------------


def build_subset(manifest: dict) -> list[dict]:
    """Pick 5 mat × 10 Q levels = 50 training cases."""
    from collections import defaultdict

    train = [c for c in manifest["cases"] if c["split"] == "train"]
    groups: dict = defaultdict(list)
    for c in train:
        groups[c["material"]].append(c)
    for k in groups:
        groups[k].sort(key=lambda c: c["q_total_mW"])

    subset = []
    for mat in manifest["materials"]:
        group = groups[mat]
        for idx in Q_INDICES:
            if idx < len(group):
                subset.append(group[idx])
    return subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_label(case: dict) -> str:
    mat = case["material"].replace("_", " ")
    q_w = case["q_total_mW"] / 1000.0
    return f"{mat}\nQ = {q_w:.1f} W"


def load_mesh(case: dict):
    """Load the VTK file for this case; returns None if missing."""
    vtk_path = CASES_DIR / case["case_id"] / "case.vtk"
    if not vtk_path.exists():
        # Fall back to any VTU produced by Elmer
        vtu_files = list((CASES_DIR / case["case_id"]).glob("case*.vtu"))
        if not vtu_files:
            return None
        vtk_path = vtu_files[-1]

    mesh = pv.read(str(vtk_path))

    # Temperature field can be stored as "temperature" or "Temperature"
    temp_key = None
    for k in mesh.point_data.keys():
        if k.lower() == "temperature":
            temp_key = k
            break
    if temp_key is None:
        return None

    mesh.point_data["T_C"] = np.array(mesh.point_data[temp_key], dtype=float)
    return mesh


def set_camera(pl, focal, azimuth_rad):
    """Position the active subplot camera at the given azimuth."""
    elev_rad = np.radians(ELEV_DEG)
    cam_x = focal[0] + CAM_DIST * np.cos(elev_rad) * np.cos(azimuth_rad)
    cam_y = focal[1] + CAM_DIST * np.cos(elev_rad) * np.sin(azimuth_rad)
    cam_z = focal[2] + CAM_DIST * np.sin(elev_rad)
    pl.camera.position = (cam_x, cam_y, cam_z)
    pl.camera.focal_point = tuple(focal)
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = VIEW_ANGLE
    pl.reset_camera_clipping_range()


# ---------------------------------------------------------------------------
# Render one 32-panel group into the already-open movie file
# ---------------------------------------------------------------------------


def render_group(
    pl,
    cases_32: list[dict],
    global_clim: tuple[float, float],
):
    """
    Clear the plotter, add 32 new meshes, then write N_FRAMES_ROT frames
    into the movie that pl already has open.
    """
    meshes = []
    labels = []
    for c in cases_32:
        m = load_mesh(c)
        meshes.append(m)
        labels.append(short_label(c))

    pl.clear()

    for idx, (mesh, label) in enumerate(zip(meshes, labels)):
        row, col = divmod(idx, COLS)
        pl.subplot(row, col)
        pl.add_text(label, font_size=6, color="black", position="upper_left")
        if mesh is not None:
            pl.add_mesh(
                mesh,
                scalars="T_C",
                cmap="plasma",  # dark-purple (cold) → yellow (hot)
                clim=global_clim,
                show_scalar_bar=False,
                show_edges=False,
                lighting=False,  # avoids colour jitter across frames
            )
            pl.reset_camera()
        else:
            pl.add_text("no data", color="red", position="upper_left")

    # Scalar bar in bottom-right panel
    pl.subplot(ROWS - 1, COLS - 1)
    pl.add_scalar_bar(
        "Temperature [°C]",
        color="black",
        title_font_size=10,
        label_font_size=8,
        n_labels=5,
        vertical=False,
        position_x=0.05,
        position_y=0.05,
    )

    # Record focal points after reset_camera (beam centre per subplot)
    focals = []
    for idx in range(PANELS):
        row, col = divmod(idx, COLS)
        pl.subplot(row, col)
        focals.append(np.array(pl.camera.focal_point))

    # Full 360° orbit: 0° → 360° (one complete revolution)
    az_seq = np.linspace(0, 2 * np.pi, N_FRAMES_ROT, endpoint=False)

    for f in range(N_FRAMES_ROT):
        az = az_seq[f]
        for idx in range(PANELS):
            row, col = divmod(idx, COLS)
            pl.subplot(row, col)
            set_camera(pl, focals[idx], az)
        pl.write_frame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("FEM temperature video generator")
    print("=" * 60)

    manifest = json.loads(MANIFEST.read_text())
    subset = build_subset(manifest)
    print(f"  Cases selected : {len(subset)}  (5 mat × 10 Q levels)")

    # Global temperature range for consistent colormap
    print("  Computing global temperature range ...")
    T_min, T_max = float("inf"), float("-inf")
    for c in subset:
        m = load_mesh(c)
        if m is not None:
            T_min = min(T_min, float(m["T_C"].min()))
            T_max = max(T_max, float(m["T_C"].max()))

    if T_min == float("inf"):
        print("  WARNING: no temperature data found — using placeholder range")
        T_min, T_max = 20.0, 100.0
    clim = (T_min, T_max)
    print(f"  T range : {T_min:.2f} – {T_max:.2f} °C")

    pl = pv.Plotter(
        shape=(ROWS, COLS),
        off_screen=True,
        window_size=[FRAME_W, FRAME_H],
    )
    pl.background_color = "white"
    pl.open_movie(str(OUTPUT), framerate=FPS, quality=8)

    # show() initialises the renderer without closing; required before write_frame
    pl.show(auto_close=False)

    n_groups = (len(subset) + PANELS - 1) // PANELS
    total_frames = 0

    for g in range(n_groups):
        cases_32 = subset[g * PANELS : (g + 1) * PANELS]
        while len(cases_32) < PANELS:
            cases_32.append(cases_32[-1])

        print(f"\n  Rendering group {g + 1}/{n_groups}  ({N_FRAMES_ROT} frames) ...")
        render_group(pl, cases_32, clim)
        total_frames += N_FRAMES_ROT

    pl.close()

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    total_sec = total_frames / FPS
    print(f"\n  Video saved : {OUTPUT}")
    print(f"  Duration : {total_sec:.0f}s  |  Size : {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
