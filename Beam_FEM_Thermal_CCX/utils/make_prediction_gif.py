#!/usr/bin/env python3
"""
Assemble all prediction_case_*.png images from a run directory into a GIF.

Each image is shown for 1 second.  Output is written next to the PNGs.

Usage:
    python3 utils/make_prediction_gif.py <run_dir>

Example:
    python3 utils/make_prediction_gif.py saves/h512-lp10.0-ld1.0-ln1.0-e50
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install Pillow")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 utils/make_prediction_gif.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"ERROR: directory not found: {run_dir}")
        sys.exit(1)

    frames_paths = sorted(run_dir.glob("prediction_case_*.png"))
    if not frames_paths:
        print(f"ERROR: no prediction_case_*.png files found in {run_dir}")
        sys.exit(1)

    print(f"Found {len(frames_paths)} prediction images in {run_dir}")

    frames = [Image.open(p) for p in frames_paths]

    output = run_dir / "predictions.gif"
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=1000,   # ms per frame → 1 second each
        loop=0,          # loop forever
    )

    print(f"GIF saved: {output}  ({len(frames)} frames, {len(frames)}s)")


if __name__ == "__main__":
    main()
