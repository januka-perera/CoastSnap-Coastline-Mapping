#!/usr/bin/env python
"""
Reproject plan-view shorelines onto the original oblique images.

Each plan-view shoreline .npy is transformed to oblique image coordinates
using a pre-computed plan-to-oblique homography, then drawn on the
corresponding oblique image.

The homography files are produced by align.py in the CoastSnap-Object-Detection
repository.  Each file is named <stem>_H_plan_to_oblique.npy and encodes the
3x3 perspective transform mapping plan-view pixel (col, row) to oblique image
pixel (u, v).

Usage
-----
    python run_reproject.py \\
        --site <site_name> \\
        --mode points \\
        --oblique-dir <path/to/oblique/images> \\
        --homography-dir <path/to/homography/npy/files>

Reads
-----
    outputs/shorelines/<mode>/<site>/<stem>.npy   — plan-view shoreline coords
    <oblique-dir>/<stem>.jpg                       — original oblique images
    <homography-dir>/<stem>_H_plan_to_oblique.npy — per-image homographies

Writes
------
    outputs/shorelines_oblique/<mode>/<site>/<stem>.jpg — shoreline on oblique
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.io import list_images, load_image_rgb
from src.utils.visualization import draw_shoreline, save_visualization

# Suffix appended to the plan stem to form the oblique stem.
# e.g. "1234_plan.npy"  →  strip "_plan"  →  "1234"
_PLAN_SUFFIX = "_plan"


def load_homography(path: Path) -> np.ndarray:
    H = np.load(str(path))
    if H.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) homography, got {H.shape} in {path}")
    return H.astype(np.float64)


def reproject_shoreline(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H to plan-view shoreline points.

    Parameters
    ----------
    pts : (N, 2) float32  — (x, y) plan-view pixel coordinates
    H   : (3, 3) float64  — plan-view → oblique homography

    Returns
    -------
    (N, 2) float32  — (x, y) oblique image pixel coordinates
    """
    if len(pts) == 0:
        return pts
    pts_h = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts_h, H.astype(np.float32))
    return out.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Reproject plan-view shorelines onto oblique images."
    )
    parser.add_argument("--site",  required=True, help="Site name")
    parser.add_argument("--mode",  required=True, choices=["points", "video"],
                        help="Segmentation mode used to produce the shorelines")
    parser.add_argument("--oblique-dir",   required=True,
                        help="Directory containing the original oblique images")
    parser.add_argument("--homography-dir", required=True,
                        help="Directory containing <stem>_H_plan_to_oblique.npy files")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    shoreline_dir  = Path("outputs/shorelines") / args.mode / args.site
    out_dir        = Path("outputs/shorelines_oblique") / args.mode / args.site
    oblique_dir    = Path(args.oblique_dir)
    homography_dir = Path(args.homography_dir)

    if not shoreline_dir.exists():
        print(f"No shoreline outputs found at {shoreline_dir}")
        print(f"Run  python run_shoreline.py --site {args.site} --mode {args.mode}  first.")
        sys.exit(1)

    npy_paths = sorted(shoreline_dir.glob("*.npy"))
    if not npy_paths:
        print(f"No shoreline .npy files found in {shoreline_dir}")
        sys.exit(1)

    # Index oblique images by stem for fast lookup
    oblique_images = {p.stem: p for p in list_images(oblique_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Site:           {args.site}")
    print(f"Mode:           {args.mode}")
    print(f"Shorelines:     {shoreline_dir}")
    print(f"Oblique images: {oblique_dir}")
    print(f"Homographies:   {homography_dir}")
    print(f"Output:         {out_dir}")
    print(f"Found {len(npy_paths)} shoreline file(s)")
    print()

    n_ok = n_skip = 0
    for i, npy_path in enumerate(npy_paths, 1):
        plan_stem = npy_path.stem   # e.g. "1234_plan"

        # Derive oblique stem by stripping the _plan suffix
        if plan_stem.endswith(_PLAN_SUFFIX):
            oblique_stem = plan_stem[: -len(_PLAN_SUFFIX)]
        else:
            oblique_stem = plan_stem

        print(f"[{i}/{len(npy_paths)}] {npy_path.name}", end=" ... ", flush=True)

        # Locate homography file
        H_path = homography_dir / f"{oblique_stem}_H_plan_to_oblique.npy"
        if not H_path.exists():
            print(f"no homography ({H_path.name}), skipping")
            n_skip += 1
            continue

        # Locate oblique image
        if oblique_stem not in oblique_images:
            print(f"no oblique image for stem '{oblique_stem}', skipping")
            n_skip += 1
            continue

        pts = np.load(str(npy_path))
        if pts.ndim != 2 or pts.shape[1] != 2:
            print(f"unexpected shape {pts.shape}, skipping")
            n_skip += 1
            continue

        H = load_homography(H_path)
        oblique_pts = reproject_shoreline(pts, H)

        image = load_image_rgb(oblique_images[oblique_stem])
        vis = draw_shoreline(image, oblique_pts)

        out_path = out_dir / f"{oblique_stem}.jpg"
        save_visualization(vis, out_path)
        n_ok += 1
        print("done")

    print()
    print(
        f"Reprojected shorelines saved to: {out_dir}  "
        f"({n_ok} saved, {n_skip} skipped)"
    )


if __name__ == "__main__":
    main()
