#!/usr/bin/env python
"""
Overlay extracted shorelines onto georeferenced GeoTIFF plan-view images.

Plan-view pixels map 1:1 to GeoTIFF pixels (both produced by rectify_plan_view
with the same xlim/ylim/dx), so the shoreline (col, row) coordinates saved by
run_shoreline.py can be drawn directly onto the GeoTIFF raster without any
coordinate conversion.

Usage
-----
    python run_shoreline_geotiff.py \\
        --site manly-plan \\
        --mode video \\
        --geotiff-dir /path/to/geotiffs \\
        --output-dir  outputs/shorelines/geotiff/video/manly-plan

Reads
-----
    outputs/shorelines/<mode>/<site>/<stem>.npy  — shoreline pixel coords

Writes
------
    <output-dir>/<stem>.tif  — GeoTIFF with shoreline drawn in red
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp


def draw_shoreline_on_array(
    rgb: np.ndarray,
    pts: np.ndarray,
    colour: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw shoreline points onto an (H, W, 3) uint8 RGB array.

    Parameters
    ----------
    rgb       : (H, W, 3) uint8 array — modified in-place.
    pts       : (N, 2) float32 array of (col, row) shoreline coordinates.
    colour    : RGB tuple to draw with (default: red).
    thickness : radius in pixels around each point (default: 2).
    """
    h, w = rgb.shape[:2]
    cols = np.round(pts[:, 0]).astype(int)
    rows = np.round(pts[:, 1]).astype(int)

    for dc in range(-thickness, thickness + 1):
        for dr in range(-thickness, thickness + 1):
            c = np.clip(cols + dc, 0, w - 1)
            r = np.clip(rows + dr, 0, h - 1)
            rgb[r, c] = colour

    return rgb


def overlay_shoreline_on_geotiff(
    pts: np.ndarray,
    geotiff_path: Path,
    output_path: Path,
    colour: tuple = (255, 0, 0),
    thickness: int = 2,
) -> None:
    """
    Read a GeoTIFF, draw the shoreline, and write a new GeoTIFF preserving
    the spatial reference and transform.
    """
    with rasterio.open(geotiff_path) as src:
        profile = src.profile.copy()
        data = src.read()          # (bands, H, W)
        n_bands = data.shape[0]

    # Build an RGB working copy (ignore alpha if present)
    rgb = np.stack([data[0], data[1], data[2]], axis=-1).copy()  # (H, W, 3)

    draw_shoreline_on_array(rgb, pts, colour=colour, thickness=thickness)

    # Write back — keep all original bands, replace RGB
    data[0] = rgb[:, :, 0]
    data[1] = rgb[:, :, 1]
    data[2] = rgb[:, :, 2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        if n_bands >= 4:
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Draw extracted shorelines onto plan-view GeoTIFFs."
    )
    parser.add_argument("--site",        required=True,
                        help="Site name (e.g. manly-plan)")
    parser.add_argument("--mode",        required=True, choices=["points", "video"],
                        help="Segmentation mode used to produce the shorelines")
    parser.add_argument("--geotiff-dir", required=True,
                        help="Directory containing the input GeoTIFF files")
    parser.add_argument("--output-dir",  default=None,
                        help="Output directory for shoreline-overlaid GeoTIFFs "
                             "(default: outputs/shorelines/geotiff/<mode>/<site>)")
    parser.add_argument("--shoreline-dir", default=None,
                        help="Directory containing .npy shoreline files "
                             "(default: outputs/shorelines/<mode>/<site>)")
    parser.add_argument("--thickness",   type=int, default=2,
                        help="Line thickness in pixels (default: 2)")
    args = parser.parse_args()

    shoreline_dir = Path(args.shoreline_dir) if args.shoreline_dir else (
        Path("outputs/shorelines") / args.mode / args.site
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("outputs/shorelines/geotiff") / args.mode / args.site
    )
    geotiff_dir = Path(args.geotiff_dir)

    if not shoreline_dir.exists():
        print(f"Shoreline directory not found: {shoreline_dir}")
        print(f"Run  python run_shoreline.py --site {args.site} --mode {args.mode}  first.")
        sys.exit(1)

    npy_paths = sorted(shoreline_dir.glob("*.npy"))
    if not npy_paths:
        print(f"No .npy shoreline files found in {shoreline_dir}")
        sys.exit(1)

    # Index GeoTIFFs by stem for fast lookup
    geotiffs = {p.stem: p for p in geotiff_dir.glob("*.tif")}
    if not geotiffs:
        print(f"No .tif files found in {geotiff_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Site:         {args.site}")
    print(f"Mode:         {args.mode}")
    print(f"Shorelines:   {shoreline_dir}  ({len(npy_paths)} files)")
    print(f"GeoTIFFs:     {geotiff_dir}  ({len(geotiffs)} files)")
    print(f"Output:       {output_dir}")
    print()

    n_ok = n_missing = 0
    for npy_path in npy_paths:
        stem = npy_path.stem
        print(f"  {stem}", end=" ... ", flush=True)

        if stem not in geotiffs:
            print(f"no matching GeoTIFF found, skipping")
            n_missing += 1
            continue

        pts = np.load(npy_path)
        if len(pts) == 0:
            print("empty shoreline, skipping")
            n_missing += 1
            continue

        out_path = output_dir / f"{stem}.tif"
        overlay_shoreline_on_geotiff(
            pts, geotiffs[stem], out_path, thickness=args.thickness
        )
        n_ok += 1
        print("done")

    print()
    print(f"Done.  {n_ok}/{len(npy_paths)} GeoTIFFs written to {output_dir}")
    if n_missing:
        print(f"       {n_missing} skipped (no matching GeoTIFF or empty shoreline)")


if __name__ == "__main__":
    main()
