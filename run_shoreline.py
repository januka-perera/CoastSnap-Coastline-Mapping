#!/usr/bin/env python
"""
Extract and visualize shorelines from segmentation masks.

For each mask in outputs/masks/<mode>/<site>/, the shoreline is detected as
the mask boundary that does not lie on the image border, smoothed with a
Savitzky-Golay filter, and drawn as a cyan line over the original image.

Usage:
    python run_shoreline.py --site <site_name> --mode points
    python run_shoreline.py --site <site_name> --mode video
    python run_shoreline.py --site <site_name> --mode points --masked-region ocean
    python run_shoreline.py --site <site_name> --mode points --config configs/config.yaml

Reads:
    outputs/masks/<mode>/<site>/   — binary PNG masks
    data/raw/<site>/               — original images (for background)

Writes:
    outputs/shorelines/<mode>/<site>/  — shoreline overlay JPEGs
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shoreline.extractor import extract_shoreline, extract_shoreline_from_logits, smooth_shoreline
from src.utils.io import list_images, load_image_rgb, load_logit
from src.utils.visualization import draw_shoreline, save_visualization


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    return img > 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract and visualize shorelines from segmentation masks."
    )
    parser.add_argument("--site",   required=True, help="Site name (e.g. narrabeen)")
    parser.add_argument("--mode",   required=True, choices=["points", "video"],
                        help="Segmentation mode the masks were produced with")
    parser.add_argument(
        "--masked-region",
        default="sand",
        choices=["sand", "ocean"],
        help="Which region was segmented (default: sand). "
             "If 'ocean', the mask is inverted before extraction so the "
             "sand-facing boundary is treated as the shoreline.",
    )
    parser.add_argument(
        "--ocean-side",
        default=None,
        choices=["top", "bottom", "left", "right"],
        help="Which image edge the ocean is closest to. When set, the contour is "
             "split at its x-extrema and the ocean-facing half is selected, "
             "discarding headland and vegetation boundaries on the opposite side.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--left-margin", type=int, default=0,
        help="Columns to skip at the left image edge when scanning for shoreline (default: 0)",
    )
    parser.add_argument(
        "--right-margin", type=int, default=0,
        help="Columns to skip at the right image edge when scanning for shoreline (default: 0)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    masks_dir  = Path(cfg["output"]["masks_dir"]) / args.mode / args.site
    logits_dir = Path(cfg["output"]["masks_dir"]).parent / "logits" / args.mode / args.site
    raw_dir    = Path(cfg["data"]["raw_dir"]) / args.site
    out_dir    = Path("outputs/shorelines") / args.mode / args.site

    if not masks_dir.exists():
        run_script = "run_segmentation_video.py" if args.mode == "video" else "run_segmentation.py"
        print(f"No masks found at {masks_dir}")
        print(f"Run  python {run_script} --site {args.site}  first.")
        sys.exit(1)

    mask_paths = sorted(masks_dir.glob("*.png"))
    if not mask_paths:
        print(f"No mask PNGs found in {masks_dir}")
        sys.exit(1)

    originals = {p.stem: p for p in list_images(raw_dir)} if raw_dir.exists() else {}
    out_dir.mkdir(parents=True, exist_ok=True)

    use_logits = logits_dir.exists() and any(logits_dir.glob("*.npy"))
    source_label = "logits (subpixel)" if use_logits else "binary masks"

    print(f"Site:           {args.site}")
    print(f"Mode:           {args.mode}")
    print(f"Masked region:  {args.masked_region}")
    print(f"Ocean side:     {args.ocean_side or 'auto (longest contour)'}")
    print(f"Source:         {source_label}")
    print(f"Masks:          {len(mask_paths)}")
    print()

    n_ok = 0
    for i, mask_path in enumerate(mask_paths, 1):
        stem = mask_path.stem
        print(f"[{i}/{len(mask_paths)}] {mask_path.name}", end=" ... ", flush=True)

        mask = load_mask(mask_path)
        if args.masked_region == "ocean":
            mask = ~mask

        logit_path = logits_dir / f"{stem}.npy"
        if use_logits and logit_path.exists():
            logit = load_logit(logit_path)
            pts = extract_shoreline_from_logits(
                logit,
                masked_region=args.masked_region,
                left_margin=args.left_margin,
                right_margin=args.right_margin,
                ocean_side=args.ocean_side,
            )
        else:
            pts = extract_shoreline(
                mask,
                left_margin=args.left_margin,
                right_margin=args.right_margin,
                ocean_side=args.ocean_side,
            )

        if len(pts) == 0:
            print("no shoreline found, skipping")
            continue

        # pts = smooth_shoreline(pts)

        # Build background: original image with mask overlay for context
        if stem in originals:
            image = load_image_rgb(originals[stem])
        else:
            h, w = mask.shape
            image = np.full((h, w, 3), 40, dtype=np.uint8)

        vis = draw_shoreline(image, pts)

        save_visualization(vis, out_dir / f"{stem}.jpg")
        np.save(out_dir / f"{stem}.npy", pts)
        n_ok += 1
        print("done")

    print()
    print(f"Shoreline visualizations saved to: {out_dir}  ({n_ok}/{len(mask_paths)} processed)")


if __name__ == "__main__":
    main()
