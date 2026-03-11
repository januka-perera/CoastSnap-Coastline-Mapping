#!/usr/bin/env python
"""
Run SAM2 beach segmentation on all images for a given site.

Usage:
    python run_segmentation.py --site <site_name>
    python run_segmentation.py --site <site_name> --config configs/config.yaml

Reads:
    data/raw/<site>/              — images to segment
    data/reference/<site>/        — annotations.json with point prompts

Writes:
    outputs/masks/<site>/         — binary PNG masks (0 / 255)
    outputs/visualizations/<site>/ — mask overlay JPEGs for QC
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from src.segmentation.postprocess import clean_mask
from src.segmentation.predictor import BeachSegmentor
from src.utils.io import list_images, load_annotations, load_image_rgb, save_mask
from src.utils.visualization import draw_points, overlay_mask, save_visualization


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 beach segmentation for a CoastSnap site.")
    parser.add_argument("--site", required=True, help="Site name (e.g. narrabeen)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg["data"]["raw_dir"]) / args.site
    ref_dir = Path(cfg["data"]["reference_dir"]) / args.site
    masks_dir = Path(cfg["output"]["masks_dir"]) / "points" / args.site
    vis_dir = Path(cfg["output"]["visualizations_dir"]) / "points" / args.site

    # Load annotations
    ann_path = ref_dir / "annotations.json"
    if not ann_path.exists():
        print(f"No annotations found at {ann_path}.")
        print("Run  python tools/annotate.py --site <site>  first.")
        sys.exit(1)

    annotations = load_annotations(ann_path)
    positive_points = annotations["positive_points"]
    negative_points = annotations.get("negative_points", [])

    if not positive_points:
        print("annotations.json contains no positive points. Add at least one.")
        sys.exit(1)

    # Collect images
    images = list_images(raw_dir)
    if not images:
        print(f"No images found in {raw_dir}")
        sys.exit(1)

    print(f"Site:       {args.site}")
    print(f"Images:     {len(images)}")
    print(f"Checkpoint: {cfg['model']['checkpoint']}")
    print(f"Device:     {cfg['model']['device']}")
    print()

    segmentor = BeachSegmentor(
        checkpoint=cfg["model"]["checkpoint"],
        device=cfg["model"]["device"],
    )

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}", end=" ... ", flush=True)

        image = load_image_rgb(img_path)
        segmentor.set_image(image)
        mask = segmentor.predict(positive_points, negative_points)
        mask = clean_mask(mask)

        save_mask(mask, masks_dir / f"{img_path.stem}.png")

        vis = overlay_mask(image, mask)
        vis = draw_points(vis, positive_points, negative_points)
        save_visualization(vis, vis_dir / f"{img_path.stem}.jpg")

        print("done")

    print()
    print(f"Masks saved to:          {masks_dir}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
