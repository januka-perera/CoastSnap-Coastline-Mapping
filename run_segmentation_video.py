#!/usr/bin/env python
"""
Run SAM2 beach segmentation using the video predictor.

Instead of prompting each image independently with fixed point coordinates,
this approach prompts a single reference frame and propagates the mask
through all frames using SAM2's temporal tracking.

Usage:
    python run_segmentation_video.py --site <site_name>
    python run_segmentation_video.py --site <site_name> --ref-frame 0
    python run_segmentation_video.py --site <site_name> --config configs/config.yaml

Reads:
    data/raw/<site>/          — images to segment
    data/reference/<site>/    — annotations.json with point prompts

Writes:
    outputs/masks/<site>/         — binary PNG masks (0 / 255)
    outputs/visualizations/<site>/ — mask overlay JPEGs for QC
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import cv2
import numpy as np
import yaml

from sam2.build_sam import build_sam2_video_predictor
from src.segmentation.predictor import _CHECKPOINT_TO_CONFIG
from src.segmentation.postprocess import clean_mask
from src.utils.io import list_images, load_annotations, load_image_rgb, save_mask
from src.utils.visualization import draw_points, overlay_mask, save_visualization


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_frames(images: list[Path], tmp_dir: Path) -> None:
    """
    Copy images into tmp_dir as numerically named JPEGs (00000.jpg, 00001.jpg, ...),
    which is the format SAM2's video predictor requires.
    """
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(tmp_dir / f"{idx:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video-predictor beach segmentation for a CoastSnap site."
    )
    parser.add_argument("--site",      required=True, help="Site name (e.g. narrabeen)")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        help="Index of the frame to use as the reference prompt (default: 0).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir   = Path(cfg["data"]["raw_dir"]) / args.site
    ref_dir   = Path(cfg["data"]["reference_dir"]) / args.site
    masks_dir = Path(cfg["output"]["masks_dir"]) / "video" / args.site
    vis_dir   = Path(cfg["output"]["visualizations_dir"]) / "video" / args.site

    # Load annotations
    ann_path = ref_dir / "annotations.json"
    if not ann_path.exists():
        print(f"No annotations found at {ann_path}.")
        print("Run  python tools/annotate.py --site <site>  first.")
        sys.exit(1)

    annotations    = load_annotations(ann_path)
    positive_points = annotations["positive_points"]
    negative_points = annotations.get("negative_points", [])

    if not positive_points:
        print("annotations.json contains no positive points. Add at least one.")
        sys.exit(1)

    images = list_images(raw_dir)
    if not images:
        print(f"No images found in {raw_dir}")
        sys.exit(1)

    ref_frame_idx = args.ref_frame
    if ref_frame_idx >= len(images):
        print(f"--ref-frame {ref_frame_idx} is out of range (only {len(images)} images).")
        sys.exit(1)

    checkpoint = Path(cfg["model"]["checkpoint"])
    config     = _CHECKPOINT_TO_CONFIG.get(checkpoint.stem)
    if config is None:
        print(f"Unknown checkpoint '{checkpoint.stem}'.")
        sys.exit(1)

    print(f"Site:        {args.site}")
    print(f"Images:      {len(images)}")
    print(f"Ref frame:   {ref_frame_idx} ({images[ref_frame_idx].name})")
    print(f"Checkpoint:  {cfg['model']['checkpoint']}")
    print(f"Device:      {cfg['model']['device']}")
    print()

    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=cfg["model"]["device"],
    )

    all_points = positive_points + negative_points
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)
    point_coords = np.array(all_points, dtype=np.float32)
    point_labels = np.array(all_labels, dtype=np.int32)

    masks_by_frame: dict[int, np.ndarray] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Preparing frames ...")
        prepare_frames(images, Path(tmp_dir))

        print("Initialising video predictor ...")
        inference_state = predictor.init_state(video_path=tmp_dir)

        # Prompt the reference frame with annotated points
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ref_frame_idx,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
            normalize_coords=True,
        )

        # Propagate forward (ref_frame → last frame)
        print(f"Propagating forward from frame {ref_frame_idx} ...")
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks_by_frame[frame_idx] = (mask_logits[0, 0].cpu().numpy() > 0)

        # Propagate backward (ref_frame → first frame) if needed
        if ref_frame_idx > 0:
            print(f"Propagating backward from frame {ref_frame_idx} ...")
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if frame_idx not in masks_by_frame:
                    masks_by_frame[frame_idx] = (mask_logits[0, 0].cpu().numpy() > 0)

    # Save masks and visualizations
    print()
    for i, img_path in enumerate(images):
        print(f"[{i + 1}/{len(images)}] {img_path.name}", end=" ... ", flush=True)

        image = load_image_rgb(img_path)
        h, w = image.shape[:2]

        raw_mask = masks_by_frame[i]
        if raw_mask.shape != (h, w):
            raw_mask = cv2.resize(
                raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        mask = clean_mask(raw_mask)

        save_mask(mask, masks_dir / f"{img_path.stem}.png")

        vis = overlay_mask(image, mask)
        if i == ref_frame_idx:
            vis = draw_points(vis, positive_points, negative_points)
        save_visualization(vis, vis_dir / f"{img_path.stem}.jpg")

        print("done")

    print()
    print(f"Masks saved to:          {masks_dir}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
