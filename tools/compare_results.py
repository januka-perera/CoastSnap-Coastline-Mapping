#!/usr/bin/env python
"""
Compare segmentation masks from two or more model runs side by side.

For each image present in all mask directories, produces a multi-panel
comparison (original | model 1 | model 2 | ...) and prints a summary table
with mask coverage and pairwise IoU between all models.

Usage:
    python tools/compare_results.py \
        --site <site> \
        --dirs  outputs/masks/tiny/manly-plan \
                outputs/masks/small/manly-plan \
                outputs/masks/base_plus/manly-plan \
        --labels tiny small base_plus

    # or comparing archives:
    python tools/compare_results.py \
        --site <site> \
        --dirs  archive/<site>/2026-03-11_10-00/masks \
                archive/<site>/2026-03-11_10-30/masks \
        --labels "run 1" "run 2"
"""

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import list_images, load_image_rgb
from src.utils.visualization import overlay_mask


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask PNG as a bool array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    return img > 0


def make_label_bar(width: int, text: str, height: int = 32) -> np.ndarray:
    """Create a dark label bar with centred white text."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (width - tw) // 2)
    y = (height + th) // 2
    cv2.putText(bar, text, (x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return bar


def make_comparison(
    original: np.ndarray,
    masks: list[np.ndarray],
    labels: list[str],
    max_width: int = 1800,
) -> np.ndarray:
    """Build a multi-panel comparison: original | mask1 | mask2 | ..."""
    h, w = original.shape[:2]
    n_panels = 1 + len(masks)
    panel_w = min(w, max_width // n_panels)
    panel_h = int(h * panel_w / w)

    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    columns = [np.vstack([
        make_label_bar(panel_w, "Original"),
        resize(cv2.cvtColor(original, cv2.COLOR_RGB2BGR)),
    ])]

    for mask, label in zip(masks, labels):
        panel = resize(cv2.cvtColor(overlay_mask(original, mask), cv2.COLOR_RGB2BGR))
        columns.append(np.vstack([make_label_bar(panel_w, label), panel]))

    return np.hstack(columns)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def coverage(mask: np.ndarray) -> float:
    return float(mask.sum() / mask.size * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare segmentation masks from two or more model runs."
    )
    parser.add_argument("--site",   required=True, help="Site name")
    parser.add_argument("--dirs",   required=True, nargs="+", help="Mask directories (2 or more)")
    parser.add_argument("--labels", nargs="+", default=None, help="Display labels (one per dir)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default=None,
                        help="Directory of original images. Defaults to data/raw/<site> from config.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Defaults to outputs/comparisons/<site>/.")
    args = parser.parse_args()

    if len(args.dirs) < 2:
        print("At least two --dirs are required.")
        sys.exit(1)

    labels = args.labels or [f"Model {chr(65 + i)}" for i in range(len(args.dirs))]
    if len(labels) != len(args.dirs):
        print(f"Number of --labels ({len(labels)}) must match number of --dirs ({len(args.dirs)}).")
        sys.exit(1)

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(cfg["data"]["raw_dir"]) / args.site
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/comparisons") / args.site
    dirs    = [Path(d) for d in args.dirs]

    for d in [*dirs, raw_dir]:
        if not d.exists():
            print(f"Directory not found: {d}")
            sys.exit(1)

    # Build {stem: path} index per directory
    mask_indexes = [
        {p.stem: p for p in d.iterdir() if p.suffix.lower() == ".png"}
        for d in dirs
    ]

    # Only process images present in every directory
    common = sorted(set.intersection(*[set(idx.keys()) for idx in mask_indexes]))
    if not common:
        print("No images found in common across all directories.")
        sys.exit(1)

    originals = {p.stem: p for p in list_images(raw_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)

    label_str = " | ".join(labels)
    print(f"Comparing {len(common)} images: {label_str}")
    print()

    # Header: image name + coverage per model + pairwise IoU
    pairs = list(itertools.combinations(range(len(labels)), 2))
    cov_headers  = "  ".join(f"{lbl[:8]:>8}" for lbl in labels)
    iou_headers  = "  ".join(f"IoU({labels[i][0]},{labels[j][0]})" for i, j in pairs)
    print(f"{'Image':<28}  {cov_headers}  {iou_headers}")
    print("-" * (28 + 2 + len(labels) * 10 + len(pairs) * 12))

    for stem in common:
        masks = [load_mask(idx[stem]) for idx in mask_indexes]

        if stem in originals:
            original = load_image_rgb(originals[stem])
        else:
            h, w = masks[0].shape
            original = np.full((h, w, 3), 40, dtype=np.uint8)

        panel = make_comparison(original, masks, labels)
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        covs     = [coverage(m) for m in masks]
        iou_vals = [iou(masks[i], masks[j]) for i, j in pairs]

        cov_str = "  ".join(f"{c:>7.1f}%" for c in covs)
        iou_str = "  ".join(f"{v:>10.3f}" for v in iou_vals)
        print(f"{stem:<28}  {cov_str}  {iou_str}")

    print()
    print(f"Comparison images saved to: {out_dir}")

    # Warn about images missing from some (but not all) directories
    all_stems = set.union(*[set(idx.keys()) for idx in mask_indexes])
    for label, idx in zip(labels, mask_indexes):
        missing = sorted(all_stems - idx.keys())
        if missing:
            print(f"Missing from {label!r}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
