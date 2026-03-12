#!/usr/bin/env python
"""
Archive segmentation results for a site.

Copies masks, visualizations, and annotations into a timestamped folder under
archive/<site>/<timestamp>/ so good runs are preserved before re-processing.

Usage:
    python tools/archive_results.py --site <site_name> --mode points
    python tools/archive_results.py --site <site_name> --mode video
    python tools/archive_results.py --site <site_name> --mode points --config configs/config.yaml
    python tools/archive_results.py --site <site_name> --mode points --note "good tide conditions"
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def copy_dir(src: Path, dst: Path) -> int:
    """Copy all files from src into dst. Returns number of files copied."""
    if not src.exists() or not any(src.iterdir()):
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Archive segmentation results for a CoastSnap site.")
    parser.add_argument("--site", required=True, help="Site name (e.g. narrabeen)")
    parser.add_argument("--mode", required=True, choices=["points", "video"], help="Segmentation mode to archive (points or video)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--note", default="", help="Optional note saved alongside the archive")
    args = parser.parse_args()

    cfg = load_config(args.config)

    masks_src      = Path(cfg["output"]["masks_dir"]) / args.mode / args.site
    vis_src        = Path(cfg["output"]["visualizations_dir"]) / args.mode / args.site
    annotations_src = Path(cfg["data"]["reference_dir"]) / args.site / "annotations.json"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    archive_dir = Path("archive") / args.site / timestamp

    # Check there's something to archive
    if not masks_src.exists() and not vis_src.exists() and not annotations_src.exists():
        print(f"Nothing to archive — no outputs found for site '{args.site}'.")
        sys.exit(1)

    archive_dir.mkdir(parents=True, exist_ok=True)

    n_masks = copy_dir(masks_src, archive_dir / "masks")
    n_vis   = copy_dir(vis_src,   archive_dir / "visualizations")

    n_ann = 0
    if annotations_src.exists():
        shutil.copy2(annotations_src, archive_dir / "annotations.json")
        n_ann = 1

    # Write optional note
    if args.note:
        (archive_dir / "note.txt").write_text(args.note)

    print(f"Archived to: {archive_dir}")
    print(f"  masks:          {n_masks} files")
    print(f"  visualizations: {n_vis} files")
    print(f"  annotations:    {'yes' if n_ann else 'not found'}")
    if args.note:
        print(f"  note:           {args.note}")


if __name__ == "__main__":
    main()
