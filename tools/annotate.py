#!/usr/bin/env python
"""
Interactive annotation tool for CoastSnap beach segmentation.

Collect positive/negative SAM2 point prompts on a reference image and save
them to data/reference/<site>/annotations.json.

Usage:
    python tools/annotate.py --site <site_name>
    python tools/annotate.py --site <site_name> --image <path/to/image.jpg>

Controls:
    Left click   Add positive point  (green — beach/sand)
    Right click  Add negative point  (red   — water/sky/non-beach)
    z            Undo last point
    r            Reset all points
    s            Save and exit
    q            Quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import list_images, load_annotations, load_image_rgb, save_annotations

def _max_display_dim() -> int:
    """Return a display limit (px) that fits within the current screen, with margin."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        # Leave ~20% margin for taskbar, window chrome, and the status bar
        return int(min(screen_w, screen_h) * 0.80)
    except Exception:
        return 900  # fallback if tkinter unavailable


def _display_scale(h: int, w: int) -> float:
    return min(1.0, _max_display_dim() / max(h, w))


def _render(image_rgb: np.ndarray, positive: list, negative: list, scale: float) -> np.ndarray:
    """Compose the display frame: scaled image + point overlays + status bar."""
    h, w = image_rgb.shape[:2]
    dh, dw = int(h * scale), int(w * scale)
    frame = cv2.resize(image_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for x, y in positive:
        px, py = int(x * scale), int(y * scale)
        cv2.circle(frame, (px, py), 8, (0, 200, 0), -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)

    for x, y in negative:
        px, py = int(x * scale), int(y * scale)
        cv2.circle(frame, (px, py), 8, (0, 0, 200), -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)

    # Status bar
    status = (
        f"  +{len(positive)} positive (L-click)   "
        f"-{len(negative)} negative (R-click)   |   "
        "z=undo   r=reset   s=save   q=quit"
    )
    bar = np.zeros((28, dw, 3), dtype=np.uint8)
    cv2.putText(bar, status, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


def annotate(image_rgb: np.ndarray, existing: dict | None = None) -> dict | None:
    """
    Open an interactive OpenCV window to collect point annotations.
    Returns the annotations dict, or None if the user cancelled (q).
    """
    h, w = image_rgb.shape[:2]
    scale = _display_scale(h, w)

    positive: list[list[int]] = [list(p) for p in (existing or {}).get("positive_points", [])]
    negative: list[list[int]] = [list(p) for p in (existing or {}).get("negative_points", [])]
    # Each entry: ("pos" | "neg", [x, y]) — used for undo
    history: list[tuple[str, list[int]]] = []

    window = "CoastSnap Annotator"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, mx, my, flags, param):
        ix = max(0, min(w - 1, int(mx / scale)))
        iy = max(0, min(h - 1, int(my / scale)))
        if event == cv2.EVENT_LBUTTONDOWN:
            positive.append([ix, iy])
            history.append(("pos", [ix, iy]))
        elif event == cv2.EVENT_RBUTTONDOWN:
            negative.append([ix, iy])
            history.append(("neg", [ix, iy]))

    cv2.setMouseCallback(window, on_mouse)

    result = None
    while True:
        cv2.imshow(window, _render(image_rgb, positive, negative, scale))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            result = {"positive_points": positive, "negative_points": negative}
            break
        elif key == ord("z") and history:
            kind, pt = history.pop()
            target = positive if kind == "pos" else negative
            if pt in target:
                target.remove(pt)
        elif key == ord("r"):
            positive.clear()
            negative.clear()
            history.clear()

    cv2.destroyAllWindows()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a CoastSnap reference image with SAM2 point prompts."
    )
    parser.add_argument("--site", required=True, help="Site name (e.g. narrabeen)")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file. Defaults to the first image in data/reference/<site>/",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Repository root directory (default: current directory)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ref_dir = data_root / "data" / "reference" / args.site
    ref_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image)
    else:
        candidates = list_images(ref_dir)
        if not candidates:
            print(f"No images found in {ref_dir}.")
            print("Place a reference image there, or pass --image <path>.")
            sys.exit(1)
        image_path = candidates[0]

    print(f"Image:  {image_path}")
    image_rgb = load_image_rgb(image_path)

    ann_path = ref_dir / "annotations.json"
    existing = None
    if ann_path.exists():
        existing = load_annotations(ann_path)
        n_pos = len(existing.get("positive_points", []))
        n_neg = len(existing.get("negative_points", []))
        print(f"Loaded existing annotations: {n_pos} positive, {n_neg} negative")

    result = annotate(image_rgb, existing)

    if result is None:
        print("Cancelled — nothing saved.")
        return

    result["image"] = image_path.name
    save_annotations(result, ann_path)
    print(
        f"Saved {len(result['positive_points'])} positive and "
        f"{len(result['negative_points'])} negative points → {ann_path}"
    )


if __name__ == "__main__":
    main()
