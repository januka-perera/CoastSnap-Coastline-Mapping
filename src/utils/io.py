import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3) uint8."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(directory: str | Path) -> list[Path]:
    """Return sorted list of image paths in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_annotations(path: str | Path) -> list[dict]:
    """Load point annotations from JSON file.

    Returns a list of annotation dicts (one per annotated frame).
    Old single-dict files are automatically wrapped in a list.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def save_annotations(annotations: list[dict], path: str | Path) -> None:
    """Save point annotations to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save binary mask as PNG (0 or 255)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))
