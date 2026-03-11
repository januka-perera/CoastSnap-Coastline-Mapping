from pathlib import Path

import cv2
import numpy as np


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Return RGB image with mask overlaid as a semi-transparent pink highlight."""
    color_mask = np.zeros_like(image_rgb)
    color_mask[mask > 0] = [255, 105, 180]
    return cv2.addWeighted(image_rgb, 1.0, color_mask, alpha, 0)


def draw_points(
    image_rgb: np.ndarray,
    positive_points: list[list[int]],
    negative_points: list[list[int]],
    radius: int = 8,
) -> np.ndarray:
    """Draw positive (green) and negative (red) points on a copy of the image."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for x, y in positive_points:
        cv2.circle(img, (x, y), radius, (0, 200, 0), -1)
        cv2.circle(img, (x, y), radius, (255, 255, 255), 2)
    for x, y in negative_points:
        cv2.circle(img, (x, y), radius, (0, 0, 200), -1)
        cv2.circle(img, (x, y), radius, (255, 255, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_visualization(image_rgb: np.ndarray, path: str | Path) -> None:
    """Save RGB image to disk as JPEG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
