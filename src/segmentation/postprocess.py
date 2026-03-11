import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask."""
    return ndimage.binary_fill_holes(mask)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Post-process a segmentation mask:
    1. Keep the largest connected component.
    2. Fill interior holes.
    """
    mask = largest_component(mask.astype(bool))
    mask = fill_holes(mask)
    return mask
