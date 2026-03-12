import numpy as np
from scipy.signal import savgol_filter


def extract_shoreline(
    mask: np.ndarray,
    border_margin: int = 5,
    left_margin: int = 0,
    right_margin: int = 0,
) -> np.ndarray:
    """
    Extract shoreline (x, y) pixel coordinates from a binary mask.

    Uses a column-scan approach suited to plan-view beach imagery where the
    shoreline runs roughly left-to-right and spans the full image width.
    For each x column, all vertical transitions between masked and unmasked
    pixels are found.  Transitions near the top or bottom image border are
    excluded (those are land/sky boundaries, not the shoreline).  The
    remaining transition in each column is taken as the shoreline y position.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask (h, w).
    border_margin : int
        Transitions within this many rows of the top or bottom image edge
        are excluded.
    left_margin : int
        Number of columns to skip at the left edge of the image.
    right_margin : int
        Number of columns to skip at the right edge of the image.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Shoreline points as (x, y) float32, one per column, sorted by x.
        Empty array (shape (0, 2)) if no shoreline is found.
    """
    h, w = mask.shape
    m = (mask > 0).astype(np.int8)

    pts = []
    for x in range(left_margin, w - right_margin):
        col = m[:, x]
        # Row indices where a transition occurs (between row i and i+1)
        transitions = np.where(np.diff(col) != 0)[0]

        # Exclude transitions near the top or bottom border
        valid = transitions[
            (transitions >= border_margin) & (transitions < h - border_margin)
        ]

        if len(valid) == 0:
            continue

        # If multiple valid transitions remain (e.g. noise patches), use the
        # one closest to the vertical centre of the image as the shoreline.
        y = float(valid[np.argmin(np.abs(valid - h / 2))])
        pts.append([float(x), y])

    if not pts:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(pts, dtype=np.float32)


def smooth_shoreline(
    pts: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Smooth shoreline coordinates using a Savitzky-Golay filter.

    The points are first resampled onto a regular integer x-grid (one point
    per column) via linear interpolation; then Savitzky-Golay smoothing is
    applied to the y values.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)
        Raw shoreline points (x, y), sorted by x.
    window_length : int
        Savitzky-Golay window length (must be odd; clamped to data length).
    polyorder : int
        Polynomial order for the Savitzky-Golay filter.

    Returns
    -------
    np.ndarray, shape (M, 2)
        Smoothed (x, y) coordinates on a regular x-grid.
    """
    if len(pts) < 2:
        return pts

    x = pts[:, 0]
    y = pts[:, 1]

    # Resample onto a regular integer x-grid
    x_grid = np.arange(int(x.min()), int(x.max()) + 1, dtype=np.float32)
    y_grid = np.interp(x_grid, x, y)

    # Clamp window_length to be <= number of grid points and odd
    wl = min(window_length, len(x_grid))
    if wl % 2 == 0:
        wl -= 1
    min_wl = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3
    wl = max(wl, min_wl)

    if wl > len(x_grid):
        # Not enough points to smooth — return as-is on regular grid
        return np.column_stack([x_grid, y_grid])

    y_smooth = savgol_filter(y_grid, window_length=wl, polyorder=polyorder)
    return np.column_stack([x_grid, y_smooth])
