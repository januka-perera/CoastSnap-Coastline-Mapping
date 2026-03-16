import numpy as np
from scipy.signal import savgol_filter
from skimage.measure import find_contours


def extract_shoreline(
    mask: np.ndarray,
    border_margin: int = 5,
    left_margin: int = 0,
    right_margin: int = 0,
    ocean_side: str | None = None,
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
    ocean_side : str or None
        Which image edge the ocean is closest to: ``"top"``, ``"bottom"``,
        ``"left"``, or ``"right"``.  When set, selects the transition nearest
        that edge rather than nearest the image centre.  ``None`` (default)
        keeps the existing closest-to-centre behaviour.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Shoreline points as (x, y) float32, one per column, sorted by x.
        Empty array (shape (0, 2)) if no shoreline is found.
    """
    h, w = mask.shape
    m = (mask > 0).astype(np.int8)

    pts = []

    if ocean_side in ("left", "right"):
        # Row scan: for each y row find the horizontal transition closest to the
        # ocean edge.  Suited to vertical shorelines where ocean is left or right.
        for y in range(border_margin, h - border_margin):
            row = m[y, :]
            transitions = np.where(np.diff(row) != 0)[0]
            valid = transitions[
                (transitions >= border_margin) & (transitions < w - border_margin)
            ]
            if len(valid) == 0:
                continue
            if ocean_side == "right":
                x = float(valid[-1])   # rightmost transition
            else:
                x = float(valid[0])    # leftmost transition
            pts.append([x, float(y)])
    else:
        # Column scan: for each x column find the vertical transition closest to
        # the ocean edge.  Suited to horizontal shorelines where ocean is top/bottom.
        for x in range(left_margin, w - right_margin):
            col = m[:, x]
            transitions = np.where(np.diff(col) != 0)[0]
            valid = transitions[
                (transitions >= border_margin) & (transitions < h - border_margin)
            ]
            if len(valid) == 0:
                continue
            if ocean_side == "bottom":
                y = float(valid[-1])
            elif ocean_side == "top":
                y = float(valid[0])
            else:
                y = float(valid[np.argmin(np.abs(valid - h / 2))])
            pts.append([float(x), y])

    if not pts:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(pts, dtype=np.float32)


def extract_shoreline_from_logits(
    logit: np.ndarray,
    masked_region: str = "sand",
    border_margin: int = 5,
    min_length: int = 50,
    left_margin: int = 0,
    right_margin: int = 0,
    ocean_side: str | None = None,
) -> np.ndarray:
    """
    Extract shoreline (x, y) coordinates as the zero contour of a SAM2 logit field.

    SAM2 logits are positive where the model predicts the segmented class and
    negative elsewhere. The zero crossing is the subpixel-stable decision boundary
    — equivalent to a 0.5 probability threshold but without quantisation to pixel
    edges. Marching squares is used so the result works for both plan-view and
    oblique images with no assumption about shoreline orientation.

    Parameters
    ----------
    logit : np.ndarray, float32 (H, W)
        Raw SAM2 logit field at image resolution.
    masked_region : str
        ``"sand"`` if the segmented class is dry/wet sand (default), or
        ``"ocean"`` if the segmented class is water. When ``"ocean"``, the
        logit is negated so the zero contour always represents the sand side.
    border_margin : int
        Contour points within this many pixels of any image edge are dropped.
    min_length : int
        Minimum number of points a contour must have to be considered. Filters
        out small noise blobs.
    left_margin : int
        Columns to skip at the left image edge. Contour points with x < this
        value are excluded.
    right_margin : int
        Columns to skip at the right image edge. Contour points with x > w - this
        value are excluded.
    ocean_side : str or None
        Which image edge the ocean is closest to: ``"top"``, ``"bottom"``,
        ``"left"``, or ``"right"``.  When set, the closed contour is split at
        its x-extrema into two halves and the half whose mean position is
        closest to the specified ocean edge is selected.  This reliably
        discards headland and vegetation boundaries on the opposite side.
        ``None`` (default) returns the longest contour without splitting.

    Returns
    -------
    np.ndarray, shape (N, 2), float32
        Shoreline points as (x, y). Returns empty array if no valid contour is found.
    """
    h, w = logit.shape
    field = logit if masked_region == "sand" else -logit

    # find_contours returns a list of (N, 2) arrays in (row, col) = (y, x) order
    contours = find_contours(field, level=0.0)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    # Filter to contours that are long enough and have interior (non-border) points
    valid = []
    for c in contours:
        if len(c) < min_length:
            continue
        rows, cols = c[:, 0], c[:, 1]
        interior = (
            (rows >= border_margin) & (rows < h - border_margin) &
            (cols >= border_margin) & (cols < w - border_margin)
        )
        if interior.sum() < min_length:
            continue
        valid.append(c)  # keep the full closed contour for half-selection

    if not valid:
        return np.empty((0, 2), dtype=np.float32)

    longest = max(valid, key=len)

    if ocean_side is not None:
        # Split the closed contour at its two extrema along the axis perpendicular
        # to the shoreline.  For a horizontal shoreline (ocean top/bottom) split at
        # x-extrema (leftmost/rightmost points); for a vertical shoreline (ocean
        # left/right) split at y-extrema (topmost/bottommost points).  This gives
        # two paths each running the full length of the beach.  Select the half
        # whose mean position is closest to the specified ocean edge.
        if ocean_side in ("top", "bottom"):
            axis = 1  # split at x-extrema; compare by row (y)
            compare_axis = 0
        else:  # "left" or "right"
            axis = 0  # split at y-extrema; compare by col (x)
            compare_axis = 1

        values = longest[:, axis]
        lo_idx = int(np.argmin(values))
        hi_idx = int(np.argmax(values))

        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx

        half1 = longest[lo_idx:hi_idx + 1]
        half2 = np.concatenate([longest[hi_idx:], longest[:lo_idx + 1]])

        if ocean_side in ("bottom", "right"):
            chosen = half1 if half1[:, compare_axis].mean() >= half2[:, compare_axis].mean() else half2
        else:  # "top" or "left"
            chosen = half1 if half1[:, compare_axis].mean() <= half2[:, compare_axis].mean() else half2
    else:
        chosen = longest

    # Apply border and margin filters to the chosen segment
    rows, cols_arr = chosen[:, 0], chosen[:, 1]
    keep = (
        (rows >= border_margin) & (rows < h - border_margin) &
        (cols_arr >= border_margin) & (cols_arr < w - border_margin) &
        (cols_arr >= left_margin) & (cols_arr <= w - right_margin)
    )
    chosen = chosen[keep]

    if len(chosen) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.column_stack([chosen[:, 1], chosen[:, 0]]).astype(np.float32)


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
