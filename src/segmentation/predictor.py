from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _mask_to_logits(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask (H, W) bool to SAM2 mask_input format (1, 256, 256) float32.
    SAM2 expects low-resolution logits; we use ±10 to represent hard foreground/background.
    """
    resized = cv2.resize(binary_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    logits = np.where(resized > 0, 10.0, -10.0).astype(np.float32)
    return logits[np.newaxis]  # (1, 256, 256)

# Maps checkpoint filename stem → SAM2.1 Hydra config name
_CHECKPOINT_TO_CONFIG = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class BeachSegmentor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        checkpoint = Path(checkpoint)
        config = _CHECKPOINT_TO_CONFIG.get(checkpoint.stem)
        if config is None:
            raise ValueError(
                f"Unknown checkpoint '{checkpoint.stem}'. "
                f"Expected one of: {list(_CHECKPOINT_TO_CONFIG)}"
            )
        model = build_sam2(config, str(checkpoint), device=device)
        self._predictor = SAM2ImagePredictor(model)

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Encode a new image. Must be called before predict()."""
        with torch.inference_mode():
            self._predictor.set_image(image_rgb)

    def predict(
        self,
        positive_points: list[list[int]],
        negative_points: list[list[int]] | None = None,
    ) -> np.ndarray:
        """
        Run SAM2 with point prompts. Returns the best binary mask (H, W) bool.
        Points are [x, y] pixel coordinates.
        """
        all_points = positive_points + (negative_points or [])
        all_labels = [1] * len(positive_points) + [0] * len(negative_points or [])

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        return masks[best_idx]  # bool (H, W)

    def predict_from_mask(self, reference_mask: np.ndarray) -> np.ndarray:
        """
        Run SAM2 using a reference mask as the prompt instead of point coordinates.
        The reference mask is converted to SAM2 logit format and passed as mask_input.
        Returns the best binary mask (H, W) bool.
        """
        mask_input = _mask_to_logits(reference_mask)

        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=mask_input,
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        return masks[best_idx]  # bool (H, W)
