"""
Module for extracting the liquid Region-of-Interest (ROI) from reaction tube photographs,
excluding glass glare, tube meniscus, labels, and background.
"""
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageOps
from src.config import DEFAULT_ROI


class TubeROIDetector:
    """
    Detects and crops the representative liquid reaction zone in test tube photographs.
    """

    def __init__(self, roi_cfg: Optional[Dict[str, float]] = None):
        self.roi_cfg = roi_cfg or DEFAULT_ROI

    def extract_roi(
        self,
        image_input,
        remove_glare: bool = True,
        glare_threshold: int = 240
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        """
        Extract liquid reaction zone from image.
        
        Args:
            image_input: PIL Image, filepath, or np.ndarray (H, W, 3)
            remove_glare: Whether to filter out glass specular reflections
            glare_threshold: Pixel intensity threshold above which pixels are marked as glare
            
        Returns:
            roi_pixels: np.ndarray (H_roi, W_roi, 3) representing the cropped ROI
            bbox: (ymin, ymax, xmin, xmax) pixel coordinates
            glare_mask: boolean mask indicating valid non-glare liquid pixels
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.array(img)
        elif isinstance(image_input, Image.Image):
            img = ImageOps.exif_transpose(image_input)
            img = img.convert("RGB")
            arr = np.array(img)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h, w = arr.shape[:2]

        ymin = int(h * self.roi_cfg["y_min_pct"])
        ymax = int(h * self.roi_cfg["y_max_pct"])
        xmin = int(w * self.roi_cfg["x_min_pct"])
        xmax = int(w * self.roi_cfg["x_max_pct"])

        # Boundary checks
        ymin = max(0, min(ymin, h - 1))
        ymax = max(ymin + 5, min(ymax, h))
        xmin = max(0, min(xmin, w - 1))
        xmax = max(xmin + 5, min(xmax, w))

        roi = arr[ymin:ymax, xmin:xmax]

        if remove_glare:
            # Glare / reflections have very high intensity across all channels
            is_glare = (roi[:, :, 0] > glare_threshold) & \
                       (roi[:, :, 1] > glare_threshold) & \
                       (roi[:, :, 2] > glare_threshold)
            valid_mask = ~is_glare
            # If all pixels were flagged as glare (unlikely), fallback to using all pixels
            if not np.any(valid_mask):
                valid_mask = np.ones((roi.shape[0], roi.shape[1]), dtype=bool)
        else:
            valid_mask = np.ones((roi.shape[0], roi.shape[1]), dtype=bool)

        return roi, (ymin, ymax, xmin, xmax), valid_mask

