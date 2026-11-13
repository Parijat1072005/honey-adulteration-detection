"""
Feature extraction pipeline for reaction tube photographs.
Extracts RGB, Beer-Lambert Optical Densities, HSV, CIELAB, and Chromogenic Indices.
"""
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from src.features.roi_detector import TubeROIDetector


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image [0, 255] to HSV image [0-360, 0-1, 0-1].
    """
    rgb_norm = rgb.astype(float) / 255.0
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc

    s = np.where(maxc != 0, deltac / (maxc + 1e-10), 0.0)

    # Hue calculation
    h = np.zeros_like(maxc)
    mask_nz = deltac > 1e-6
    mask_r = mask_nz & (maxc == r)
    mask_g = mask_nz & (maxc == g) & (~mask_r)
    mask_b = mask_nz & (maxc == b) & (~mask_r) & (~mask_g)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / deltac[mask_r]) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 4
    h = h * 60.0
    h[h < 0] += 360.0

    return np.stack([h, s, v], axis=-1)


def rgb_to_cielab_np(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB [0, 255] to CIELAB using D65 reference illuminant.
    """
    rgb_norm = rgb.astype(float) / 255.0
    # sRGB inverse companding (gamma linearization)
    mask = rgb_norm > 0.04045
    linear_rgb = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # Matrix multiplication to XYZ (D65)
    r = linear_rgb[..., 0]
    g = linear_rgb[..., 1]
    b = linear_rgb[..., 2]

    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    # Nonlinear transformation
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3
    factor = (1.0 / 3.0) * ((29.0 / 6.0) ** 2)

    fx = np.where(x > delta_cubed, np.cbrt(x), factor * x + 4.0 / 29.0)
    fy = np.where(y > delta_cubed, np.cbrt(y), factor * y + 4.0 / 29.0)
    fz = np.where(z > delta_cubed, np.cbrt(z), factor * z + 4.0 / 29.0)

    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return np.stack([l_star, a_star, b_star], axis=-1)


class TubeFeatureExtractor:
    """
    Extracts multi-color-space features from reaction tube photographs.
    """

    def __init__(self, roi_detector: Optional[TubeROIDetector] = None, blank_rgb: Optional[np.ndarray] = None):
        self.roi_detector = roi_detector or TubeROIDetector()
        # Default blank intensity for 0 ug/mL proline reference tube
        self.blank_rgb = blank_rgb if blank_rgb is not None else np.array([208.2, 208.2, 207.0])

    def extract_from_roi(self, roi: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract features from a pre-cropped ROI array.
        """
        if valid_mask is None:
            valid_mask = np.ones((roi.shape[0], roi.shape[1]), dtype=bool)

        pixels_rgb = roi[valid_mask].astype(float)
        if len(pixels_rgb) == 0:
            pixels_rgb = roi.reshape(-1, 3).astype(float)

        # RGB statistics
        r_mean, g_mean, b_mean = np.mean(pixels_rgb, axis=0)
        r_std, g_std, b_std = np.std(pixels_rgb, axis=0)
        r_med, g_med, b_med = np.median(pixels_rgb, axis=0)

        # Normalized RGB
        sum_rgb = r_mean + g_mean + b_mean + 1e-6
        norm_r = r_mean / sum_rgb
        norm_g = g_mean / sum_rgb
        norm_b = b_mean / sum_rgb

        # Ratios & Chromogenic indices
        rg_ratio = r_mean / (g_mean + 1e-6)
        rb_ratio = r_mean / (b_mean + 1e-6)
        gb_ratio = g_mean / (b_mean + 1e-6)
        rg_diff_index = (r_mean - g_mean) / (r_mean + g_mean + 1e-6)
        rb_diff_index = (r_mean - b_mean) / (r_mean + b_mean + 1e-6)

        # Beer-Lambert Optical Densities: OD = -log10(I / I0)
        # 520nm absorption primarily attenuates the Green channel!
        r0, g0, b0 = self.blank_rgb
        od_r = -np.log10(max(r_mean, 1e-3) / r0)
        od_g = -np.log10(max(g_mean, 1e-3) / g0)
        od_b = -np.log10(max(b_mean, 1e-3) / b0)

        # HSV
        roi_hsv = rgb_to_hsv_np(roi)
        pixels_hsv = roi_hsv[valid_mask]
        h_mean, s_mean, v_mean = np.mean(pixels_hsv, axis=0)
        h_std, s_std, v_std = np.std(pixels_hsv, axis=0)

        # CIELAB
        roi_lab = rgb_to_cielab_np(roi)
        pixels_lab = roi_lab[valid_mask]
        l_mean, a_mean, b_lab_mean = np.mean(pixels_lab, axis=0)
        l_std, a_std, b_lab_std = np.std(pixels_lab, axis=0)

        chroma = np.sqrt(a_mean ** 2 + b_lab_mean ** 2)
        hue_angle_deg = np.degrees(np.arctan2(b_lab_mean, a_mean)) % 360.0

        return {
            "r_mean": float(r_mean),
            "g_mean": float(g_mean),
            "b_mean": float(b_mean),
            "r_std": float(r_std),
            "g_std": float(g_std),
            "b_std": float(b_std),
            "r_med": float(r_med),
            "g_med": float(g_med),
            "b_med": float(b_med),
            "norm_r": float(norm_r),
            "norm_g": float(norm_g),
            "norm_b": float(norm_b),
            "rg_ratio": float(rg_ratio),
            "rb_ratio": float(rb_ratio),
            "gb_ratio": float(gb_ratio),
            "rg_diff_index": float(rg_diff_index),
            "rb_diff_index": float(rb_diff_index),
            "od_r": float(od_r),
            "od_g": float(od_g),
            "od_b": float(od_b),
            "h_mean": float(h_mean),
            "s_mean": float(s_mean),
            "v_mean": float(v_mean),
            "l_star": float(l_mean),
            "a_star": float(a_mean),
            "b_star": float(b_lab_mean),
            "chroma": float(chroma),
            "hue_angle": float(hue_angle_deg),
        }

    def extract_from_image(self, image_input) -> Dict[str, Any]:
        """
        Extract ROI and compute all multi-color-space features.
        """
        roi, bbox, valid_mask = self.roi_detector.extract_roi(image_input)
        features = self.extract_from_roi(roi, valid_mask)
        features["roi_bbox"] = bbox
        features["valid_pixels_pct"] = float(np.mean(valid_mask) * 100.0)
        return features

