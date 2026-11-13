"""
Configuration constants and paths for Honey Adulteration Detection System.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
APP_DIR = PROJECT_ROOT / "app"

# Regulatory / Authenticity Standards (Codex Alimentarius & IHC)
# Genuine honey must contain a minimum proline content (typically >= 180 mg/kg)
MIN_AUTHENTIC_PROLINE_MG_KG = 180.0
HIGH_QUALITY_PROLINE_MG_KG = 250.0

# Assay parameters (Ninhydrin assay standard dilution ratio)
# Standard test dilution factor: e.g. 5g in 100mL = 50 mg/mL solution (factor of 20 to convert ug/mL to mg/kg)
ASSAY_DILUTION_FACTOR = 20.0 

# Default Region-of-Interest (ROI) for test tubes (relative coordinates: y_min, y_max, x_min, x_max)
# Liquid column is located in the bottom portion of the tube (away from meniscus, glare, and glass rim)
DEFAULT_ROI = {
    "y_min_pct": 0.55,
    "y_max_pct": 0.85,
    "x_min_pct": 0.28,
    "x_max_pct": 0.72
}

# Thresholds for classification
ADULTERATION_THRESHOLD_PCT = 15.0 # Adulteration >= 15% classified as Adulterated
HAI_AUTHENTIC_THRESHOLD = 80.0     # HAI >= 80: High Authenticity
HAI_SUSPECT_THRESHOLD = 50.0        # 50 <= HAI < 80: Suspect / Borderline

