"""
Dataset Builder:
1. Dynamically reads laboratory records and spectroscopy data from data/raw/experiment_manifest.csv.
2. Extracts optical, RGB, HSV, CIELAB, and Beer-Lambert OD features from each linked tube image.
3. Automatically fits the standard calibration curve on standards and computes proline equivalents.
4. Produces master structured datasets (master_dataset.csv, proline_standards.csv, honey_dilutions.csv).
"""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ASSAY_DILUTION_FACTOR,
    MIN_AUTHENTIC_PROLINE_MG_KG
)
from src.features.extractor import TubeFeatureExtractor


def build_all_datasets(manifest_path: Path = RAW_DATA_DIR / "experiment_manifest.csv"):
    print("=== Building Honey Adulteration Detection Datasets ===")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found at: {manifest_path}. Please create data/raw/experiment_manifest.csv")

    print(f"Reading experimental records from: {manifest_path}")
    df_manifest = pd.read_csv(manifest_path)

    # Clean up comments or empty lines if any
    df_manifest = df_manifest[~df_manifest["sample_id"].astype(str).str.startswith("#")].dropna(subset=["sample_id"]).copy()

    # Calculate a520_mean
    if "a520_mean" not in df_manifest.columns:
        if "a520_rep1" in df_manifest.columns and "a520_rep2" in df_manifest.columns:
            rep2_vals = df_manifest["a520_rep2"].fillna(df_manifest["a520_rep1"])
            df_manifest["a520_mean"] = ((df_manifest["a520_rep1"] + rep2_vals) / 2.0).round(4)
        elif "a520" in df_manifest.columns:
            df_manifest["a520_mean"] = df_manifest["a520"].astype(float).round(4)
        else:
            raise KeyError("Manifest must contain 'a520_mean', 'a520', or ('a520_rep1' and 'a520_rep2')")

    # Flag adulteration binary target (0 = Authentic, 1 = Adulterated)
    if "is_adulterated" not in df_manifest.columns:
        df_manifest["is_adulterated"] = np.where(
            df_manifest["sample_type"] == "honey",
            np.where(df_manifest["adulteration_pct"] > 0.0, 1, 0),
            np.nan
        )

    # 1. Proline Standard Curve Calibration (Blank-Corrected Analytical Model)
    df_std_raw = df_manifest[df_manifest["sample_type"] == "standard"].copy()
    if len(df_std_raw) > 0 and "proline_ug_ml" in df_std_raw.columns:
        x_std = df_std_raw["proline_ug_ml"].values.astype(float)
        y_std = df_std_raw["a520_mean"].values.astype(float)
        blank_a520 = float(df_std_raw[df_std_raw["proline_ug_ml"] == 0]["a520_mean"].values[0]) if 0 in x_std else 0.1125
        
        # Zero-centered calibration: (A520 - A_blank) = slope * Proline
        delta_y = y_std - blank_a520
        slope = float(np.sum(delta_y * x_std) / np.sum(x_std ** 2))
        intercept = float(blank_a520)
        ss_res = np.sum((delta_y - slope * x_std) ** 2)
        ss_tot = np.sum(delta_y ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))
        print(f"Blank-Corrected Calibration: (A520 - {blank_a520:.4f}) = {slope:.5f} * Proline(ug/mL) (R2 = {r2:.4f})")
    else:
        slope, intercept, r2, blank_a520 = 0.00476, 0.1125, 0.9848, 0.1125
        print(f"Using default calibration curve: (A520 - {blank_a520:.4f}) = {slope:.5f} * Proline")

    # Compute Proline for Honey samples using blank-corrected model
    for idx, row in df_manifest.iterrows():
        if row["sample_type"] == "honey":
            net_a = max(0.0, float(row["a520_mean"]) - blank_a520)
            calc_proline_ug = net_a / slope
            df_manifest.at[idx, "proline_ug_ml"] = round(calc_proline_ug, 2)
            df_manifest.at[idx, "proline_mg_kg"] = round(calc_proline_ug * ASSAY_DILUTION_FACTOR, 2)

    # 2. Extract Tube Image Features
    extractor = TubeFeatureExtractor()
    feature_rows = []

    for _, row in df_manifest.iterrows():
        img_path = str(row["image_path"]).strip()
        print(f"Processing tube: {row['sample_id']} ({img_path})")
        if not Path(img_path).exists():
            print(f"  WARNING: Image file not found: {img_path}. Skipping image features.")
            feats = {}
        else:
            feats = extractor.extract_from_image(img_path)
            feats["roi_bbox"] = str(feats.get("roi_bbox", ""))

        record = dict(row)
        record.update(feats)
        feature_rows.append(record)

    df_master = pd.DataFrame(feature_rows)
    df_master.to_csv(PROCESSED_DATA_DIR / "master_dataset.csv", index=False)

    df_std_full = df_master[df_master["sample_type"] == "standard"].copy()
    df_honey_full = df_master[df_master["sample_type"] == "honey"].copy()
    df_std_full.to_csv(PROCESSED_DATA_DIR / "proline_standards.csv", index=False)
    df_honey_full.to_csv(PROCESSED_DATA_DIR / "honey_dilutions.csv", index=False)

    # 3. Save Calibration Metadata
    metadata = {
        "standard_curve": {
            "equation": f"A520 = {slope:.5f} * Proline(ug/mL) + {intercept:.4f}",
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "blank_a520": float(blank_a520)
        },
        "dilution_factor": ASSAY_DILUTION_FACTOR,
        "authentic_min_proline_mg_kg": MIN_AUTHENTIC_PROLINE_MG_KG,
        "samples_count": len(df_master),
        "standards_count": len(df_std_full),
        "honey_samples_count": len(df_honey_full),
        "manifest_path": str(manifest_path)
    }

    with open(PROCESSED_DATA_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully processed {len(df_master)} samples from manifest.")
    print(f"Master dataset saved: {PROCESSED_DATA_DIR / 'master_dataset.csv'}")


if __name__ == "__main__":
    build_all_datasets()
