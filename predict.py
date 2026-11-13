"""
Inference and Prediction Pipeline.
Evaluates a single reaction tube image (or batch of images), extracts optical and chromogenic features,
estimates A520 absorbance and proline concentration, predicts adulteration percentage,
and computes the Honey Authenticity Index (HAI).
"""
import sys
import argparse
import json
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    ASSAY_DILUTION_FACTOR,
    MIN_AUTHENTIC_PROLINE_MG_KG
)
from src.features.extractor import TubeFeatureExtractor
from src.models.stage_a_spectro import StageASpectroModel
from src.models.stage_b_image_adulteration import StageBImageAdulterationModel
from src.models.stage_c_spectro_adulteration import StageCSpectroAdulterationModel
from src.models.stage_d_multimodal import StageDMultimodalModel
from src.models.stage_e_authenticity_index import HoneyAuthenticityClassifier, HoneyAuthenticityIndexEngine


class HoneyAdulterationPredictor:
    """
    Unified predictor executing Stage A through Stage E.
    """

    def __init__(self):
        self.extractor = TubeFeatureExtractor()
        
        # Load calibration metadata
        meta_path = PROCESSED_DATA_DIR / "dataset_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.calib_slope = self.metadata["standard_curve"]["slope"]
            self.calib_intercept = self.metadata["standard_curve"]["intercept"]
            self.blank_a520 = self.metadata["standard_curve"]["blank_a520"]
        else:
            self.calib_slope = 0.00428
            self.calib_intercept = 0.1751
            self.blank_a520 = 0.1125

        # Load models
        self.model_a = StageASpectroModel.load(str(MODELS_DIR / "stage_a_image_to_a520.pkl"))
        self.model_b = StageBImageAdulterationModel.load(str(MODELS_DIR / "stage_b_image_to_adulteration.pkl"))
        self.model_c = StageCSpectroAdulterationModel.load(str(MODELS_DIR / "stage_c_spectro_to_adulteration.pkl"))
        self.model_d = StageDMultimodalModel.load(str(MODELS_DIR / "stage_d_multimodal.pkl"))
        self.model_e = HoneyAuthenticityClassifier.load(str(MODELS_DIR / "stage_e_authenticity_classifier.pkl"))
        self.hai_engine = HoneyAuthenticityIndexEngine()

    def predict_image(
        self,
        image_input,
        actual_a520: Optional[float] = None,
        honey_variety_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run end-to-end evaluation on a test tube photograph.
        """
        # 1. Extract optical & colorimetric features
        feats = self.extractor.extract_from_image(image_input)
        df_feats = pd.DataFrame([feats])

        # 2. Stage A: Image -> A520 estimation
        pred_a520 = float(self.model_a.predict(df_feats)[0])
        effective_a520 = actual_a520 if actual_a520 is not None else pred_a520

        # 3. Proline content estimation via blank-corrected calibration
        net_a520 = max(0.0, effective_a520 - self.blank_a520)
        proline_ug_ml = net_a520 / self.calib_slope
        proline_mg_kg = proline_ug_ml * ASSAY_DILUTION_FACTOR

        df_feats["a520_mean"] = effective_a520
        df_feats["proline_ug_ml"] = proline_ug_ml
        df_feats["proline_mg_kg"] = proline_mg_kg

        # 4. Stage B: Image-only adulteration prediction
        pred_adulteration_b = float(self.model_b.predict(df_feats)[0])

        # 5. Stage C: Spectrophotometric adulteration prediction
        pred_adulteration_c = float(self.model_c.predict(df_feats)[0])

        # 6. Stage D: Multimodal fusion prediction
        pred_adulteration_d = float(self.model_d.predict(df_feats)[0])

        # Consensus adulteration prediction
        if honey_variety_hint and "H4" in honey_variety_hint:
            ref_pure = 0.5105
            analytical_dilution = max(0.0, min(100.0, (1.0 - max(0.0, effective_a520 - self.blank_a520) / max(0.01, ref_pure - self.blank_a520)) * 100.0))
            consensus_adulteration = 0.70 * analytical_dilution + 0.30 * pred_adulteration_b
        elif honey_variety_hint and "H2" in honey_variety_hint:
            ref_pure = 0.3125
            analytical_dilution = max(0.0, min(100.0, (1.0 - max(0.0, effective_a520 - self.blank_a520) / max(0.01, ref_pure - self.blank_a520)) * 100.0))
            consensus_adulteration = 0.70 * analytical_dilution + 0.30 * pred_adulteration_b
        elif actual_a520 is not None:
            consensus_adulteration = 0.50 * pred_adulteration_d + 0.30 * pred_adulteration_b + 0.20 * pred_adulteration_c
        else:
            consensus_adulteration = pred_adulteration_b

        # 7. Stage E: Binary classification & Honey Authenticity Index
        proba = self.model_e.predict_proba(df_feats)[0]
        adulterated_proba = float(proba[1])

        hai_report = self.hai_engine.evaluate_sample(
            predicted_adulteration_pct=consensus_adulteration,
            proline_mg_kg=proline_mg_kg,
            predicted_a520=pred_a520,
            actual_a520=actual_a520,
            adulteration_proba=adulterated_proba
        )

        return {
            "image_features": {
                "od_g": round(feats["od_g"], 4),
                "od_b": round(feats["od_b"], 4),
                "rg_ratio": round(feats["rg_ratio"], 3),
                "a_star": round(feats["a_star"], 2),
                "b_star": round(feats["b_star"], 2),
                "l_star": round(feats["l_star"], 2),
                "rgb_mean": [round(feats["r_mean"], 1), round(feats["g_mean"], 1), round(feats["b_mean"], 1)]
            },
            "predictions": {
                "stage_a_predicted_a520": round(pred_a520, 4),
                "actual_a520": round(actual_a520, 4) if actual_a520 is not None else None,
                "estimated_proline_ug_ml": round(proline_ug_ml, 2),
                "estimated_proline_mg_kg": round(proline_mg_kg, 2),
                "stage_b_image_adulteration_pct": round(pred_adulteration_b, 1),
                "stage_c_spectro_adulteration_pct": round(pred_adulteration_c, 1),
                "stage_d_multimodal_adulteration_pct": round(pred_adulteration_d, 1),
                "consensus_adulteration_pct": round(consensus_adulteration, 1),
                "adulteration_probability": round(adulterated_proba, 3)
            },
            "authenticity_assessment": hai_report
        }


def format_report(result: Dict[str, Any]) -> str:
    preds = result["predictions"]
    auth = result["authenticity_assessment"]
    feats = result["image_features"]

    verdict_symbol = "✓ PASS" if auth["verdict"] == "PASS" else ("! WARNING" if auth["verdict"] == "WARNING" else "✗ FAIL")

    lines = [
        "==================================================================",
        "          HONEY AUTHENTICITY & ADULTERATION REPORT                ",
        "==================================================================",
        f"  FINAL VERDICT:             [{verdict_symbol}] {auth['status']}",
        f"  HONEY AUTHENTICITY INDEX:  {auth['hai_score']:.1f} / 100.0 (Confidence: {auth['confidence_pct']}%)",
        "------------------------------------------------------------------",
        "  [SPECTROPHOTOMETRIC & BIOCHEMICAL ESTIMATES]",
        f"  - Image-Predicted A520:    {preds['stage_a_predicted_a520']:.4f}" + (f" (Instrument: {preds['actual_a520']:.4f})" if preds['actual_a520'] else " (No benchtop A520 entered)"),
        f"  - Estimated Proline:       {preds['estimated_proline_mg_kg']:.1f} mg/kg (Assay: {preds['estimated_proline_ug_ml']:.2f} µg/mL)",
        f"  - Regulatory Minimum:      {MIN_AUTHENTIC_PROLINE_MG_KG:.1f} mg/kg (Codex Alimentarius / IHC)",
        f"  - Compliance Status:       {'COMPLIANT' if auth['metrics']['regulatory_threshold_met'] else 'NON-COMPLIANT (DEFICIENT PROLINE)'}",
        "------------------------------------------------------------------",
        "  [ADULTERATION PREDICTIONS]",
        f"  - Consensus Adulteration:  {preds['consensus_adulteration_pct']:.1f}% added syrup",
        f"  - Image-Only Model (B):    {preds['stage_b_image_adulteration_pct']:.1f}%",
        f"  - Spectro Model (C):       {preds['stage_c_spectro_adulteration_pct']:.1f}%",
        f"  - Multimodal Model (D):    {preds['stage_d_multimodal_adulteration_pct']:.1f}%",
        "------------------------------------------------------------------",
        "  [COLOR & OPTICAL DENSITY SIGNATURES]",
        f"  - Liquid Core RGB:         {feats['rgb_mean']}",
        f"  - Green Optical Density:   OD_G = {feats['od_g']:.4f} (Peak chromophore absorbance)",
        f"  - Red-Green Ratio (R/G):   {feats['rg_ratio']:.3f}",
        f"  - CIELAB Coordinates:      L* = {feats['l_star']:.1f}, a* = {feats['a_star']:.1f}, b* = {feats['b_star']:.1f}",
        "=================================================================="
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Honey Adulteration from Tube Image.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to reaction tube photograph")
    parser.add_argument("--a520", "-a", type=float, default=None, help="Optional measured A520 spectrophotometric reading")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    predictor = HoneyAdulterationPredictor()
    res = predictor.predict_image(args.image, actual_a520=args.a520)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_report(res))

