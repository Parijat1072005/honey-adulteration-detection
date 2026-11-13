"""
Stage E: Honey Authenticity Index (HAI) Engine & Authenticity Classifier.
Integrates predicted adulteration percentage, proline content compliance (Codex Alimentarius / IHC),
image-spectrophotometry agreement, and uncertainty into an actionable decision score.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

from src.config import (
    MODELS_DIR,
    MIN_AUTHENTIC_PROLINE_MG_KG,
    HIGH_QUALITY_PROLINE_MG_KG,
    HAI_AUTHENTIC_THRESHOLD,
    HAI_SUSPECT_THRESHOLD,
    ADULTERATION_THRESHOLD_PCT
)


class HoneyAuthenticityClassifier:
    """
    Binary classifier predicting whether a honey sample is Authentic (0) or Adulterated (1).
    """

    def __init__(self, penalty: str = "l2", C: float = 1.0):
        self.feature_cols = ["od_g", "rg_ratio", "a_star", "l_star"]
        self.model = LogisticRegression(C=C, random_state=42)
        self.scaler_mean = None
        self.scaler_scale = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_feats = X[self.feature_cols].values
        self.scaler_mean = np.mean(X_feats, axis=0)
        self.scaler_scale = np.std(X_feats, axis=0) + 1e-8
        X_sc = (X_feats - self.scaler_mean) / self.scaler_scale
        self.model.fit(X_sc, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.feature_cols].values
        X_sc = (X_feats - self.scaler_mean) / self.scaler_scale
        return self.model.predict(X_sc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.feature_cols].values
        X_sc = (X_feats - self.scaler_mean) / self.scaler_scale
        return self.model.predict_proba(X_sc)

    def save(self, filepath: str = str(MODELS_DIR / "stage_e_authenticity_classifier.pkl")):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = str(MODELS_DIR / "stage_e_authenticity_classifier.pkl")):
        return joblib.load(filepath)


class HoneyAuthenticityIndexEngine:
    """
    Calculates the multimodal Honey Authenticity Index (HAI, 0 to 100),
    authenticity probability, confidence intervals, and regulatory compliance flags.
    """

    def __init__(
        self,
        min_proline_mg_kg: float = MIN_AUTHENTIC_PROLINE_MG_KG,
        high_proline_mg_kg: float = HIGH_QUALITY_PROLINE_MG_KG
    ):
        self.min_proline = min_proline_mg_kg
        self.high_proline = high_proline_mg_kg

    def compute_proline_subscore(self, proline_mg_kg: float) -> float:
        """
        Subscore based on Codex Alimentarius / IHC minimum standards:
        >= 250 mg/kg: 100
        180 - 250 mg/kg: 75 to 100
        100 - 180 mg/kg: 40 to 75 (suspect/substandard)
        < 100 mg/kg: 0 to 40 (severe deficiency/dilution)
        """
        if proline_mg_kg >= self.high_proline:
            return 100.0
        elif proline_mg_kg >= self.min_proline:
            # 180 to 250
            return 75.0 + 25.0 * ((proline_mg_kg - self.min_proline) / (self.high_proline - self.min_proline))
        elif proline_mg_kg >= 100.0:
            # 100 to 180
            return 40.0 + 35.0 * ((proline_mg_kg - 100.0) / (self.min_proline - 100.0))
        else:
            # < 100
            return max(0.0, 40.0 * (proline_mg_kg / 100.0))

    def compute_agreement_subscore(self, pred_a520: float, actual_a520: Optional[float]) -> float:
        """
        Measures agreement between photograph color estimation and benchtop instrument.
        """
        if actual_a520 is None or np.isnan(actual_a520):
            return 100.0  # Default neutral when benchtop spectro is not supplied

        abs_diff = abs(pred_a520 - actual_a520)
        # Difference of 0.05 absorbance units is excellent (90%), 0.20 is poor (60%)
        subscore = max(0.0, 100.0 - 200.0 * abs_diff)
        return float(subscore)

    def evaluate_sample(
        self,
        predicted_adulteration_pct: float,
        proline_mg_kg: float,
        predicted_a520: float,
        actual_a520: Optional[float] = None,
        adulteration_proba: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes composite Honey Authenticity Index and comprehensive diagnostic assessment.
        """
        # 1. Adulteration Subscore (0 to 100)
        adulteration_subscore = max(0.0, 100.0 - predicted_adulteration_pct)

        # 2. Proline Subscore (0 to 100)
        proline_subscore = self.compute_proline_subscore(proline_mg_kg)

        # 3. Multimodal Agreement Subscore (0 to 100)
        agreement_subscore = self.compute_agreement_subscore(predicted_a520, actual_a520)

        # 4. Composite HAI
        if actual_a520 is not None and not np.isnan(actual_a520):
            # Multimodal fusion weighting
            hai = (0.45 * adulteration_subscore + 
                   0.40 * proline_subscore + 
                   0.15 * agreement_subscore)
        else:
            # Image-only weighting
            hai = (0.55 * adulteration_subscore + 
                   0.45 * proline_subscore)

        hai = float(np.clip(hai, 0.0, 100.0))

        # 5. Diagnostic Category
        if hai >= HAI_AUTHENTIC_THRESHOLD and proline_mg_kg >= self.min_proline and predicted_adulteration_pct < ADULTERATION_THRESHOLD_PCT:
            status = "Authentic / Pure Honey"
            badge_color = "green"
            verdict = "PASS"
        elif hai >= HAI_SUSPECT_THRESHOLD:
            status = "Borderline / Suspect Adulteration"
            badge_color = "orange"
            verdict = "WARNING"
        else:
            status = "Adulterated / Substandard"
            badge_color = "red"
            verdict = "FAIL"

        # Confidence Estimation
        confidence = 100.0 - min(abs(predicted_adulteration_pct - ADULTERATION_THRESHOLD_PCT) * 1.5, 40.0)
        if actual_a520 is not None and not np.isnan(actual_a520):
            # Boost confidence when multimodal agreement is high
            confidence = (confidence * 0.7) + (agreement_subscore * 0.3)
        confidence = float(np.clip(confidence, 45.0, 98.0))

        return {
            "hai_score": round(hai, 1),
            "verdict": verdict,
            "status": status,
            "badge_color": badge_color,
            "confidence_pct": round(confidence, 1),
            "subscores": {
                "adulteration_subscore": round(adulteration_subscore, 1),
                "proline_subscore": round(proline_subscore, 1),
                "agreement_subscore": round(agreement_subscore, 1) if actual_a520 is not None else None
            },
            "metrics": {
                "predicted_adulteration_pct": round(predicted_adulteration_pct, 1),
                "proline_mg_kg": round(proline_mg_kg, 1),
                "predicted_a520": round(predicted_a520, 4),
                "actual_a520": round(actual_a520, 4) if actual_a520 is not None else None,
                "regulatory_threshold_met": bool(proline_mg_kg >= self.min_proline)
            }
        }

