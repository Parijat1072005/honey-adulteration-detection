"""
Stage C Model: Spectrophotometric A520 & Calculated Proline -> True Adulteration Percentage.
Evaluates how conventional benchtop laboratory spectrophotometry performs.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.config import MODELS_DIR


class StageCSpectroAdulterationModel:
    """
    Predicts adulteration percentage using A520 absorbance and calculated proline content.
    """

    def __init__(self, blank_a520: float = 0.1125):
        self.blank_a520 = blank_a520
        self.feature_cols = ["a520_mean", "proline_ug_ml"]
        self.model = Ridge(alpha=1.0)
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
        preds = self.model.predict(X_sc)
        return np.clip(preds, 0.0, 100.0)

    def analytical_dilution_estimate(self, a520: float, reference_pure_a520: float) -> float:
        """
        Analytical calculation based on dilution equation:
        adulteration = 1 - (A520 - A_blank) / (A_pure - A_blank)
        """
        net_measured = max(0.0, a520 - self.blank_a520)
        net_reference = max(0.01, reference_pure_a520 - self.blank_a520)
        ratio = net_measured / net_reference
        ratio = np.clip(ratio, 0.0, 1.0)
        adulteration_pct = (1.0 - ratio) * 100.0
        return float(adulteration_pct)

    def cross_validate_loocv(self, df_honey: pd.DataFrame) -> Dict[str, float]:
        loo = LeaveOneOut()
        y_true_list, y_pred_list = [], []
        X_feats = df_honey[self.feature_cols].values
        y = df_honey["adulteration_pct"].values

        for train_idx, test_idx in loo.split(X_feats):
            X_tr, X_te = X_feats[train_idx], X_feats[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m_mean = np.mean(X_tr, axis=0)
            m_std = np.std(X_tr, axis=0) + 1e-8
            X_tr_sc = (X_tr - m_mean) / m_std
            X_te_sc = (X_te - m_mean) / m_std

            m = Ridge(alpha=1.0)
            m.fit(X_tr_sc, y_tr)
            p = float(np.clip(m.predict(X_te_sc)[0], 0.0, 100.0))
            y_true_list.append(float(y_te[0]))
            y_pred_list.append(p)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }

    def save(self, filepath: str = str(MODELS_DIR / "stage_c_spectro_to_adulteration.pkl")):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = str(MODELS_DIR / "stage_c_spectro_to_adulteration.pkl")):
        return joblib.load(filepath)

