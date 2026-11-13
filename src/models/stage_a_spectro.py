"""
Stage A Model: Tube Image Features -> A520 Spectrophotometric Absorbance.
Evaluates whether camera photographs can accurately replace or estimate a benchtop spectrophotometer.
"""
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.config import MODELS_DIR

# Key robust features for A520 optical estimation
STAGE_A_FEATURE_COLS = [
    "od_g",          # Beer-Lambert optical density in green (peaks at 520nm)
    "rg_ratio",      # Red-to-green ratio (reflects orange chromophore)
    "a_star",        # CIELAB red-green axis
    "l_star",        # CIELAB lightness
    "rg_diff_index", # Normalized difference index
    "norm_r"         # Normalized red fraction
]


class StageASpectroModel:
    """
    Predicts A520 spectrophotometric absorbance directly from tube image color features.
    """

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        self.model_type = model_type
        self.alpha = alpha
        self.feature_cols = STAGE_A_FEATURE_COLS
        self.scaler_mean = None
        self.scaler_scale = None
        self.model = None

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=alpha, positive=True)
        elif model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
        elif model_type == "svr":
            self.model = SVR(kernel="rbf", C=10.0, epsilon=0.01)
        elif model_type == "beer_lambert":
            # Simple 1D physical model: A520 = slope * OD_G + intercept
            self.feature_cols = ["od_g"]
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_feats = X[self.feature_cols].values
        self.scaler_mean = np.mean(X_feats, axis=0)
        self.scaler_scale = np.std(X_feats, axis=0) + 1e-8
        X_scaled = (X_feats - self.scaler_mean) / self.scaler_scale

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.feature_cols].values
        X_scaled = (X_feats - self.scaler_mean) / self.scaler_scale
        preds = self.model.predict(X_scaled)
        # Absorbance cannot be negative
        return np.maximum(0.0, preds)

    def cross_validate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Leave-One-Out Cross-Validation (LOOCV) for rigorous assessment on small sample sizes.
        """
        loo = LeaveOneOut()
        y_true_list = []
        y_pred_list = []

        X_feats = X[self.feature_cols].values

        for train_idx, test_idx in loo.split(X_feats):
            X_tr, X_te = X_feats[train_idx], X_feats[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m_mean = np.mean(X_tr, axis=0)
            m_std = np.std(X_tr, axis=0) + 1e-8
            X_tr_sc = (X_tr - m_mean) / m_std
            X_te_sc = (X_te - m_mean) / m_std

            # Clone and fit
            if self.model_type == "linear" or self.model_type == "beer_lambert":
                m = LinearRegression()
            elif self.model_type == "ridge":
                m = Ridge(alpha=self.alpha, positive=True)
            elif self.model_type == "rf":
                m = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
            elif self.model_type == "svr":
                m = SVR(kernel="rbf", C=10.0, epsilon=0.01)

            m.fit(X_tr_sc, y_tr)
            pred = max(0.0, float(m.predict(X_te_sc)[0]))

            y_true_list.append(float(y_te[0]))
            y_pred_list.append(pred)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        pearson_r = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(pearson_r),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }

    def save(self, filepath: str = str(MODELS_DIR / "stage_a_image_to_a520.pkl")):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = str(MODELS_DIR / "stage_a_image_to_a520.pkl")):
        return joblib.load(filepath)

