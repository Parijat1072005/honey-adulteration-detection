"""
Stage B Model: Tube Image Features -> True Adulteration Percentage.
Evaluates whether camera photographs alone can directly estimate honey adulteration percentage.
Includes Leave-One-Honey-Out (LOHO) cross-validation across floral varieties.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.config import MODELS_DIR

STAGE_B_FEATURE_COLS = [
    "od_g",
    "od_b",
    "rg_ratio",
    "a_star",
    "b_star",
    "l_star",
    "rg_diff_index",
    "s_mean",
    "g_med"
]


class StageBImageAdulterationModel:
    """
    Directly predicts adulteration percentage from tube image features.
    """

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        self.model_type = model_type
        self.alpha = alpha
        self.feature_cols = STAGE_B_FEATURE_COLS
        self.scaler_mean = None
        self.scaler_scale = None
        self.model = None

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
        elif model_type == "gb":
            self.model = GradientBoostingRegressor(n_estimators=40, random_state=42, max_depth=2)
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
        # Clip to realistic bounds [0%, 100%]
        return np.clip(preds, 0.0, 100.0)

    def cross_validate_loocv(self, df_honey: pd.DataFrame) -> Dict[str, float]:
        """
        Leave-One-Out CV across honey dilution samples.
        """
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

            if self.model_type == "ridge":
                m = Ridge(alpha=self.alpha)
            elif self.model_type == "rf":
                m = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
            else:
                m = LinearRegression()

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

    def cross_validate_loho(self, df_honey: pd.DataFrame) -> Dict[str, Any]:
        """
        Leave-One-Honey-Out (LOHO) cross-validation:
        Train on one floral honey variety (e.g. H2), test on unseen honey variety (e.g. H4), and vice versa.
        """
        honeys = df_honey["honey_id"].unique()
        fold_results = {}
        all_true = []
        all_pred = []

        for h_test in honeys:
            train_mask = df_honey["honey_id"] != h_test
            test_mask = df_honey["honey_id"] == h_test

            df_tr = df_honey[train_mask]
            df_te = df_honey[test_mask]

            m = StageBImageAdulterationModel(model_type=self.model_type, alpha=self.alpha)
            m.fit(df_tr, df_tr["adulteration_pct"].values)
            preds = m.predict(df_te)

            y_te = df_te["adulteration_pct"].values
            fold_results[f"test_on_{h_test}"] = {
                "true": y_te.tolist(),
                "pred": preds.tolist(),
                "mae": float(mean_absolute_error(y_te, preds))
            }
            all_true.extend(y_te.tolist())
            all_pred.extend(preds.tolist())

        overall_mae = mean_absolute_error(all_true, all_pred)
        overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))

        return {
            "loho_mae": float(overall_mae),
            "loho_rmse": float(overall_rmse),
            "fold_results": fold_results
        }

    def save(self, filepath: str = str(MODELS_DIR / "stage_b_image_to_adulteration.pkl")):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = str(MODELS_DIR / "stage_b_image_to_adulteration.pkl")):
        return joblib.load(filepath)

