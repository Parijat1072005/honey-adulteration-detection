"""
Model Training and Cross-Validation Pipeline.
Trains and benchmarks Stage A, Stage B, Stage C, Stage D, and Stage E models.
Saves serialized model artifacts in models/ and benchmark reports in models/model_benchmark_results.json.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from src.models.stage_a_spectro import StageASpectroModel
from src.models.stage_b_image_adulteration import StageBImageAdulterationModel
from src.models.stage_c_spectro_adulteration import StageCSpectroAdulterationModel
from src.models.stage_d_multimodal import StageDMultimodalModel
from src.models.stage_e_authenticity_index import HoneyAuthenticityClassifier, HoneyAuthenticityIndexEngine
from src.utils.visualization import generate_all_figures


def run_training_pipeline():
    print("==================================================================")
    print("   HONEY ADULTERATION DETECTION: MULTI-STAGE MODEL TRAINING       ")
    print("==================================================================")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load Datasets
    df_master = pd.read_csv(PROCESSED_DATA_DIR / "master_dataset.csv")
    df_std = pd.read_csv(PROCESSED_DATA_DIR / "proline_standards.csv")
    df_honey = pd.read_csv(PROCESSED_DATA_DIR / "honey_dilutions.csv")

    results = {}

    # ------------------------------------------------------------------
    # STAGE A: Tube Image Features -> A520 Absorbance
    # ------------------------------------------------------------------
    print("\n--- [STAGE A]: Image Color Features -> A520 Spectrophotometric Absorbance ---")
    stage_a_candidates = {
        "Beer-Lambert (OD_G only)": StageASpectroModel(model_type="beer_lambert"),
        "Multi-Color Ridge Regression": StageASpectroModel(model_type="ridge", alpha=1.0),
        "Multi-Color Linear Regression": StageASpectroModel(model_type="linear"),
        "Random Forest Regressor": StageASpectroModel(model_type="rf"),
        "Support Vector Regressor (SVR)": StageASpectroModel(model_type="svr")
    }

    stage_a_results = {}
    best_stage_a_r2 = -1.0
    best_stage_a_name = None
    best_stage_a_model = None

    for name, model in stage_a_candidates.items():
        metrics = model.cross_validate(df_master, df_master["a520_mean"].values)
        stage_a_results[name] = {
            "loocv_r2": round(metrics["r2"], 4),
            "loocv_rmse": round(metrics["rmse"], 4),
            "loocv_mae": round(metrics["mae"], 4),
            "pearson_r": round(metrics["pearson_r"], 4)
        }
        print(f"  > {name:32s} | LOOCV R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | Pearson r: {metrics['pearson_r']:.4f}")
        if metrics["r2"] > best_stage_a_r2:
            best_stage_a_r2 = metrics["r2"]
            best_stage_a_name = name
            best_stage_a_model = model

    # Fit best model on all data and save
    best_stage_a_model.fit(df_master, df_master["a520_mean"].values)
    best_stage_a_model.save(str(MODELS_DIR / "stage_a_image_to_a520.pkl"))
    print(f"  * Best Stage A model ({best_stage_a_name}) saved to models/stage_a_image_to_a520.pkl")
    results["stage_a"] = {"best_model": best_stage_a_name, "candidates": stage_a_results}

    # ------------------------------------------------------------------
    # STAGE B: Tube Image Features -> True Adulteration %
    # ------------------------------------------------------------------
    print("\n--- [STAGE B]: Image Color Features -> True Adulteration Percentage ---")
    stage_b_candidates = {
        "Ridge Regression": StageBImageAdulterationModel(model_type="ridge", alpha=1.0),
        "Linear Regression": StageBImageAdulterationModel(model_type="linear"),
        "Random Forest": StageBImageAdulterationModel(model_type="rf")
    }

    stage_b_results = {}
    best_stage_b_mae = 999.0
    best_stage_b_model = None
    best_stage_b_name = None

    for name, model in stage_b_candidates.items():
        loocv = model.cross_validate_loocv(df_honey)
        loho = model.cross_validate_loho(df_honey)
        stage_b_results[name] = {
            "loocv_r2": round(loocv["r2"], 4),
            "loocv_rmse": round(loocv["rmse"], 4),
            "loocv_mae": round(loocv["mae"], 4),
            "loho_mae": round(loho["loho_mae"], 4),
            "loho_rmse": round(loho["loho_rmse"], 4),
            "loho_folds": loho["fold_results"]
        }
        print(f"  > {name:20s} | LOOCV MAE: {loocv['mae']:.2f}% | R2: {loocv['r2']:.4f} | LOHO Cross-Honey MAE: {loho['loho_mae']:.2f}%")
        if loocv["mae"] < best_stage_b_mae:
            best_stage_b_mae = loocv["mae"]
            best_stage_b_model = model
            best_stage_b_name = name

    best_stage_b_model.fit(df_honey, df_honey["adulteration_pct"].values)
    best_stage_b_model.save(str(MODELS_DIR / "stage_b_image_to_adulteration.pkl"))
    print(f"  * Best Stage B model ({best_stage_b_name}) saved to models/stage_b_image_to_adulteration.pkl")
    results["stage_b"] = {"best_model": best_stage_b_name, "candidates": stage_b_results}

    # ------------------------------------------------------------------
    # STAGE C: Spectrophotometric A520 & Proline -> True Adulteration %
    # ------------------------------------------------------------------
    print("\n--- [STAGE C]: Spectrophotometric Data (A520 + Proline) -> Adulteration % ---")
    model_c = StageCSpectroAdulterationModel()
    metrics_c = model_c.cross_validate_loocv(df_honey)
    model_c.fit(df_honey, df_honey["adulteration_pct"].values)
    model_c.save(str(MODELS_DIR / "stage_c_spectro_to_adulteration.pkl"))
    print(f"  > Stage C Benchtop Model | LOOCV MAE: {metrics_c['mae']:.2f}% | RMSE: {metrics_c['rmse']:.2f}% | R2: {metrics_c['r2']:.4f}")
    results["stage_c"] = {
        "loocv_r2": round(metrics_c["r2"], 4),
        "loocv_rmse": round(metrics_c["rmse"], 4),
        "loocv_mae": round(metrics_c["mae"], 4)
    }

    # ------------------------------------------------------------------
    # STAGE D: Multimodal Fusion (Image Features + A520) -> Adulteration %
    # ------------------------------------------------------------------
    print("\n--- [STAGE D]: Multimodal Fusion (Image + A520) -> Adulteration % ---")
    model_d = StageDMultimodalModel(alpha=1.0)
    metrics_d = model_d.cross_validate_loocv(df_honey)
    model_d.fit(df_honey, df_honey["adulteration_pct"].values)
    model_d.save(str(MODELS_DIR / "stage_d_multimodal.pkl"))
    print(f"  > Stage D Multimodal Model | LOOCV MAE: {metrics_d['mae']:.2f}% | RMSE: {metrics_d['rmse']:.2f}% | R2: {metrics_d['r2']:.4f}")
    results["stage_d"] = {
        "loocv_r2": round(metrics_d["r2"], 4),
        "loocv_rmse": round(metrics_d["rmse"], 4),
        "loocv_mae": round(metrics_d["mae"], 4)
    }

    # ------------------------------------------------------------------
    # STAGE E: Binary Authenticity Classifier & HAI Engine
    # ------------------------------------------------------------------
    print("\n--- [STAGE E]: Binary Authenticity Classifier & Honey Authenticity Index ---")
    clf_e = HoneyAuthenticityClassifier(C=1.0)
    clf_e.fit(df_honey, df_honey["is_adulterated"].values)
    train_acc = float(np.mean(clf_e.predict(df_honey) == df_honey["is_adulterated"].values))
    clf_e.save(str(MODELS_DIR / "stage_e_authenticity_classifier.pkl"))
    print(f"  > Stage E Binary Classifier | Training Accuracy: {train_acc*100.0:.1f}%")
    results["stage_e"] = {"binary_classifier_accuracy": train_acc}

    # Save benchmark results
    benchmark_path = MODELS_DIR / "model_benchmark_results.json"
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {benchmark_path}")

    # Plot benchmark comparison bar chart
    plot_stage_benchmarks(results)

    # Generate all publication figures
    generate_all_figures()

    print("\n==================================================================")
    print("   MODEL TRAINING AND VALIDATION COMPLETED SUCCESSFULLY!          ")
    print("==================================================================")


def plot_stage_benchmarks(results: dict):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    stages = ["Stage B\n(Image Only)", "Stage C\n(A520 Spectro Only)", "Stage D\n(Multimodal Fusion)"]
    maes = [
        results["stage_b"]["candidates"][results["stage_b"]["best_model"]]["loocv_mae"],
        results["stage_c"]["loocv_mae"],
        results["stage_d"]["loocv_mae"]
    ]
    rmses = [
        results["stage_b"]["candidates"][results["stage_b"]["best_model"]]["loocv_rmse"],
        results["stage_c"]["loocv_rmse"],
        results["stage_d"]["loocv_rmse"]
    ]

    x = np.arange(len(stages))
    width = 0.35

    rects1 = ax.bar(x - width/2, maes, width, label="LOOCV MAE (%)", color="#4285F4")
    rects2 = ax.bar(x + width/2, rmses, width, label="LOOCV RMSE (%)", color="#EA4335")

    ax.set_ylabel("Adulteration Prediction Error (%)")
    ax.set_title("Adulteration Prediction Error Comparison Across Modeling Stages")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "model_benchmarks_stages.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved benchmark figure: {out}")


if __name__ == "__main__":
    run_training_pipeline()

