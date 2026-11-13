"""
Visualization and reporting utilities for honey adulteration analysis.
Generates publication-quality scientific figures.
"""
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def plot_standard_curve(df_std: pd.DataFrame, output_path: Optional[Path] = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    x = df_std["proline_ug_ml"].values
    y = df_std["a520_mean"].values
    yerr = np.abs(df_std["a520_rep1"].values - df_std["a520_rep2"].values) / 2.0

    blank_a = float(df_std[df_std["proline_ug_ml"] == 0]["a520_mean"].values[0])
    delta_y = y - blank_a
    slope = np.sum(delta_y * x) / np.sum(x ** 2)
    intercept = blank_a
    ss_res = np.sum((delta_y - slope * x) ** 2)
    ss_tot = np.sum(delta_y ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    # Scatter points with error bars
    ax.errorbar(
        x, y, yerr=yerr, fmt="o", color="#1f77b4", ecolor="#ff7f0e",
        elinewidth=1.5, capsize=4, markersize=8, label="Standard Replicates (Mean ± Range)"
    )

    # Fit line
    x_line = np.linspace(0, 210, 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line, color="#e377c2", linestyle="-", linewidth=2,
        label=f"Blank-Corrected Model: (A520 - {blank_a:.4f}) = {slope:.5f}·C\n($R^2 = {r2:.4f}$)"
    )

    ax.set_title("Ninhydrin-Proline Assay Calibration Curve (520 nm)")
    ax.set_xlabel("Proline Standard Concentration (µg/mL)")
    ax.set_ylabel("Absorbance at 520 nm ($A_{520}$)")
    ax.set_xlim(-10, 220)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out = output_path or (FIGURES_DIR / "proline_standard_curve.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_beer_lambert_relation(df_master: pd.DataFrame, output_path: Optional[Path] = None):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Optical Density in Green vs A520
    x_od = df_master["od_g"].values
    y_a520 = df_master["a520_mean"].values
    types = df_master["sample_type"].values

    mask_std = types == "standard"
    mask_h2 = (types == "honey") & (df_master["honey_id"] == "H2")
    mask_h4 = (types == "honey") & (df_master["honey_id"] == "H4")

    ax1.scatter(x_od[mask_std], y_a520[mask_std], color="#1f77b4", s=70, marker="o", label="Proline Standards (0-200 µg/mL)")
    ax1.scatter(x_od[mask_h2], y_a520[mask_h2], color="#2ca02c", s=80, marker="^", label="Honey H2 (Light Floral)")
    ax1.scatter(x_od[mask_h4], y_a520[mask_h4], color="#d62728", s=80, marker="s", label="Honey H4 (Dark Amber)")

    slope1, intercept1 = np.polyfit(x_od, y_a520, 1)
    r1 = np.corrcoef(x_od, y_a520)[0, 1]
    x_fit1 = np.linspace(min(x_od), max(x_od), 50)
    ax1.plot(x_fit1, slope1 * x_fit1 + intercept1, "k--", alpha=0.7, label=f"Fit ($r = {r1:.4f}$, $R^2 = {r1**2:.3f}$)")

    ax1.set_title("Physical Coupling: Image Optical Density vs $A_{520}$")
    ax1.set_xlabel("Green Channel Optical Density: $OD_G = -\\log_{10}(G / G_0)$")
    ax1.set_ylabel("Spectrophotometer Reading ($A_{520}$)")
    ax1.legend(loc="upper left")

    # 2. Red-Green Ratio vs A520
    x_rg = df_master["rg_ratio"].values
    ax2.scatter(x_rg[mask_std], y_a520[mask_std], color="#1f77b4", s=70, marker="o", label="Standards")
    ax2.scatter(x_rg[mask_h2], y_a520[mask_h2], color="#2ca02c", s=80, marker="^", label="Honey H2")
    ax2.scatter(x_rg[mask_h4], y_a520[mask_h4], color="#d62728", s=80, marker="s", label="Honey H4")

    slope2, intercept2 = np.polyfit(x_rg, y_a520, 1)
    r2 = np.corrcoef(x_rg, y_a520)[0, 1]
    x_fit2 = np.linspace(min(x_rg), max(x_rg), 50)
    ax2.plot(x_fit2, slope2 * x_fit2 + intercept2, "k--", alpha=0.7, label=f"Fit ($r = {r2:.4f}$, $R^2 = {r2**2:.3f}$)")

    ax2.set_title("Chromogenic Indicator: Red/Green Channel Ratio vs $A_{520}$")
    ax2.set_xlabel("Image Red / Green Ratio ($R / G$)")
    ax2.set_ylabel("Spectrophotometer Reading ($A_{520}$)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out = output_path or (FIGURES_DIR / "beer_lambert_od_vs_a520.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_adulteration_color_trajectories(df_honey: pd.DataFrame, output_path: Optional[Path] = None):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    h2 = df_honey[df_honey["honey_id"] == "H2"].sort_values("adulteration_pct")
    h4 = df_honey[df_honey["honey_id"] == "H4"].sort_values("adulteration_pct")

    # 1. A520 vs Adulteration %
    ax1.plot(h2["adulteration_pct"], h2["a520_mean"], "o-", color="#2ca02c", linewidth=2, markersize=8, label="Honey H2 (Light Floral)")
    ax1.plot(h4["adulteration_pct"], h4["a520_mean"], "s-", color="#d62728", linewidth=2, markersize=8, label="Honey H4 (Dark Amber)")
    ax1.set_title("Assay Response Across Adulteration Levels")
    ax1.set_xlabel("Adulteration Percentage (%)")
    ax1.set_ylabel("Absorbance ($A_{520}$)")
    ax1.axhline(0.18, color="gray", linestyle=":", label="Authenticity Threshold Region")
    ax1.legend()

    # 2. CIELAB a* (Redness) vs b* (Yellowness) Color Trajectory
    ax2.plot(h2["a_star"], h2["b_star"], "o-", color="#2ca02c", linewidth=2, markersize=8, label="H2 Dilution Path")
    ax2.plot(h4["a_star"], h4["b_star"], "s-", color="#d62728", linewidth=2, markersize=8, label="H4 Dilution Path")

    for _, row in h2.iterrows():
        ax2.annotate(f"{int(row['adulteration_pct'])}%", (row["a_star"]+0.3, row["b_star"]-0.5), fontsize=9, color="#1b611b")
    for _, row in h4.iterrows():
        ax2.annotate(f"{int(row['adulteration_pct'])}%", (row["a_star"]+0.3, row["b_star"]+0.3), fontsize=9, color="#8b1819")

    ax2.set_title("CIELAB Color Space Trajectory with Adulteration")
    ax2.set_xlabel("Redness ($a^*$)")
    ax2.set_ylabel("Yellowness ($b^*$)")
    ax2.legend()

    plt.tight_layout()
    out = output_path or (FIGURES_DIR / "color_trajectory_adulteration.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_correlation_matrix(df_master: pd.DataFrame, output_path: Optional[Path] = None):
    set_plot_style()
    features = [
        "a520_mean", "proline_ug_ml", "od_g", "od_b", "rg_ratio",
        "rg_diff_index", "a_star", "b_star", "l_star", "norm_r", "g_mean", "b_mean"
    ]
    labels = [
        "$A_{520}$", "Proline", "$OD_G$", "$OD_B$", "$R/G$",
        "$(R-G)/(R+G)$", "$a^*$ (Red)", "$b^*$ (Yellow)", "$L^*$ (Light)", "$r_{norm}$", "Green", "Blue"
    ]

    corr = df_master[features].corr().values

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Print correlation values on cells
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Feature Correlation Matrix (Spectrophotometry & Multi-Color Space)")
    plt.tight_layout()
    out = output_path or (FIGURES_DIR / "correlation_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def generate_all_figures():
    print("=== Generating Scientific Visualizations ===")
    df_master = pd.read_csv(PROCESSED_DATA_DIR / "master_dataset.csv")
    df_std = pd.read_csv(PROCESSED_DATA_DIR / "proline_standards.csv")
    df_honey = pd.read_csv(PROCESSED_DATA_DIR / "honey_dilutions.csv")

    plot_standard_curve(df_std)
    plot_beer_lambert_relation(df_master)
    plot_adulteration_color_trajectories(df_honey)
    plot_correlation_matrix(df_master)
    print("All figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()

