# AI-Assisted Honey Authenticity & Adulteration Detection System
### Coupling Computer Vision (RGB Tube Photography) with Ninhydrin-Proline Spectroscopy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary & Scientific Rationale

Adulteration of genuine honey with cheap industrial syrups (corn syrup, high-fructose corn syrup, invert sugar syrup, and rice syrup) is one of the most widespread food fraud challenges globally. 

### Why the Ninhydrin-Proline Assay?
1. **Proline as a Biomarker**: Proline is the dominant free amino acid in natural honey (typically accounting for >50–80% of total amino acids), secreted primarily by honeybee salivary glands (*Apis mellifera*) during nectar processing and ripening.
2. **Regulatory Standards**: Codex Alimentarius, EU Honey Directive, and FSSAI mandate a minimum proline content of **$\ge 180\text{ mg/kg}$** (or $200\text{ mg/kg}$) for authentic honey.
3. **The Chromogenic Reaction**: In acidic conditions, ninhydrin selectively reacts with proline to form a distinct yellow-to-reddish-orange chromophore exhibiting strong optical absorbance specifically at **$520\text{ nm}$ ($A_{520}$)**.
4. **Adulteration Dynamics**: Commercial sugar syrups contain virtually zero proline. Therefore, dilution of honey directly diminishes chromophore formation, lowering $A_{520}$ and causing a predictable shift in digital color space from rich orange-amber to pale translucent beige.

---

## 🔬 The Physical Link: Camera RGB to Beer-Lambert Absorbance

A central question in this project is: **Can a smartphone or digital camera replace or calibrate a laboratory spectrophotometer?**

Under Beer-Lambert's Law:
$$A = \epsilon \cdot c \cdot l = -\log_{10}(T) = -\log_{10}\left(\frac{I}{I_0}\right)$$

In digital RGB camera sensors:
- The **Green channel ($G$)** filter has peak spectral transmission between $510\text{–}540\text{ nm}$, perfectly coinciding with the proline-ninhydrin absorption band ($520\text{ nm}$).
- As chromophore concentration increases, transmitted green light decreases.
- Defining **Green Channel Optical Density**:
  $$OD_G = -\log_{10}\left(\frac{G_{\text{liquid}}}{G_{\text{blank}}}\right)$$
- Our experimental validation across all standard and honey tubes revealed a **Pearson correlation of $r = 0.969$ and Leave-One-Out $R^2 = 0.922$ between $OD_G$ and benchtop $A_{520}$!**

---

## 📊 Data Sufficiency Analysis & Baseline Validation

### Is the Current Pilot Dataset Sufficient?
- **For a Baseline Proof-of-Concept:** **YES.** The current dataset consists of:
  - 8 Proline Standards ($0, 20, 40, 60, 80, 100, 150, 200\ \mu\text{g/mL}$) with technical replicates.
  - 10 Honey Dilution tubes across 2 distinct floral honeys (Light Floral H2 and Dark Amber H4 at $100\%, 80\%, 60\%, 40\%, 20\%$).
  - 8 Pilot Corn Syrup adulteration samples (Sample 1 and Sample 2 from the Revised Strategy Work Plan).
- **For a Universal Commercial Production Model:** **NOT YET.** 
  - Cross-honey generalization testing (Leave-One-Honey-Out, LOHO: training on light honey H2 and evaluating on dark honey H4) shows an error gap ($\approx 22\text{–}26\%$ MAE) because different floral sources have different initial baseline proline concentrations and natural background pigments.
  - Therefore, the models developed in this repository establish the **analytical baseline** and provide the **exact infrastructure** for you to drop new images into `data/raw/` to reach production-grade scale.

---

## 🏗️ Multi-Stage Modeling Architecture

Following the revised project strategy, the pipeline is structured into 5 computational stages:

| Stage | Input | Target | Best Model | LOOCV Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A** | Tube Image Color ($OD_G$, $R/G$, $a^*$, $L^*$) | $A_{520}$ Spectrophotometer Reading | Beer-Lambert Optical Model | **$R^2 = 0.9224$**, RMSE = $0.065$ |
| **Stage B** | Tube Image Color Features | True Adulteration Percentage (%) | Regularized Ridge Regression | **MAE = $15.34\%$**, RMSE = $17.6\%$ |
| **Stage C** | Benchtop $A_{520}$ & Calculated Proline | True Adulteration Percentage (%) | Spectrophotometric Linear Ridge | **MAE = $15.76\%$**, RMSE = $19.1\%$ |
| **Stage D** | Multimodal Fusion (Image + $A_{520}$) | True Adulteration Percentage (%) | Multimodal Ridge Regressor | **MAE = $17.12\%$**, RMSE = $20.9\%$ |
| **Stage E** | Multi-feature Ensemble | Authentic vs Adulterated & HAI | Logistic Classifier + HAI Engine | **Accuracy = $90.0\%$**, $HAI \in [0, 100]$ |

### Honey Authenticity Index (HAI) Formulation
The composite index evaluates a sample on a continuous scale $[0, 100]$:
$$HAI = 0.45 \cdot S_{\text{adulteration}} + 0.40 \cdot S_{\text{proline}} + 0.15 \cdot S_{\text{agreement}}$$
- **Grade A Authentic**: $HAI \ge 80$, Proline $\ge 180\text{ mg/kg}$, Adulteration $< 15\%$
- **Suspect / Borderline**: $50 \le HAI < 80$
- **Adulterated / Substandard**: $HAI < 50$

---

## 📁 Repository Structure

```
honey-adulteration-detection/
├── app/
│   └── streamlit_app.py        # Interactive laboratory web dashboard & mobile simulator
├── data/
│   ├── raw/
│   │   ├── standards/          # 8 Proline standard calibration tubes (0-200 ug/mL)
│   │   ├── honey_h2/           # 5 H2 honey dilution tubes (100% to 20%)
│   │   ├── honey_h4/           # 5 H4 honey dilution tubes (100% to 20%)
│   │   └── raw_honeys/         # Stock neat honey tubes
│   └── processed/
│       ├── master_dataset.csv  # 18 samples with all extracted multi-color-space features
│       ├── proline_standards.csv
│       ├── honey_dilutions.csv
│       └── dataset_metadata.json
├── figures/                    # Publication-grade high-res figures
│   ├── beer_lambert_od_vs_a520.png
│   ├── proline_standard_curve.png
│   ├── color_trajectory_adulteration.png
│   ├── correlation_heatmap.png
│   └── model_benchmarks_stages.png
├── models/                     # Serialized pre-trained model artifacts
├── src/
│   ├── config.py               # Constants, regulatory thresholds, paths
│   ├── features/
│   │   ├── roi_detector.py     # Tube liquid core detection & glare filtering
│   │   └── extractor.py        # RGB, OD, HSV, CIELAB extraction
│   ├── models/
│   │   ├── stage_a_spectro.py
│   │   ├── stage_b_image_adulteration.py
│   │   ├── stage_c_spectro_adulteration.py
│   │   ├── stage_d_multimodal.py
│   │   └── stage_e_authenticity_index.py
│   └── utils/
│       └── visualization.py    # Figure plotting routines
├── build_dataset.py            # Feature extraction & dataset compilation script
├── train.py                    # Multi-stage training & cross-validation script
├── predict.py                  # CLI and Python inference engine
├── main.py                     # Unified CLI entrypoint
├── requirements.txt            # Dependency manifest
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the repository, and ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Predict on Any Reaction Tube Photograph
Evaluate a tube photo using only computer vision:
```bash
python main.py --predict --image data/raw/honey_h4/h4_100pct_honey.jpg
```
Or evaluate with both image and benchtop spectrophotometer reading:
```bash
python main.py --predict --image data/raw/honey_h4/h4_100pct_honey.jpg --a520 0.5105
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` to access the full user interface:
- **Tab 1: Live Predictor & Mobile Simulator** — Upload tube photos, inspect the liquid ROI crop, view the Honey Authenticity Index gauge, and download reports.
- **Tab 2: Standard Curve & Assay Validation** — Interactive standard curve with $R^2 = 0.970$.
- **Tab 3: Optical & Color Space Explorer** — Interactive visualizer of Beer-Lambert $OD_G$ vs $A_{520}$ and CIELAB trajectories.
- **Tab 4: Model Benchmarks & Validation** — Stage A through E performance tables and residual plots.
- **Tab 5: Data Expansion & Experimental Protocol** — Instructions on expanding the dataset.

---

## 📥 How to Add More Images and Spectroscopy Data

For supervised model training and validation, each sample requires **both its reaction tube photograph and its corresponding laboratory measurements ($A_{520}$ and known dilution %)**.

### Master Manifest: [`data/raw/experiment_manifest.csv`](file:///c:/Users/Parijat/Desktop/honey-adulteration-detection/data/raw/experiment_manifest.csv)

To add new samples:
1. **Save the tube photos** in `data/raw/` (e.g. `data/raw/honey_mustard/mustard_100.jpg`, `data/raw/honey_mustard/mustard_080.jpg`).
2. **Open [`data/raw/experiment_manifest.csv`](file:///c:/Users/Parijat/Desktop/honey-adulteration-detection/data/raw/experiment_manifest.csv)** and add rows for your new samples:

```csv
sample_id,sample_type,honey_id,honey_type,honey_pct,adulteration_pct,a520_rep1,a520_rep2,proline_ug_ml,image_path
MUSTARD_100,honey,Mustard,Mustard floral,100.0,0.0,0.380,0.385,,data/raw/honey_mustard/mustard_100.jpg
MUSTARD_080,honey,Mustard,Mustard floral,80.0,20.0,0.310,0.315,,data/raw/honey_mustard/mustard_080.jpg
```

*Note: You can also refer to the starter template in [`data/raw/new_samples_template.csv`](file:///c:/Users/Parijat/Desktop/honey-adulteration-detection/data/raw/new_samples_template.csv).*

3. **Re-run the pipeline**:
   ```bash
   python main.py --build-data
   python main.py --train
   ```
The pipeline dynamically reads the manifest, calculates the updated proline calibration line, extracts multi-color features from the new tube photos, retrains all 5 modeling stages, and updates all benchmark charts!

*(Note: For evaluating a test image where you don't know the truth or don't have $A_{520}$, you do NOT need to edit the manifest—simply run `python main.py --predict --image path/to/tube.jpg` or upload it in the Streamlit app).*

