# SpectraHoney™ — Project Write-up & PPT Slide Contents

---

## WRITE-UP

### Title
**Low-Cost Honey Adulteration Detection via Ninhydrin-Proline Spectrophotometry and Smartphone Digital Image Analysis**

---

### Abstract
Honey adulteration with sugar syrups is a widespread food fraud issue. Standard detection methods (HPLC, NMR) are expensive and inaccessible. This work presents a low-cost, multi-stage analytical pipeline that couples the ninhydrin-proline colorimetric assay (peak absorbance 520 nm) with digital photograph analysis of reaction tubes using computer vision and machine learning. Five model stages (A–E) predict proline concentration and adulteration percentage from a smartphone photo, with or without a benchtop spectrophotometer reading. The system is validated against the Codex Alimentarius minimum threshold (≥ 180 mg/kg proline) and deployed as an interactive web platform.

---

### 1. Introduction
- Honey is adulterated by diluting with corn syrup, HFCS, invert sugar syrup.
- Proline is the dominant free amino acid in authentic honey (~180–2000 mg/kg). Its concentration drops proportionally with syrup dilution.
- Ninhydrin reagent reacts with proline → purple-pink chromophore, peak absorbance at 520 nm (Beer-Lambert Law).
- **Problem**: Benchtop spectrophotometers cost thousands of dollars. Proposed solution: use the **green camera channel** as a proxy for 520 nm absorbance.

---

### 2. Materials & Methods

#### 2.1 Honey Samples
- **H2**: Light floral honey (Acacia/Clover type)
- **H4**: Dark amber honey (Buckwheat/Forest type)
- Dilution series: 100%, 80%, 60%, 40%, 20% honey (balanced with corn syrup)
- 8 proline standard solutions (0–200 µg/mL)
- **Total dataset: N = 18 reaction tubes**

#### 2.2 Ninhydrin Assay Protocol
1. Dissolve 5 g honey in 100 mL distilled water (dilution factor = 20×)
2. Add ninhydrin reagent, heat at 100°C for 20 min
3. Photograph the cooled tube against white backlight
4. Optionally measure A₅₂₀ with benchtop spectrophotometer

#### 2.3 Digital Image Feature Extraction
From each tube photograph, a ROI (Region of Interest) is extracted (glare-filtered), and the following features are computed:

| Feature | Description |
|---|---|
| `R, G, B` | Mean RGB channel values |
| `OD_G` | Green channel optical density: −log₁₀(G / G_blank) |
| `RG_ratio` | R/G ratio (sensitive to dilution) |
| `H, S, V` | HSV color space |
| `L*, a*, b*` | CIELAB perceptual color space |

#### 2.4 Calibration Curve (Standard Curve)
- **Blank-corrected Beer-Lambert model**:
  
  **(A₅₂₀ − 0.1125) = 0.00476 × C_proline (µg/mL)**
  
- R² = 0.9848, N = 8 standards
- Blank correction eliminates the intercept bias that caused the "0% proline" bug in unconstrained OLS.

#### 2.5 Five-Stage ML Pipeline

| Stage | Input | Output | Best Model | LOOCV Performance |
|---|---|---|---|---|
| A | Image OD_G | Estimated A₅₂₀ | Linear Regression | R² = 0.9224, RMSE = 0.065 |
| B | Image multi-color features | Adulteration % | Ridge Regression | MAE = 15.34% |
| C | Benchtop A₅₂₀ + proline | Adulteration % | Ridge Regression | MAE = 15.31% |
| D | Image + A₅₂₀ fusion | Adulteration % | Ridge Regression | MAE = 17.14% |
| E | All features | PASS / FAIL | Random Forest Classifier | Accuracy = 90% |

- **Consensus estimate**: weighted average of Stage B, C, D predictions
- **Honey Authenticity Index (HAI)**: 0–100 score derived from proline concentration, adulteration %, and Stage E classifier confidence

---

### 3. Results

#### 3.1 Beer-Lambert Coupling Validated
- Green camera channel OD_G correlates with benchtop A₅₂₀ at **Pearson r = 0.969**
- Demonstrates that a smartphone camera can emulate a 520 nm spectrophotometer under controlled lighting.

#### 3.2 Cross-Botanical Overlap Problem
- H2 pure honey (A₅₂₀ ≈ 0.3125) overlaps in raw RGB with H4 60%-diluted (A₅₂₀ ≈ 0.375)
- Solution: CIELAB (a* vs b*) decouples the two botanical types in perceptual color space
- Botanical Prior selector in the web UI allows the user to hint the honey type

#### 3.3 Model Limitations
- Dataset N = 18 is a pilot study; LOOCV MAE of ~15% is acceptable for a proof-of-concept
- Leave-one-honey-out (LOHO) cross-validation MAE = 26.18% — highlights botanical generalization gap
- No out-of-distribution (OOD) detection: model will hallucinate on non-honey-tube inputs

---

### 4. Web Platform (SpectraHoney™)
Built with **Streamlit**, deployed locally at `localhost:8501`.

**Tabs:**
1. **Live Analytical Predictor** — upload or select a tube photo, get instant prediction
2. **Proline Standard Calibration** — view standard curve and replicate data
3. **Optical & Color Space Explorer** — Beer-Lambert plots, CIELAB color trajectories
4. **Model Benchmarks & Validation** — LOOCV and LOHO performance tables
5. **Laboratory Protocol & Manifest** — SOP for data expansion and retraining

**Key UI Features:**
- Amazon-inspired greenish-white + honey-gold theme
- Predict button visible without scrolling (no sidebar)
- Results hidden until user clicks "Run Authenticity Prediction"
- Multi-stage analytical details hidden behind expander

---

### 5. Discussion & Future Work
- **Expand dataset**: Add Acacia, Manuka, Sidr, Rapeseed, multifloral varieties with 5–10 dilution levels each
- **OOD guardrails**: Stage 0 hue-range validator to reject non-tube inputs
- **Portable device**: Pair with a 3D-printed tube holder + fixed LED backlight for field use
- **Regulatory**: Align with EU Commission Regulation 2023/803 (updated honey composition rules)
- **Deployment**: Streamlit Cloud or mobile app wrapper (Pydroid / Android WebView)

---

### 6. Conclusion
This work demonstrates that smartphone digital image analysis of ninhydrin-stained proline assay tubes can serve as a low-cost proxy for 520 nm spectrophotometry, achieving R² = 0.922 for absorbance emulation and 90% binary authenticity classification accuracy on a pilot dataset of 18 honey/standard tubes across two botanical origins.

---

---

## PPT SLIDE OUTLINE

### Slide 1 — Title
- **Title**: Low-Cost Honey Adulteration Detection using Ninhydrin-Proline Assay & Smartphone Image Analysis
- **Subtitle**: A Multi-Stage Machine Learning Pipeline
- **Authors / Institution / Date**
- Background: photo of reaction tubes (pink-purple gradient from 0% to 100% honey)

---

### Slide 2 — The Problem
- Global honey fraud: ~30% of commercial honey is adulterated (cite FAO/EU reports)
- Adulteration agents: corn syrup, HFCS, invert sugar
- Current gold standard: HPLC, NMR — expensive, lab-only
- **Our Goal**: Field-deployable, \$0 instrument cost beyond a smartphone

---

### Slide 3 — Scientific Principle
- Ninhydrin + Proline → purple chromophore (Ruhemann's purple), λ_max = 520 nm
- Beer-Lambert Law: A = ε × l × C
- **Key insight**: Green camera channel (510–540 nm) ≈ 520 nm spectrophotometer
- Diagram: Tube → Smartphone Camera → Green Channel → OD_G

---

### Slide 4 — Experimental Setup
- Two honey types: H2 (light floral), H4 (dark amber)
- Dilution series: 100%, 80%, 60%, 40%, 20% + 8 proline standards
- Photo of actual tube series (gradient visual)
- Simple setup: white paper backlight + phone camera

---

### Slide 5 — Standard Calibration Curve
- Plot: `proline_standard_curve.png`
- Equation: (A₅₂₀ − 0.1125) = 0.00476 × C, R² = 0.9848
- Table: 8-point standard series with replicate A₅₂₀ readings

---

### Slide 6 — Beer-Lambert Validation
- Plot: `beer_lambert_od_vs_a520.png`
- OD_G vs benchtop A₅₂₀, Pearson r = 0.969
- Statement: *"The green camera channel reliably emulates a 520 nm spectrophotometer"*

---

### Slide 7 — Feature Engineering
- Table of extracted features: RGB, OD_G, RG_ratio, HSV, CIELAB
- Plot: `color_trajectory_adulteration.png` — CIELAB a* vs b* separating H2 and H4
- Highlight: a*/b* decouples botanical types that overlap in raw RGB

---

### Slide 8 — Five-Stage ML Pipeline
- Flowchart:
  ```
  Photo → [Stage A] → Estimated A₅₂₀
  Photo → [Stage B] → Adulteration %
  A₅₂₀ → [Stage C] → Adulteration %  (optional, if benchtop available)
  Photo+A₅₂₀ → [Stage D] → Adulteration %
  All → [Stage E] → PASS / FAIL
         ↓
      Consensus HAI Score
  ```
- One line per stage: model type + best LOOCV metric

---

### Slide 9 — Model Performance
- Table: Stages A–E with LOOCV MAE / R² / Accuracy
- Bar chart: `model_benchmarks_stages.png`
- Honest note: LOHO MAE = 26.18% (botanical generalization gap — needs more data)

---

### Slide 10 — SpectraHoney™ Web Platform
- Screenshot of the UI (Live Analytical Predictor tab)
- Features bullet list: upload/select tube, botanical prior, instant prediction, expander for details
- Tech stack: Python · Streamlit · scikit-learn · OpenCV · PIL

---

### Slide 11 — Limitations & Future Work
| Limitation | Future Fix |
|---|---|
| N = 18 pilot dataset | Expand to 10+ botanical origins × 7 dilution levels |
| OOD hallucination (tea → 100% authentic) | Stage 0 hue validator + OOD classifier |
| Lighting sensitivity | 3D-printed fixed holder + LED backlight |
| Two honey types only | Commercial multi-floral varieties |

---

### Slide 12 — Conclusion
- Smartphone camera green channel ≈ 520 nm spectrophotometer (r = 0.969)
- 5-stage ML pipeline: R² = 0.922 (spectro emulation), 90% authenticity classification
- Deployed as interactive web platform
- **Impact**: Enables low-cost honey authentication in resource-limited settings
- **Next step**: Expand dataset + field validation trial

---

### Slide 13 — Thank You / Q&A
- Contact / acknowledgements
- QR code linking to `localhost:8501` (or deployed URL)
- Key references: Codex Stan 12-1981, IHC Harmonized Methods, Beer-Lambert Law

---

## KEY FIGURES TO EMBED IN PPT
1. `figures/proline_standard_curve.png`
2. `figures/beer_lambert_od_vs_a520.png`
3. `figures/color_trajectory_adulteration.png`
4. `figures/correlation_heatmap.png`
5. `figures/model_benchmarks_stages.png`
6. Screenshot of SpectraHoney™ web UI
7. Photo of actual reaction tube gradient (H2 or H4 dilution series)
