"""
SpectraHoney™ — Greenish-White & Honey Gold Analytical Phenotyping Platform.
Features:
- Botanical Deep Emerald (#0F3822) & Honey Gold (#D4AF37) Theme.
- Greenish-White Crisp Background (#F4F8F4).
- Functional "Back to Top ↑" Anchor Navigation.
- Equal Box Sizes, Predict Button, & Clutter-Free Results View.
- Finger Pointer Cursors on all Clickable Elements.
"""
import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    MIN_AUTHENTIC_PROLINE_MG_KG,
    ASSAY_DILUTION_FACTOR
)
from src.features.roi_detector import TubeROIDetector
from src.features.extractor import TubeFeatureExtractor
from predict import HoneyAdulterationPredictor

# Page Configuration - Collapsed sidebar
st.set_page_config(
    page_title="SpectraHoney | Botanical & Honey Gold Portal",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Anchor tag at the very top for Back to Top functionality
st.markdown('<div id="top" style="height: 0; width: 0; margin: 0; padding: 0;"></div>', unsafe_allow_html=True)

# Greenish-White & Honey Golden Theme Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html {
        scroll-behavior: smooth !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F4F8F4 !important; /* Elegant Greenish-White */
        color: #111827 !important;           /* High contrast dark text */
    }

    /* Completely hide default Streamlit sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    div[data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Pointer Cursor on all interactive and clickable elements */
    button, [role="button"], a, input, select, textarea,
    .stSelectbox, .stRadio, .stFileUploader, .stButton, div[data-baseweb="select"],
    .stCheckbox, label, [data-baseweb="tab"], [data-baseweb="radio"], .clickable-card,
    div[data-testid="stFileUploadDropzone"], span[data-baseweb="tag"], .gold-back-to-top {
        cursor: pointer !important;
    }
    button:hover, [role="button"]:hover, a:hover {
        cursor: pointer !important;
    }

    /* Top Header Bar (Deep Botanical Emerald #0F3822 with Honey Gold #D4AF37) */
    .amz-header-main {
        background-color: #0F3822;
        border-bottom: 2px solid #D4AF37;
        padding: 14px 28px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -4rem -4rem 0 -4rem;
        box-shadow: 0 3px 8px rgba(15, 56, 34, 0.25);
    }
    .amz-logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .amz-logo-text {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .amz-logo-accent {
        color: #F3C044;
        font-weight: 800;
    }
    .amz-deliver-tag {
        font-size: 0.78rem;
        color: #D1E7DD;
        line-height: 1.2;
        margin-left: 20px;
        border-left: 1px solid #235338;
        padding-left: 15px;
    }
    .amz-deliver-bold {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F3C044;
    }
    .amz-header-right {
        display: flex;
        align-items: center;
        gap: 28px;
        font-size: 0.85rem;
    }
    .amz-header-link {
        color: #FFFFFF;
        text-decoration: none;
        display: flex;
        flex-direction: column;
    }
    .amz-header-link-small {
        font-size: 0.72rem;
        color: #A3CFBB;
    }
    .amz-header-link-bold {
        font-weight: 700;
        color: #FFFFFF;
    }

    /* Sub-Navbar (Forest Green #1B4D33) */
    .amz-subnav {
        background-color: #1B4D33;
        padding: 9px 28px;
        color: #FFFFFF;
        margin: 0 -4rem 18px -4rem;
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .amz-subnav-badge {
        color: #F3C044;
        font-weight: 700;
        margin-right: 10px;
    }

    /* Native Streamlit Container Styling (Crisp White Card with Soft Botanical Sage Border) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DDE6DE !important;
        border-top: 3px solid #D4AF37 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(15, 56, 34, 0.05) !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
    }

    /* Card Header Titles */
    .amz-card-title {
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #0F3822 !important;
        margin-bottom: 12px !important;
        padding-bottom: 8px !important;
        border-bottom: 1px solid #E5EBE6 !important;
    }

    /* Primary Predict Button (Honey Golden Amber Gradient) */
    .stButton > button {
        background: linear-gradient(180deg, #F5BA42 0%, #D49B24 100%) !important;
        border: 1px solid #B8860B !important;
        color: #0F291E !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        box-shadow: 0 2px 6px rgba(212, 155, 36, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
        font-size: 1.02rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #F3C044 0%, #E5A93C 100%) !important;
        box-shadow: 0 3px 8px rgba(212, 155, 36, 0.5) !important;
        border-color: #996515 !important;
    }

    /* Reset Button */
    .amz-reset-btn button {
        background: #FFFFFF !important;
        border: 1px solid #DDE6DE !important;
        color: #0F3822 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 5px 14px !important;
        font-size: 0.85rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        width: auto !important;
    }
    .amz-reset-btn button:hover {
        background: #F4F8F4 !important;
        border-color: #2D6A4F !important;
    }

    /* Constrain preview image heights to fit within viewport without scrolling */
    div[data-testid="stImage"] img {
        max-height: 230px !important;
        object-fit: contain !important;
    }

    /* Equal Metric Boxes */
    .amz-metric-box {
        background-color: #F8FAF8;
        border: 1px solid #DDE6DE;
        border-radius: 8px;
        padding: 16px 12px;
        text-align: center;
        height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .amz-metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F3822;
        line-height: 1.1;
    }
    .amz-metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #4B6354;
        text-transform: uppercase;
        margin-top: 5px;
        letter-spacing: 0.04em;
    }
    .amz-metric-sub {
        font-size: 0.72rem;
        color: #D4AF37;
        font-weight: 700;
        margin-top: 2px;
    }

    /* Banners */
    .amz-banner-pass {
        background-color: #F0FDF4;
        border: 1.5px solid #22C55E;
        border-left: 6px solid #15803D;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .amz-banner-warn {
        background-color: #FFFBEB;
        border: 1.5px solid #F59E0B;
        border-left: 6px solid #D97706;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .amz-banner-fail {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        border-left: 6px solid #DC2626;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .amz-banner-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0;
        color: #111827;
    }
    .amz-prime-badge {
        display: inline-block;
        background-color: #0F3822;
        color: #F3C044;
        border: 1px solid #D4AF37;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 3px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Functional Back to Top Button */
    .gold-back-to-top {
        background: #1B4D33;
        color: #F3C044 !important;
        border: 1px solid #D4AF37;
        padding: 12px 20px;
        font-weight: 700;
        text-align: center;
        border-radius: 6px;
        margin: 30px -4rem 0 -4rem;
        transition: all 0.2s ease;
        display: block;
        text-decoration: none;
    }
    .gold-back-to-top:hover {
        background: #0F3822;
        color: #FFFFFF !important;
    }

    /* Footer */
    .amz-footer {
        background-color: #0F3822;
        border-top: 1px solid #D4AF37;
        color: #D1E7DD;
        padding: 24px 20px;
        margin: 0 -4rem -4rem -4rem;
        text-align: center;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Botanical Emerald & Honey Gold Top Header ---
st.markdown("""
<div class="amz-header-main">
    <div class="amz-logo-container">
        <span style="font-size: 1.85rem;">🍯</span>
        <span class="amz-logo-text">Spectra<span class="amz-logo-accent">Honey</span></span>
        <div class="amz-deliver-tag">
            <span>Analytical Division:</span><br>
            <span class="amz-deliver-bold">Botanical Authenticity & Spectroscopy Bench</span>
        </div>
    </div>
    <div class="amz-header-right">
        <div class="amz-header-link">
            <span class="amz-header-link-small">Standard Protocol</span>
            <span class="amz-header-link-bold">Codex Stan 12-1981 / IHC</span>
        </div>
        <div class="amz-header-link">
            <span class="amz-header-link-small">Optical Coupling</span>
            <span class="amz-header-link-bold">520 nm (A520) • Green OD</span>
        </div>
        <div class="amz-header-link">
            <span class="amz-header-link-small">Benchmark Library</span>
            <span class="amz-header-link-bold">🛒 18 Verified Standards</span>
        </div>
    </div>
</div>
<div class="amz-subnav">
    <span class="amz-subnav-badge">🌿 BOTANICAL LAB MODULES</span>
    <span>• Green Channel Optical Density (Beer-Lambert OD_G)</span>
    <span>• Ninhydrin-Proline Chromophore Coupling</span>
    <span>• Codex Regulatory Minimum: 180 mg/kg</span>
</div>
""", unsafe_allow_html=True)


# Session State Initialization
if "module" not in st.session_state:
    st.session_state.module = "🧪 Live Analytical Predictor"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "has_spectro" not in st.session_state:
    st.session_state.has_spectro = False
if "measured_a520_val" not in st.session_state:
    st.session_state.measured_a520_val = 0.350
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Select Verified Benchmark Tube"
if "preset_choice" not in st.session_state:
    st.session_state.preset_choice = "H4: 100% Pure Dark Amber Honey (0% adulteration)"
if "floral_prior" not in st.session_state:
    st.session_state.floral_prior = "Auto-detect / Generic Model (Codex Global Cutoff ≥ 180 mg/kg)"
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False


@st.cache_resource
def load_predictor():
    return HoneyAdulterationPredictor()


@st.cache_data
def load_datasets():
    df_master = pd.read_csv(PROCESSED_DATA_DIR / "master_dataset.csv")
    df_std = pd.read_csv(PROCESSED_DATA_DIR / "proline_standards.csv")
    df_honey = pd.read_csv(PROCESSED_DATA_DIR / "honey_dilutions.csv")
    with open(PROCESSED_DATA_DIR / "dataset_metadata.json", "r") as f:
        meta = json.load(f)
    with open(MODELS_DIR / "model_benchmark_results.json", "r") as f:
        benchmarks = json.load(f)
    return df_master, df_std, df_honey, meta, benchmarks


predictor = load_predictor()
df_master, df_std, df_honey, metadata, benchmarks = load_datasets()


# Horizontal Header Navigation
header_tabs = [
    "🧪 Live Analytical Predictor",
    "📊 Proline Standard Calibration",
    "🔬 Optical & Color Space Explorer",
    "📈 Model Benchmarks & Validation",
    "📋 Laboratory Protocol & Manifest"
]

selected_tab = st.radio(
    "Amazon Header Navigation Bar",
    header_tabs,
    index=header_tabs.index(st.session_state.module) if st.session_state.module in header_tabs else 0,
    horizontal=True,
    label_visibility="collapsed"
)

if selected_tab != st.session_state.module:
    st.session_state.module = selected_tab
    st.session_state.has_predicted = False
    st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: LIVE ANALYTICAL PREDICTOR & MOBILE SIMULATOR
# ==============================================================================
if st.session_state.module == "🧪 Live Analytical Predictor":
    col_input, col_result = st.columns([1, 1], gap="medium")

    # --- LEFT CONTAINER: INPUT & CONTROLS ---
    with col_input:
        with st.container(border=True):
            # Header with Reset Button
            h_left, h_right = st.columns([3, 1])
            with h_left:
                st.markdown('<div class="amz-card-title">1. Test Sample Acquisition & Controls</div>', unsafe_allow_html=True)
            with h_right:
                st.markdown('<div class="amz-reset-btn">', unsafe_allow_html=True)
                if st.button("🔄 Reset All", key="btn_reset_all", help="Reset all selections and clear previous results"):
                    st.session_state.uploader_key += 1
                    st.session_state.has_spectro = False
                    st.session_state.measured_a520_val = 0.350
                    st.session_state.input_mode = "Select Verified Benchmark Tube"
                    st.session_state.preset_choice = "H4: 100% Pure Dark Amber Honey (0% adulteration)"
                    st.session_state.floral_prior = "Auto-detect / Generic Model (Codex Global Cutoff ≥ 180 mg/kg)"
                    st.session_state.has_predicted = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            # Input Mode Selection
            input_choice = st.radio(
                "Input Mode:",
                ["Select Verified Benchmark Tube", "Upload Laboratory Photograph"],
                index=0 if st.session_state.input_mode == "Select Verified Benchmark Tube" else 1,
                horizontal=True,
                key="radio_input_mode"
            )
            if input_choice != st.session_state.input_mode:
                st.session_state.input_mode = input_choice
                st.session_state.has_predicted = False
                st.rerun()

            sample_img = None
            sample_name = ""

            if input_choice == "Select Verified Benchmark Tube":
                preset_options = {
                    "H4: 100% Pure Dark Amber Honey (0% adulteration)": "data/raw/honey_h4/h4_100pct_honey.jpg",
                    "H4: 80% Honey / 20% Diluted with Syrup": "data/raw/honey_h4/h4_080pct_honey.jpg",
                    "H4: 60% Honey / 40% Diluted with Syrup": "data/raw/honey_h4/h4_060pct_honey.jpg",
                    "H4: 40% Honey / 60% Diluted with Syrup": "data/raw/honey_h4/h4_040pct_honey.jpg",
                    "H4: 20% Honey / 80% Diluted with Syrup": "data/raw/honey_h4/h4_020pct_honey.jpg",
                    "H2: 100% Pure Light Floral Honey (0% adulteration)": "data/raw/honey_h2/h2_100pct_honey.jpg",
                    "H2: 80% Honey / 20% Diluted with Syrup": "data/raw/honey_h2/h2_080pct_honey.jpg",
                    "H2: 60% Honey / 40% Diluted with Syrup": "data/raw/honey_h2/h2_060pct_honey.jpg",
                    "H2: 40% Honey / 60% Diluted with Syrup": "data/raw/honey_h2/h2_040pct_honey.jpg",
                    "H2: 20% Honey / 80% Diluted with Syrup": "data/raw/honey_h2/h2_020pct_honey.jpg",
                    "Standard: 200 µg/mL Reference Proline": "data/raw/standards/std_200ug_ml.jpg",
                    "Standard: 100 µg/mL Reference Proline": "data/raw/standards/std_100ug_ml.jpg",
                    "Standard: 0 µg/mL Assay Blank (Control)": "data/raw/standards/std_000ug_ml.jpg"
                }
                preset_keys = list(preset_options.keys())
                curr_idx = preset_keys.index(st.session_state.preset_choice) if st.session_state.preset_choice in preset_keys else 0
                
                selected_preset = st.selectbox(
                    "Select Verified Benchmark Tube:",
                    preset_keys,
                    index=curr_idx,
                    key="select_benchmark_tube"
                )
                if selected_preset != st.session_state.preset_choice:
                    st.session_state.preset_choice = selected_preset
                    st.session_state.has_predicted = False
                    st.rerun()

                sample_img_path = preset_options[selected_preset]
                sample_img = Image.open(sample_img_path)
                sample_name = selected_preset
            else:
                uploaded_file = st.file_uploader(
                    "Upload Tube Photograph (JPG, PNG):",
                    type=["jpg", "jpeg", "png"],
                    key=f"uploader_{st.session_state.uploader_key}"
                )
                if uploaded_file is not None:
                    sample_img = Image.open(uploaded_file)
                    sample_name = uploaded_file.name
                else:
                    st.session_state.has_predicted = False
                    st.info("📁 Drag and drop or upload a reaction tube photo to begin.")

            # Botanical Prior Selector
            st.markdown("<hr style='margin: 10px 0; border: 0.5px solid #E5EBE6;'>", unsafe_allow_html=True)
            floral_prior_options = [
                "Auto-detect / Generic Model (Codex Global Cutoff ≥ 180 mg/kg)",
                "Light Floral Honey (e.g., Acacia, Clover, Mustard, H2 Baseline)",
                "Dark Amber / Forest Honey (e.g., Buckwheat, Heather, Forest, H4 Baseline)"
            ]
            curr_prior_idx = floral_prior_options.index(st.session_state.floral_prior) if st.session_state.floral_prior in floral_prior_options else 0
            
            floral_prior_choice = st.selectbox(
                "Honey Botanical & Color Grade Prior (Optional):",
                floral_prior_options,
                index=curr_prior_idx,
                key="select_floral_prior"
            )
            if floral_prior_choice != st.session_state.floral_prior:
                st.session_state.floral_prior = floral_prior_choice
                st.session_state.has_predicted = False
                st.rerun()

            floral_hint = None
            if "H2" in floral_prior_choice or "Light" in floral_prior_choice:
                floral_hint = "H2"
            elif "H4" in floral_prior_choice or "Dark" in floral_prior_choice:
                floral_hint = "H4"

            # Optional Benchtop Spectrophotometer Input
            has_spectro = st.checkbox(
                "Pair with measured benchtop A520 reading",
                value=st.session_state.has_spectro,
                key="chk_spectro"
            )
            if has_spectro != st.session_state.has_spectro:
                st.session_state.has_spectro = has_spectro
                st.session_state.has_predicted = False
                st.rerun()

            measured_a520 = None
            if has_spectro:
                measured_a520 = st.number_input(
                    "Benchtop Spectrophotometer Absorbance (520 nm):",
                    min_value=0.000, max_value=3.000, value=float(st.session_state.measured_a520_val), step=0.005, format="%.4f",
                    key="num_a520"
                )
                if measured_a520 != st.session_state.measured_a520_val:
                    st.session_state.measured_a520_val = measured_a520
                    st.session_state.has_predicted = False

            # Equal Sized Image Previews
            if sample_img is not None:
                st.markdown("<hr style='margin: 10px 0; border: 0.5px solid #E5EBE6;'>", unsafe_allow_html=True)
                roi, bbox, valid_mask = predictor.extractor.roi_detector.extract_roi(sample_img)
                
                c_full, c_roi = st.columns(2)
                with c_full:
                    st.markdown("""
                    <div style="font-size: 0.8rem; font-weight: 700; color: #4B6354; margin-bottom: 4px;">
                        Acquired Photograph
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(sample_img, use_container_width=True)
                with c_roi:
                    st.markdown("""
                    <div style="font-size: 0.8rem; font-weight: 700; color: #4B6354; margin-bottom: 4px;">
                        Extracted Liquid ROI (Glare Filtered)
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(roi, use_container_width=True)

    # --- RIGHT CONTAINER: DIAGNOSTIC REPORT (CLEAN FRONT VIEW, BUTTON ON RIGHT) ---
    with col_result:
        with st.container(border=True):
            st.markdown('<div class="amz-card-title">2. Analytical Evaluation & Diagnostic Summary</div>', unsafe_allow_html=True)

            # Prominent Action Button directly on the right of the pictures (Zero Scrolling Needed!)
            if st.button("⚡ Run Authenticity Prediction", key="btn_run_predict", type="primary", disabled=(sample_img is None)):
                st.session_state.has_predicted = True

            st.markdown("<hr style='margin: 12px 0; border: 0.5px solid #E5EBE6;'>", unsafe_allow_html=True)

            if sample_img is not None and st.session_state.has_predicted:
                res = predictor.predict_image(sample_img, actual_a520=measured_a520, honey_variety_hint=floral_hint)
                preds = res["predictions"]
                auth = res["authenticity_assessment"]
                feats = res["image_features"]

                # 1. Primary Diagnostic Status Banner
                if auth["verdict"] == "PASS":
                    st.markdown(f"""
                    <div class="amz-banner-pass">
                        <span class="amz-prime-badge">✓ PRIME AUTHENTICITY VERIFIED</span>
                        <div class="amz-banner-title">STATUS: AUTHENTIC / GENUINE HONEY</div>
                        <div style="font-size: 0.85rem; color: #15803D; margin-top: 3px;">
                            Complies with Codex Alimentarius & IHC Standards. Proline: {preds['estimated_proline_mg_kg']:.0f} mg/kg (≥180 mg/kg minimum).
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif auth["verdict"] == "WARNING":
                    st.markdown(f"""
                    <div class="amz-banner-warn">
                        <span class="amz-prime-badge" style="background-color: #B45309; color: #FFFFFF;">⚠ QUALITY ADVISORY</span>
                        <div class="amz-banner-title">STATUS: BORDERLINE / SUSPECT DILUTION</div>
                        <div style="font-size: 0.85rem; color: #92400E; margin-top: 3px;">
                            Dilution detected or proline concentration degraded. Requires secondary laboratory testing.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="amz-banner-fail">
                        <span class="amz-prime-badge" style="background-color: #B91C1C; color: #FFFFFF;">✗ FAILED REGULATORY SPECIFICATION</span>
                        <div class="amz-banner-title">STATUS: ADULTERATED / SUBSTANDARD</div>
                        <div style="font-size: 0.85rem; color: #B91C1C; margin-top: 3px;">
                            Significant sugar syrup adulteration detected ({preds['consensus_adulteration_pct']:.1f}%). Proline severely deficient.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Three Symmetrical Metric Boxes (Core Answers)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="amz-metric-box">
                        <div class="amz-metric-val">{auth['hai_score']:.1f}<span style="font-size: 1rem; color: #D4AF37;">/100</span></div>
                        <div class="amz-metric-label">Authenticity Index</div>
                        <div class="amz-metric-sub">Confidence: {auth['confidence_pct']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    color_pct = '#DC2626' if preds['consensus_adulteration_pct'] >= 15 else '#15803D'
                    st.markdown(f"""
                    <div class="amz-metric-box">
                        <div class="amz-metric-val" style="color: {color_pct};">{preds['consensus_adulteration_pct']:.1f}%</div>
                        <div class="amz-metric-label">Predicted Adulteration</div>
                        <div class="amz-metric-sub">Syrup Dilution</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    color_pro = '#15803D' if preds['estimated_proline_mg_kg'] >= 180 else '#DC2626'
                    st.markdown(f"""
                    <div class="amz-metric-box">
                        <div class="amz-metric-val" style="color: {color_pro};">{preds['estimated_proline_mg_kg']:.0f}</div>
                        <div class="amz-metric-label">Proline (mg/kg)</div>
                        <div class="amz-metric-sub">Codex Min: 180 mg/kg</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Authenticity Progress Bar
                st.write("**Authenticity Scale:**")
                st.progress(float(auth["hai_score"]) / 100.0)

                # 4. Collapsible Detailed Parameters (Clean Front View)
                with st.expander("🔬 View Detailed Multi-Stage Analytical Parameters & Optical Signatures"):
                    summary_df = pd.DataFrame({
                        "Parameter / Model": [
                            "Stage A: Image-Estimated A520",
                            "Benchtop Measured A520",
                            "Calculated Proline (Assay Conc.)",
                            "Proline in Honey (Dilution Factor Adjusted)",
                            "Stage B: Image-Only Predicted Adulteration",
                            "Stage C: Spectrophotometric Adulteration",
                            "Stage D: Multimodal Fusion Adulteration",
                            "Consensus Final Adulteration Estimate"
                        ],
                        "Observed / Estimated Value": [
                            f"{preds['stage_a_predicted_a520']:.4f} AU",
                            f"{preds['actual_a520']:.4f} AU" if preds['actual_a520'] else "Not provided (Vision Estimated)",
                            f"{preds['estimated_proline_ug_ml']:.2f} µg/mL",
                            f"{preds['estimated_proline_mg_kg']:.1f} mg/kg",
                            f"{preds['stage_b_image_adulteration_pct']:.1f}%",
                            f"{preds['stage_c_spectro_adulteration_pct']:.1f}%",
                            f"{preds['stage_d_multimodal_adulteration_pct']:.1f}%",
                            f"{preds['consensus_adulteration_pct']:.1f}%"
                        ]
                    })
                    st.table(summary_df)
                    st.markdown("**Raw Multi-Color Space Signatures:**")
                    st.json(feats)

            elif sample_img is not None and not st.session_state.has_predicted:
                st.markdown("""
                <div style="text-align: center; padding: 100px 20px; color: #4B6354;">
                    <div style="font-size: 3.2rem;">⚡</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F3822; margin-top: 10px;">Ready for Evaluation</div>
                    <div style="font-size: 0.9rem; margin-top: 4px;">Sample is selected. Click the golden <b>'⚡ Run Authenticity Prediction'</b> button above to analyze.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 100px 20px; color: #4B6354;">
                    <div style="font-size: 3.2rem;">📦</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F3822; margin-top: 10px;">Awaiting Reaction Tube Input</div>
                    <div style="font-size: 0.9rem; margin-top: 4px;">Please choose a verified benchmark tube or upload a photograph on the left.</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: PROLINE STANDARD CALIBRATION
# ==============================================================================
elif st.session_state.module == "📊 Proline Standard Calibration":
    with st.container(border=True):
        st.markdown('<div class="amz-card-title">Ninhydrin-Proline Assay Calibration Curve (520 nm)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.2, 1], gap="large")
        with col1:
            st.image(str(FIGURES_DIR / "proline_standard_curve.png"), use_container_width=True)

        with col2:
            st.subheader("Linear Calibration Model")
            st.latex(r"(A_{520} - 0.1125) = 0.00476 \times C_{\text{proline}}\,(\mu\text{g/mL})")
            st.markdown(f"""
            - **Linearity ($R^2$):** `{metadata['standard_curve']['r2']:.4f}`
            - **Calibration Slope:** `{metadata['standard_curve']['slope']:.5f} AU / (µg/mL)`
            - **Assay Reagent Blank ($A_0$ at 0 µg/mL):** `{metadata['standard_curve']['blank_a520']:.4f} AU`
            - **Sample Dilution Factor:** `{metadata['dilution_factor']}×`
            - **Codex / IHC Minimum Threshold:** `≥ {metadata['authentic_min_proline_mg_kg']} mg/kg`
            """)

            st.subheader("Standard Series Replicate Data")
            st.dataframe(df_std[["sample_id", "proline_ug_ml", "a520_rep1", "a520_rep2", "a520_mean"]], hide_index=True)


# ==============================================================================
# TAB 3: OPTICAL & COLOR SPACE EXPLORER
# ==============================================================================
elif st.session_state.module == "🔬 Optical & Color Space Explorer":
    with st.container(border=True):
        st.markdown('<div class="amz-card-title">Beer-Lambert Law in Digital Photography: Green OD vs A520</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Under Beer-Lambert's Law, transmitted green light decreases logarithmically with chromophore concentration.
        Defining **Green Channel Optical Density**:
        $$OD_G = -\\log_{10}(G / G_{\\text{blank}})$$
        """)
        st.image(str(FIGURES_DIR / "beer_lambert_od_vs_a520.png"), use_container_width=True)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.image(str(FIGURES_DIR / "color_trajectory_adulteration.png"), use_container_width=True)
        with col2:
            st.image(str(FIGURES_DIR / "correlation_heatmap.png"), use_container_width=True)

        st.subheader("Extracted Multi-Color Feature Master Matrix")
        st.dataframe(
            df_master[["sample_id", "honey_id", "honey_pct", "adulteration_pct", "a520_mean", "od_g", "rg_ratio", "a_star", "b_star", "l_star"]],
            hide_index=True
        )


# ==============================================================================
# TAB 4: MODEL BENCHMARKS & VALIDATION
# ==============================================================================
elif st.session_state.module == "📈 Model Benchmarks & Validation":
    with st.container(border=True):
        st.markdown('<div class="amz-card-title">Multi-Stage Model Benchmarks & Validation Results</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1], gap="large")
        with col1:
            st.image(str(FIGURES_DIR / "model_benchmarks_stages.png"), use_container_width=True)

        with col2:
            st.markdown("#### Peer-Reviewed Model Performance")
            st.markdown("""
            | Stage | Input Modality | Evaluation Protocol | Primary Metric | Secondary Metric |
            | :--- | :--- | :--- | :--- | :--- |
            | **Stage A** | Photograph ($OD_G$) | LOOCV ($N=18$) | **$R^2 = 0.9224$** | RMSE = $0.065$ |
            | **Stage B** | Photograph (Multi-Color) | LOOCV ($N=10$) | **MAE = $15.34\%$** | $R^2 = 0.5487$ |
            | **Stage B (Gen)** | Cross-Floral Honeys | LOHO (H2 $\leftrightarrow$ H4) | **MAE = $26.18\%$** | Floral Variance |
            | **Stage C** | Benchtop $A_{520}$ + Proline | LOOCV ($N=10$) | **MAE = $15.31\%$** | $R^2 = 0.5603$ |
            | **Stage D** | Multimodal Fusion | LOOCV ($N=10$) | **MAE = $17.14\%$** | $R^2 = 0.4476$ |
            | **Stage E** | Binary Authenticity Clf | Training ($N=10$) | **Accuracy = $90\%$** | PASS / FAIL |
            """)

        st.markdown("#### Stage A: Spectrophotometer Emulator (Image -> A520)")
        st.dataframe(pd.DataFrame(benchmarks["stage_a"]["candidates"]).T)

        st.markdown("#### Stage B: Direct Adulteration Predictor (Image -> Adulteration %)")
        st.dataframe(pd.DataFrame(benchmarks["stage_b"]["candidates"]).T[["loocv_mae", "loocv_rmse", "loocv_r2", "loho_mae", "loho_rmse"]])


# ==============================================================================
# TAB 5: DATA EXPANSION & EXPERIMENTAL PROTOCOL
# ==============================================================================
elif st.session_state.module == "📋 Laboratory Protocol & Manifest":
    with st.container(border=True):
        st.markdown('<div class="amz-card-title">Laboratory Protocol & Master Experiment Manifest</div>', unsafe_allow_html=True)

        st.info("""
        **Data Expansion Protocol**: The current pilot set ($N=18$ tubes: 8 standards, 5 H2 dilutions, 5 H4 dilutions) 
        provides an **extremely strong baseline proof-of-concept** ($R^2 = 0.922$ for camera optical density vs benchtop $A_{520}$).
        To build a universal commercial model robust across all botanical origins, follow the standardized protocol below.
        """)

        st.markdown("""
        ### 1. Recommended Expanded Adulteration Panel
        Prepare standard dilutions using corn syrup, high fructose corn syrup (HFCS), and invert syrup:
        - **0% (Authentic Honey)**: Baseline reference
        - **5% & 10% (Low-level challenge)**: Realistic commercial adulteration
        - **20% & 25% (Moderate adulteration)**
        - **40% & 50% (High adulteration)**
        - **75% (Severe adulteration)**

        ### 2. Standardization of Photography
        - **Fixed Distance & Angle**: Keep phone/camera at 15–20 cm in a fixed 3D-printed or card tube holder.
        - **Uniform Backlight**: Place white diffusive paper or lightbox behind the tubes.
        - **Constant Exposure**: Lock exposure and white balance on the camera app.
        - **Always Include Blank Tube (0 µg/mL)**: Capturing the blank tube alongside sample tubes enables real-time auto-calibration of $OD_G = -\\log_{10}(G / G_0)$.

        ### 3. Master Manifest for Laboratory Data Entry
        When collecting new samples, save the tube photos in `data/raw/` and record their measured $A_{520}$ and prepared adulteration % in:
        
        📁 **Manifest Path:** `data/raw/experiment_manifest.csv`
        """)

        manifest_csv_path = RAW_DATA_DIR / "experiment_manifest.csv"
        if manifest_csv_path.exists():
            df_cur_manifest = pd.read_csv(manifest_csv_path)
            st.subheader("Current Master Manifest Records")
            st.dataframe(df_cur_manifest, hide_index=True)

            with open(manifest_csv_path, "r") as f:
                manifest_bytes = f.read()
            st.download_button(
                label="📥 Download Master Experiment Manifest (CSV)",
                data=manifest_bytes,
                file_name="experiment_manifest.csv",
                mime="text/csv"
            )

        st.markdown("""
        ### 4. Retraining the Models
        Once new photos are placed in `data/raw/` and entries added to `experiment_manifest.csv`, run:
        ```bash
        python main.py --build-data
        python main.py --train
        ```
        The pipeline will automatically re-extract features, recalibrate the standard curve, retrain all 5 stages, and refresh this platform!
        """)


# --- Working Anchor Back to Top Link ---
st.markdown("""
<a href="#top" target="_self" style="text-decoration: none; display: block; margin: 30px -4rem 0 -4rem;">
    <div class="gold-back-to-top">Back to top ↑</div>
</a>
<div class="amz-footer">
    <div style="margin-bottom: 8px;">
        <span style="color: #F3C044; font-weight: 700;">SpectraHoney™</span> • Botanical & Honey-Gold Phenotyping Engine
    </div>
    <div style="color: #A3CFBB;">
        Standardized according to Codex Alimentarius Stan 12-1981 & International Honey Commission (IHC) Harmonized Methods.<br>
        © 2026 SpectraHoney Chemometrics Initiative. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
