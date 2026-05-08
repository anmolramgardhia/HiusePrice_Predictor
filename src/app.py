"""
app.py — Streamlit web application for house price prediction
House Price Prediction Project | Ames Housing Dataset

Run with:
    streamlit run src/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from train import load_model, APP_FEATURES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("House Price Predictor")
st.caption("Ames, Iowa — Ames Housing Dataset (2006–2010)")
st.markdown("Enter the property details below to get an estimated sale price.")

# ── Load model and baseline data ─────────────────────────────────────────────
@st.cache_resource
def get_model_and_baseline():
    from data import load_train, split_features_target
    from features import full_preprocess
    try:
        loaded_model = load_model()
        
        # Load baseline data to satisfy pipeline feature requirements
        train_df = load_train()
        X, _ = split_features_target(train_df)
        X_proc, _ = full_preprocess(X)
        
        baseline = X_proc.mode().iloc[0:1].copy()
        # Use median for numeric columns
        for col in X_proc.select_dtypes(include='number').columns:
            baseline[col] = X_proc[col].median()
            
        return loaded_model, baseline
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None

model, baseline_df = get_model_and_baseline()

if model is None:
    st.error("Model not found. Please run `python src/train.py` first to train and save the model.")
    st.stop()

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Property details")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input(
        "Above-ground living area (sq ft)",
        min_value=300, max_value=6000, value=1500, step=50
    )
    overall_qual = st.slider(
        "Overall quality (1 = poor, 10 = excellent)", 1, 10, 6
    )
    total_bsmt_sf = st.number_input(
        "Total basement area (sq ft)",
        min_value=0, max_value=3000, value=800, step=50
    )
    garage_cars = st.selectbox(
        "Garage capacity (cars)", options=[0, 1, 2, 3, 4], index=2
    )
    year_built = st.number_input(
        "Year built", min_value=1872, max_value=2010, value=1990, step=1
    )

with col2:
    neighborhood = st.selectbox(
        "Neighborhood",
        options=[
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
            "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW",
            "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
            "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV",
            "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste",
        ],
        index=0
    )
    bldg_type = st.selectbox(
        "Building type",
        options=["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
        index=0
    )
    full_bath = st.selectbox(
        "Full bathrooms above grade", options=[0, 1, 2, 3, 4], index=2
    )
    bedroom_abv_gr = st.selectbox(
        "Bedrooms above grade", options=[0, 1, 2, 3, 4, 5, 6], index=3
    )
    kitchen_qual = st.selectbox(
        "Kitchen quality",
        options=["Ex", "Gd", "TA", "Fa", "Po"],
        index=1,
        format_func=lambda x: {"Ex": "Excellent", "Gd": "Good", "TA": "Typical/Average", "Fa": "Fair", "Po": "Poor"}[x]
    )

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("Estimate price", type="primary"):
    # Start with baseline to satisfy the pipeline's 79 feature requirement
    input_data = baseline_df.copy()
    
    # Map the selected 10 features into the baseline
    input_data["Gr Liv Area"]   = gr_liv_area
    input_data["Overall Qual"]  = overall_qual
    input_data["Total Bsmt SF"] = total_bsmt_sf
    input_data["Garage Cars"]   = garage_cars
    input_data["Year Built"]    = year_built
    input_data["Neighborhood"]  = neighborhood
    input_data["Bldg Type"]     = bldg_type
    input_data["Full Bath"]     = full_bath
    input_data["Bedroom AbvGr"] = bedroom_abv_gr
    input_data["Kitchen Qual"]  = kitchen_qual

    try:
        log_pred = model.predict(input_data)[0]
        price = np.expm1(log_pred)

        # Confidence range: ±1 log-RMSE unit (~0.12) converted back
        low  = np.expm1(log_pred - 0.12)
        high = np.expm1(log_pred + 0.12)

        st.success(f"### Estimated sale price: ${price:,.0f}")
        st.info(f"Confidence range: ${low:,.0f} — ${high:,.0f}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Estimate",        f"${price:,.0f}")
        col_b.metric("Low (–1 RMSE)",   f"${low:,.0f}")
        col_c.metric("High (+1 RMSE)",  f"${high:,.0f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("The full pipeline requires all 79 features. This demo uses the top 10 — run train.py on the complete dataset for full accuracy.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost trained on the Ames Housing Dataset (De Cock, 2011). "
    "Predictions are estimates only and do not constitute a formal appraisal."
)
