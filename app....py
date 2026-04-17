import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'garment_dt_model.pkl')
    columns_path = os.path.join(BASE_DIR, 'garment_dt_columns.pkl')
    
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### 📊 Decision Tree Model Deployment")

st.info("""
Predict production productivity level:
✔ Low / Moderate / High
✔ Model: Decision Tree (Cleaned Features)
""")

form_is_invalid = False

# =========================
# INPUT SECTION
# =========================
st.divider()

st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Must be between 2 and 90")
        form_is_invalid = True
    
    wip = st.number_input("Work in Progress (WIP)", value=500)

with col2:
    smv = st.number_input("SMV", value=22.0)
    style_change = st.selectbox("Style Changes", ["0", "1", "2"])

# =========================
# SUPPORTING FEATURES
# =========================
st.divider()
st.markdown("## 🟡 Supporting Details")

col3, col4 = st.columns(2)

with col3:
    day = st.selectbox("Day",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )
    
    quarter = st.selectbox("Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )
    
    dept = st.selectbox("Department", ["Sewing", "Finishing"])

with col4:
    incentive = st.number_input("Incentive", value=100)
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
    idle_time = st.number_input("Idle Time", value=0)
    idle_men = st.number_input("Idle Workers", value=0)

# =========================
# PREDICTION
# =========================
st.divider()
st.markdown("## 🚀 Prediction Result")

if form_is_invalid:
    st.warning("⚠️ Fix input errors first.")
    st.button("Predict", disabled=True)

else:
    if st.button("🔍 Generate Prediction", use_container_width=True):

        # --- PREPROCESSING ---
        
        # 1. Round workers
        workers = int(round(workers))
        
        # 2. Handle WIP outlier
        if wip > 23122:
            wip = 0

        # --- CREATE INPUT ---
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical features (NO TEAM)
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

        # --- ENCODING ---
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # Align columns
        input_df = input_df[model_columns]

        # --- PREDICT ---
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # --- OUTPUT ---
        st.markdown(f"### 🏷️ Predicted Productivity: **{result}**")

        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%}")
            st.balloons()

        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%}")

        else:
            st.error(f"Confidence: {probs[0]:.2%}")

        # Debug view
        with st.expander("📋 Processed Input Data"):
            st.dataframe(input_df)
