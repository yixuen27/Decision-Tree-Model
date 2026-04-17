import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### 📊 Decision Tree Model Deployment")

st.info("""
This system predicts **production productivity level** based on operational inputs.  
✔ Model: Decision Tree Classifier  
✔ Output: Low / Moderate / High Productivity
""")

form_is_invalid = False

# --- MAIN INPUT SECTIONS ---
st.divider()

# =========================
# SECTION 1: IMPORTANT DETAILS
# =========================
st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce & Workload")
    
    team = st.slider("Team Number", 1, 12, 1)
    
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Must be between 2 and 90")
        form_is_invalid = True
    
    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("⚠️ Max allowed is 23,122")
        form_is_invalid = True

with col2:
    st.subheader("⚙️ Production Complexity")
    
    smv = st.number_input("SMV (Standard Minute Value)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ Must be between 2.9 and 54.6")
        form_is_invalid = True

    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])

# =========================
# SECTION 2: SUPPORTING DETAILS
# =========================
st.divider()
st.markdown("## 🟡 Supporting Operational Details")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📅 Time & Department")
    
    day = st.selectbox("Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )
    
    quarter = st.selectbox("Production Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )
    
    dept = st.selectbox("Department", ["Sewing", "Finishing"])

with col4:
    st.subheader("💰 Incentives & Efficiency")
    
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Max is 3,600")
        form_is_invalid = True
    
    # ✅ CHANGED RANGE HERE
    overtime = st.slider("Overtime", 0, 25920, 0)
    
    idle_time = st.number_input("Idle Time (Minutes)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max is 300")
        form_is_invalid = True
    
    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("⚠️ Max is 45")
        form_is_invalid = True

# =========================
# PREDICTION SECTION
# =========================
st.divider()
st.markdown("## 🚀 Prediction Result")

if form_is_invalid:
    st.warning("⚠️ Please correct the highlighted errors before prediction.")
    st.button("Generate Productivity Forecast", disabled=True)

else:
    if st.button("🔍 Generate Productivity Forecast", use_container_width=True):

        # --- CREATE INPUT DATA ---
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical features
        if 'team' in input_df.columns:
            input_df['team'] = team

        if 'smv' in input_df.columns:
            input_df['smv'] = smv

        if 'wip' in input_df.columns:
            input_df['wip'] = wip

        if 'incentive' in input_df.columns:
            input_df['incentive'] = incentive

        if 'idle_time' in input_df.columns:
            input_df['idle_time'] = idle_time

        if 'idle_men' in input_df.columns:
            input_df['idle_men'] = idle_men

        if 'no_of_workers' in input_df.columns:
            input_df['no_of_workers'] = workers

        # handle both naming possibilities
        if 'over_time_scaled' in input_df.columns:
            input_df['over_time_scaled'] = overtime
        elif 'over_time' in input_df.columns:
            input_df['over_time'] = overtime

        # --- ENCODING ---
        def set_dummy(category, value):
            options = [
                f"{category}_{value}",
                f"{category}_{str(value).lower()}",
                f"{category}_{str(value).upper()}"
            ]
            for col in options:
                if col in model_columns:
                    input_df[col] = 1
                    break

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        # --- PREDICTION ---
        raw_pred = model.predict(input_df)[0]

        # ✅ FORCE OUTPUT LABELS
        label_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
        result = label_map.get(int(raw_pred), str(raw_pred))

        # probability (safe)
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(input_df)[0]
            except:
                probs = None

        # --- DISPLAY RESULT ---
        st.markdown(f"### 🏷️ Predicted Productivity Level: **{result}**")

        if probs is not None:
            confidence = float(np.max(probs))

            if result == 'High':
                st.success(f"Confidence: {confidence:.2%} — Excellent performance expected.")
                st.balloons()

            elif result == 'Moderate':
                st.warning(f"Confidence: {confidence:.2%} — Stable but can be improved.")

            else:
                st.error(f"Confidence: {confidence:.2%} — Risk of low productivity.")
        else:
            st.info("Prediction generated successfully.")

        # --- OPTIONAL: SHOW INPUT SUMMARY ---
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_df)
