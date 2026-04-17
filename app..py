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

# Track validation
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
    
    # ✅ FIXED OVERTIME RANGE
    overtime = st.slider("Overtime (Minutes)", 0, 25920, 0)
    
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
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers

        # ✅ IMPORTANT FIX: use correct column name
        input_df['over_time'] = overtime

        # --- ENCODING ---
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        # --- PREDICTION ---
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # --- DISPLAY RESULT ---
        st.markdown(f"### 🏷️ Predicted Productivity Level: **{result}**")

        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%} — Excellent performance expected.")
            st.balloons()

        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%} — Stable but can be improved.")

        else:
            st.error(f"Confidence: {probs[0]:.2%} — Risk of low productivity.")

        # --- OPTIONAL: SHOW INPUT SUMMARY ---
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_df)
