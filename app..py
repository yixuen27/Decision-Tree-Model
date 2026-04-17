import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# =========================================================
# LOAD MODEL & FEATURES
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# =========================================================
# HEADER
# =========================================================
st.title("🧵 AI-Powered Garment Factory Productivity Predictor")
st.markdown("### 📊 Decision Tree Model Deployment System")

st.info("""
This intelligent system predicts **garment production productivity levels**  
based on operational and workforce inputs.

**Model:** Decision Tree Classifier  
**Output:** Low | Moderate | High Productivity
""")

# =========================================================
# FORM VALIDATION FLAG
# =========================================================
form_is_invalid = False

st.divider()

# =========================================================
# SECTION 1: CORE PRODUCTION FACTORS
# =========================================================
st.markdown("## 🔴 Core Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce Details")

    team = st.slider("Team Number", 1, 12, 1)

    workers = st.number_input("Number of Workers", min_value=1, max_value=120, value=30)
    if not (2 <= workers <= 90):
        st.error("⚠️ Workers must be between 2 and 90")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", min_value=0, value=500)
    if wip > 23122:
        st.error("⚠️ Maximum WIP is 23,122")
        form_is_invalid = True

with col2:
    st.subheader("⚙️ Production Complexity")

    smv = st.number_input("SMV (Standard Minute Value)", value=22.0)
    if not (2.9 <= smv <= 54.6):
        st.error("⚠️ SMV must be between 2.9 and 54.6")
        form_is_invalid = True

    style_change = st.selectbox(
        "Number of Style Changes",
        ["0", "1", "2"]
    )

# =========================================================
# SECTION 2: OPERATIONAL FACTORS
# =========================================================
st.divider()
st.markdown("## 🟡 Operational & Time Factors")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📅 Production Scheduling")

    day = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )

    quarter = st.selectbox(
        "Production Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )

    dept = st.selectbox(
        "Department",
        ["Sewing", "Finishing"]
    )

with col4:
    st.subheader("💰 Efficiency & Time")

    incentive = st.number_input("Incentive Amount", min_value=0, value=100)
    if incentive > 3600:
        st.error("⚠️ Maximum incentive is 3,600")
        form_is_invalid = True

    # ✅ UPDATED RANGE (0 → 25950)
    over_time = st.slider(
        "Over Time (Minutes)",
        min_value=0,
        max_value=25950,
        value=0,
        step=30
    )

    idle_time = st.number_input("Idle Time (Minutes)", min_value=0, value=0)
    if idle_time > 300:
        st.error("⚠️ Maximum idle time is 300 minutes")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers", min_value=0, value=0)
    if idle_men > 45:
        st.error("⚠️ Maximum idle workers is 45")
        form_is_invalid = True

# =========================================================
# PREDICTION SECTION
# =========================================================
st.divider()
st.markdown("## 🚀 Productivity Prediction")

if form_is_invalid:
    st.warning("⚠️ Please fix input errors before generating prediction.")
    st.button("Generate Prediction", disabled=True)

else:
    if st.button("🔍 Generate Productivity Forecast", use_container_width=True):

        # =====================================================
        # PREPARE INPUT DATAFRAME
        # =====================================================
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical Features
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers

        # IMPORTANT: If your model used scaled overtime, adjust here
        input_df['over_time_scaled'] = over_time  # Modify if scaling was applied during training

        # =====================================================
        # ENCODING FUNCTION
        # =====================================================
        def set_dummy(prefix, value):
            col_name = f"{prefix}_{value}"
            if col_name in input_df.columns:
                input_df[col_name] = 1

        # Apply Encoding
        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # Align columns
        input_df = input_df[model_columns]

        # =====================================================
        # MODEL PREDICTION
        # =====================================================
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        labels = ["Low", "Moderate", "High"]
        result = labels[prediction]

        # =====================================================
        # DISPLAY RESULTS
        # =====================================================
        st.markdown(f"### 🏷️ Predicted Productivity Level: **{result}**")

        if result == "High":
            st.success(f"Confidence: {probabilities[2]:.2%} — Excellent productivity expected.")
            st.balloons()

        elif result == "Moderate":
            st.warning(f"Confidence: {probabilities[1]:.2%} — Stable performance with room for improvement.")

        else:
            st.error(f"Confidence: {probabilities[0]:.2%} — High risk of low productivity.")

        # =====================================================
        # INPUT SUMMARY
        # =====================================================
        with st.expander("📋 View Processed Input Data"):
            st.dataframe(input_df)
