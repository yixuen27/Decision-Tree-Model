import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### 📊 Decision Tree Model")

st.info("""
Predict productivity level of garment factory teams based on operational inputs.

✔ Output Classes:
- 🔴 Low
- 🟡 Moderate
- 🟢 High
""")

form_is_invalid = False

st.divider()

# =========================
# 🔴 IMPORTANT FACTORS
# =========================
st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    team = st.slider("👥 Team Number", 1, 12, 1)
    
    workers = st.number_input("👷 Number of Workers", 30)
    if workers < 2 or workers > 90:
        st.error("Workers must be between 2 and 90")
        form_is_invalid = True

    wip = st.number_input("📦 Work in Progress (WIP)", 500)
    if wip > 23122:
        st.error("Max WIP is 23,122")
        form_is_invalid = True

with col2:
    smv = st.number_input("⚙️ SMV (Task Complexity)", 22.0)
    if smv < 2.9 or smv > 55:
        st.error("SMV must be between 2.9 and 54.6")
        form_is_invalid = True

    style_change = st.selectbox("🎨 Style Changes", ["0", "1", "2"])

# =========================
# 🟡 SUPPORTING FACTORS
# =========================
st.divider()
st.markdown("## 🟡 Supporting Factors")

col3, col4 = st.columns(2)

with col3:
    day = st.selectbox("📅 Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])
    quarter = st.selectbox("📆 Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    dept = st.selectbox("🏭 Department", ["Sewing","Finishing"])

with col4:
    incentive = st.number_input("💰 Incentive", 100)
    if incentive > 3600:
        st.error("Max incentive is 3600")
        form_is_invalid = True

    # ✅ UPDATED RANGE
    overtime = st.slider("⏱️ Overtime (minutes)", 0, 25920, 0)

    idle_time = st.number_input("⏳ Idle Time", 0)
    if idle_time > 300:
        st.error("Max idle time is 300")
        form_is_invalid = True

    idle_men = st.number_input("🧍 Idle Workers", 0)
    if idle_men > 45:
        st.error("Max idle workers is 45")
        form_is_invalid = True

# =========================
# 🔮 PREDICTION
# =========================
st.divider()
st.markdown("## 🚀 Prediction")

if form_is_invalid:
    st.warning("Please fix errors before prediction.")
    st.button("Predict", disabled=True)
else:
    if st.button("🔍 Generate Prediction", use_container_width=True):

        # Create dataframe
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time'] = overtime   # ✅ IMPORTANT FIX

        # Encoding
        def set_dummy(category, value):
            col = f"{category}_{value}"
            if col in model_columns:
                input_df[col] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        # Predict
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[pred]

        # =========================
        # RESULT DISPLAY
        # =========================
        st.markdown(f"### 🏷️ Prediction: **{result}**")

        if result == "High":
            st.success(f"Confidence: {probs[2]:.2%}")
            st.balloons()
        elif result == "Moderate":
            st.warning(f"Confidence: {probs[1]:.2%}")
        else:
            st.error(f"Confidence: {probs[0]:.2%}")

        # =========================
        # 📊 PROBABILITY CHART
        # =========================
        st.markdown("### 📊 Class Probabilities")

        prob_df = pd.DataFrame({
            "Productivity Level": ["Low", "Moderate", "High"],
            "Probability": probs
        })

        st.bar_chart(prob_df.set_index("Productivity Level"))

        # =========================
        # 📋 INPUT SUMMARY
        # =========================
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df)
