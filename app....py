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

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🧵 Garment Productivity Predictor")
st.markdown("### 📊 Decision Tree Model for Factory Performance Forecasting")
st.info("This tool predicts whether production will be **Low, Moderate, or High** based on operational inputs.")

st.divider()

form_is_invalid = False

# ==============================
# 🔹 IMPORTANT INPUTS
# ==============================
st.markdown("## 🔑 Important Production Inputs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Time & Department")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.markdown("### ⚙️ Workforce & Workload")
    
    wip = st.number_input("Work in Progress (wip)", value=500)
    if wip > 23122:
        st.error("⚠️ Max wip is 23,122")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Range is 2 to 90")
        form_is_invalid = True

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ Range is 2.9 to 54.6")
        form_is_invalid = True

st.divider()

# ==============================
# 🔹 SUPPORTING INPUTS
# ==============================
st.markdown("## ⚙️ Supporting Operational Details")

col3, col4 = st.columns(2)

with col3:
    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

with col4:
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Max Incentive is 3,600")
        form_is_invalid = True
        
    idle_time = st.number_input("Idle Time (Minutes)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max idle time is 300")
        form_is_invalid = True
        
    idle_men = st.number_input("Idle Workers Count", value=0)
    if idle_men > 45:
        st.error("⚠️ Max idle workers is 45")
        form_is_invalid = True

st.divider()

# ==============================
# 🔹 PREDICTION
# ==============================
st.markdown("## 🚀 Prediction Result")

if form_is_invalid:
    st.warning("Please fix input errors before prediction.")
    st.button("Generate Productivity Forecast", disabled=True)
else:
    if st.button("🔮 Generate Productivity Forecast", use_container_width=True):

        # Create dataframe
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical inputs
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

        # Encoding
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        # Prediction
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # ==============================
        # 🎯 RESULT DISPLAY (BIG + CLEAR)
        # ==============================
        st.markdown("---")

        if result == 'High':
            st.success("### 🟢 HIGH PRODUCTIVITY")
            st.markdown(f"## ✅ Confidence: **{probs[2]:.2%}**")
            st.balloons()
            st.markdown("Production is running at an **optimized level**. Resources are well utilized.")

        elif result == 'Moderate':
            st.warning("### 🟡 MODERATE PRODUCTIVITY")
            st.markdown(f"## ⚖️ Confidence: **{probs[1]:.2%}**")
            st.markdown("Production is **stable but has room for improvement**.")

        else:
            st.error("### 🔴 LOW PRODUCTIVITY")
            st.markdown(f"## ⚠️ Confidence: **{probs[0]:.2%}**")
            st.markdown("There is a **risk of not meeting targets**. Consider adjusting workforce or workload.")

        # Optional: Show probability breakdown
        st.markdown("### 📊 Prediction Breakdown")
        prob_df = pd.DataFrame({
            "Category": labels,
            "Probability": probs
        })
        st.bar_chart(prob_df.set_index("Category"))