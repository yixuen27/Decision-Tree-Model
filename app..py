import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# =========================================================
# HEADER
# =========================================================
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### 📊 Decision Tree Decision Support System")

st.info("""
Predict factory productivity using operational inputs.  
**Model:** Decision Tree Classifier  
**Output:** Low / Moderate / High Productivity  
""")

form_is_invalid = False
st.divider()

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 🔧 Production Inputs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce")

    team = st.slider("Team Number", 1, 12, 1)
    workers = st.number_input("Number of Workers", value=30)

    if not (2 <= workers <= 90):
        st.error("Workers must be between 2 and 90")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", value=500)

with col2:
    st.subheader("⚙️ Operations")

    smv = st.number_input("SMV", value=22.0)
    style_change = st.selectbox("Style Changes", ["0", "1", "2"])

    incentive = st.number_input("Incentive", value=100)
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

# =========================================================
# FIXED INPUTS (Simplified for simulation)
# =========================================================
day = "Monday"
quarter = "Quarter1"
dept = "Sewing"
idle_time = 0
idle_men = 0

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_productivity(input_df):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    return pred, prob

# =========================================================
# SINGLE PREDICTION
# =========================================================
st.divider()
st.markdown("## 🔍 Current Prediction")

if not form_is_invalid:
    if st.button("Generate Prediction", use_container_width=True):

        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

        # Encoding
        def set_dummy(cat, val):
            col = f"{cat}_{val}"
            if col in model_columns:
                input_df[col] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        pred, prob = predict_productivity(input_df)

        labels = ['Low', 'Moderate', 'High']
        result = labels[pred]

        st.markdown(f"### 🎯 Result: **{result}**")

        if result == "Low":
            st.error(f"⚠️ High Risk (Confidence: {prob[0]:.2%})")
        elif result == "Moderate":
            st.warning(f"⚠️ متوسط (Confidence: {prob[1]:.2%})")
        else:
            st.success(f"✅ Excellent (Confidence: {prob[2]:.2%})")

# =========================================================
# TIME RANGE ANALYSIS (0 → 25950)
# =========================================================
st.divider()
st.markdown("## ⏱️ Productivity Trend Over Time (0 → 25,950)")

if st.button("📈 Run Time Simulation", use_container_width=True):

    time_range = np.linspace(0, 25950, 100)
    results = []

    for t in time_range:
        temp_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Simulate overtime trend based on time
        overtime_sim = (t / 25950) * 2 - 1  # scale to [-1, 1]

        temp_df['team'] = team
        temp_df['smv'] = smv
        temp_df['wip'] = wip
        temp_df['incentive'] = incentive
        temp_df['idle_time'] = idle_time
        temp_df['idle_men'] = idle_men
        temp_df['no_of_workers'] = workers
        temp_df['over_time_scaled'] = overtime_sim

        # Encoding
        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        temp_df = temp_df[model_columns]

        pred = model.predict(temp_df)[0]
        results.append(pred)

    # Convert to numeric labels
    results = np.array(results)

    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots()

    ax.plot(time_range, results)
    ax.set_title("Productivity Prediction Over Time")
    ax.set_xlabel("Time Range (0 → 25950)")
    ax.set_ylabel("Productivity Level")

    # Highlight LOW regions
    low_indices = np.where(results == 0)[0]
    ax.scatter(time_range[low_indices], results[low_indices])

    st.pyplot(fig)

    # =========================================================
    # LOW RESULT INSIGHT
    # =========================================================
    low_percentage = (results == 0).mean() * 100

    st.markdown("### ⚠️ Low Productivity Analysis")

    if low_percentage > 50:
        st.error(f"🚨 Critical: {low_percentage:.2f}% of time shows LOW productivity")
    elif low_percentage > 20:
        st.warning(f"⚠️ Warning: {low_percentage:.2f}% LOW productivity detected")
    else:
        st.success(f"✅ Only {low_percentage:.2f}% LOW productivity (Healthy)")
