import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Garment Productivity Predictor Dashboard")
st.caption("Decision Tree Model • Smart Factory Decision Support System")

# --- SIDEBAR (IMPORTANT DETAILS) ---
st.sidebar.header("📌 Important Details")

day = st.sidebar.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
quarter = st.sidebar.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
dept = st.sidebar.selectbox("Department", ["Sewing", "Finishing"])
team = st.sidebar.slider("Team Number", 1, 12, 1)

st.sidebar.markdown("---")
st.sidebar.info("These factors strongly influence productivity patterns.")

# --- MAIN INPUT AREA (SUB DETAILS) ---
st.subheader("⚙️ Operational Inputs")

col1, col2 = st.columns(2)

form_is_invalid = False

with col1:
    st.markdown("### 🔧 Resource Allocation")
    
    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("Max: 23,122")
        form_is_invalid = True

    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("Range: 2–90")
        form_is_invalid = True

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("Range: 2.9–54.6")
        form_is_invalid = True

with col2:
    st.markdown("### 💰 Incentives & Efficiency")
    
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("Max: 3,600")
        form_is_invalid = True

    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

    idle_time = st.number_input("Idle Time (Minutes)", value=0)
    if idle_time > 300:
        st.error("Max: 300")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("Max: 45")
        form_is_invalid = True

    style_change = st.selectbox("Style Changes", ["0", "1", "2"])

# --- PREDICTION BUTTON ---
st.divider()

center = st.columns([1,2,1])[1]

if form_is_invalid:
    st.warning("⚠️ Please fix input errors before prediction.")
    center.button("Predict Productivity", disabled=True)
else:
    if center.button("🚀 Predict Productivity", use_container_width=True):

        # --- DATA PREPARATION ---
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

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

        # --- RESULT SECTION ---
        st.divider()
        st.subheader("📊 Prediction Result")

        if result == "High":
            st.markdown("## 🟢 HIGH PRODUCTIVITY")
            st.success("The production line is operating efficiently with optimal resource usage.")
            confidence = probs[2]
            st.balloons()

        elif result == "Moderate":
            st.markdown("## 🟡 MODERATE PRODUCTIVITY")
            st.warning("Performance is stable, but there is room for improvement.")
            confidence = probs[1]

        else:
            st.markdown("## 🔴 LOW PRODUCTIVITY")
            st.error("High risk of underperformance. Immediate action may be required.")
            confidence = probs[0]

        # --- CONFIDENCE ---
        st.metric("Prediction Confidence", f"{confidence:.2%}")

        # --- VISUALIZATION ---
        st.subheader("📈 Prediction Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence Across Classes")

        st.pyplot(fig)
