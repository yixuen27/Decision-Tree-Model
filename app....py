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

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #2E86C1;
}
.card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.small-text {
    font-size: 14px;
    color: grey;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- TITLE ---
st.markdown('<p class="main-title">🧵 Garment Productivity Predictor</p>', unsafe_allow_html=True)
st.caption("Decision Tree Model • Professional Prediction Dashboard")

# --- INPUT SECTION ---
form_is_invalid = False

col1, col2 = st.columns(2)

# =========================
# 🔵 IMPORTANT DETAILS
# =========================
with col1:
    st.markdown("### 🔵 Important Details")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

    workers = st.number_input("Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("Range: 2–90")
        form_is_invalid = True

    smv = st.number_input("SMV (Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("Range: 2.9–54.6")
        form_is_invalid = True

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🟡 SUB DETAILS
# =========================
with col2:
    st.markdown("### 🟡 Sub Details")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("Max: 23,122")
        form_is_invalid = True

    incentive = st.number_input("Incentive", value=100)
    if incentive > 3600:
        st.error("Max: 3,600")
        form_is_invalid = True

    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

    idle_time = st.number_input("Idle Time", value=0)
    if idle_time > 300:
        st.error("Max: 300")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("Max: 45")
        form_is_invalid = True

    style_change = st.selectbox("Style Change", ["0", "1", "2"])

    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION BUTTON ---
st.divider()
center = st.columns([1,2,1])[1]

if form_is_invalid:
    st.warning("⚠️ Please correct input errors before prediction.")
    center.button("Predict", disabled=True)
else:
    if center.button("🚀 Generate Productivity Prediction", use_container_width=True):

        # --- PREPARE DATA ---
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

        # --- PREDICT ---
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # =========================
        # 🎯 RESULT SECTION
        # =========================
        st.divider()

        if result == "High":
            st.markdown('<div class="result-box" style="background-color:#d4edda;color:#155724;">✅ HIGH PRODUCTIVITY</div>', unsafe_allow_html=True)
            st.write("💡 **Insight:** Production is highly efficient. Maintain current strategy.")
        elif result == "Moderate":
            st.markdown('<div class="result-box" style="background-color:#fff3cd;color:#856404;">⚠️ MODERATE PRODUCTIVITY</div>', unsafe_allow_html=True)
            st.write("💡 **Insight:** Performance is stable but can be improved.")
        else:
            st.markdown('<div class="result-box" style="background-color:#f8d7da;color:#721c24;">❌ LOW PRODUCTIVITY</div>', unsafe_allow_html=True)
            st.write("💡 **Insight:** Risk of underperformance. Immediate action recommended.")

        # --- CONFIDENCE ---
        st.metric("Prediction Confidence", f"{max(probs):.2%}")

        # =========================
        # 📊 VISUALIZATION
        # =========================
        st.subheader("📊 Prediction Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence by Class")

        st.pyplot(fig)
