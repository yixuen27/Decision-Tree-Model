import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --- CUSTOM STYLE ---
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #2E86C1;
}
.card {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}
.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.small-text {
    font-size: 16px;
    color: gray;
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
st.markdown('<p class="big-title">🧵 Garment Productivity Predictor</p>', unsafe_allow_html=True)
st.caption("Decision Tree Model • Factory Decision Support System")

# --- INPUT VALIDATION FLAG ---
form_is_invalid = False

# --- LAYOUT ---
col1, col2, col3 = st.columns(3)

# =========================
# 📅 IMPORTANT DETAILS
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📅 Important Details")

    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Production Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ⚙️ RESOURCE DETAILS
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Resource Details")

    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("Max: 23,122")
        form_is_invalid = True

    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("Range: 2–90")
        form_is_invalid = True

    style_change = st.selectbox("Style Change", ["0", "1", "2"])

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("Range: 2.9–54.6")
        form_is_invalid = True

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 💰 PERFORMANCE DETAILS
# =========================
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Performance Details")

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

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🚀 PREDICTION BUTTON
# =========================
st.divider()

center_col = st.columns([1,2,1])[1]

if form_is_invalid:
    st.warning("⚠️ Please fix input errors before prediction.")
    center_col.button("Generate Prediction", disabled=True)
else:
    if center_col.button("🚀 Generate Productivity Forecast", use_container_width=True):

        # --- CREATE INPUT ---
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

        st.divider()

        # =========================
        # 🎯 RESULT DISPLAY (IMPROVED)
        # =========================
        if result == "High":
            st.markdown(
                f'<div class="result-box" style="background:#d4edda;color:#155724;">✅ HIGH PRODUCTIVITY</div>',
                unsafe_allow_html=True
            )
            st.metric("Confidence Level", f"{probs[2]:.2%}")
            st.markdown('<p class="small-text">Factory is performing optimally. Maintain current strategy.</p>', unsafe_allow_html=True)
            st.balloons()

        elif result == "Moderate":
            st.markdown(
                f'<div class="result-box" style="background:#fff3cd;color:#856404;">⚠️ MODERATE PRODUCTIVITY</div>',
                unsafe_allow_html=True
            )
            st.metric("Confidence Level", f"{probs[1]:.2%}")
            st.markdown('<p class="small-text">Performance is stable but can be improved with better resource allocation.</p>', unsafe_allow_html=True)

        else:
            st.markdown(
                f'<div class="result-box" style="background:#f8d7da;color:#721c24;">❌ LOW PRODUCTIVITY</div>',
                unsafe_allow_html=True
            )
            st.metric("Confidence Level", f"{probs[0]:.2%}")
            st.markdown('<p class="small-text">High risk of underperformance. Review workforce and production conditions.</p>', unsafe_allow_html=True)
