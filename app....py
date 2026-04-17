import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG ---
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
st.markdown("### 📊 Decision Tree Model for Production Performance")
st.info("This system predicts whether a production team will achieve **Low, Moderate, or High productivity** based on operational inputs.")

form_is_invalid = False

# =========================
# 🔵 IMPORTANT INPUTS
# =========================
st.markdown("## 🔵 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👥 Workforce & Production Load")
    
    team = st.slider("Team Number", 1, 12, 1)
    
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Workers must be between 2 and 90")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("⚠️ Max WIP is 23,122")
        form_is_invalid = True

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ SMV must be between 2.9 and 54.6")
        form_is_invalid = True

with col2:
    st.markdown("### 💰 Incentives & Efficiency")

    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Max Incentive is 3,600")
        form_is_invalid = True

    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

    idle_time = st.number_input("Idle Time (Minutes)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max Idle Time is 300")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("⚠️ Max Idle Workers is 45")
        form_is_invalid = True


# =========================
# 🟢 SUB DETAILS
# =========================
st.markdown("## 🟢 Additional Context")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 📅 Time Factors")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])

with col4:
    st.markdown("### 🏭 Production Setup")
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])


# =========================
# 🔮 PREDICTION SECTION
# =========================
st.divider()
st.markdown("## 🔮 Productivity Prediction")

if form_is_invalid:
    st.warning("⚠️ Please correct the highlighted errors before proceeding.")
    st.button("Generate Prediction", disabled=True)
else:
    if st.button("🚀 Generate Productivity Forecast", use_container_width=True):

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

        # =========================
        # 🎯 RESULT DISPLAY (IMPROVED)
        # =========================
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        if result == 'High':
            st.markdown(f"# 🟢 **HIGH PRODUCTIVITY**")
            st.success(f"Confidence Level: **{probs[2]:.2%}**")
            st.markdown("📈 *Production is optimized. Team is performing efficiently with strong output.*")
            st.balloons()

        elif result == 'Moderate':
            st.markdown(f"# 🟡 **MODERATE PRODUCTIVITY**")
            st.warning(f"Confidence Level: **{probs[1]:.2%}**")
            st.markdown("⚖️ *Performance is stable but has room for improvement. Monitor key factors like WIP and idle time.*")

        else:
            st.markdown(f"# 🔴 **LOW PRODUCTIVITY**")
            st.error(f"Confidence Level: **{probs[0]:.2%}**")
            st.markdown("⚠️ *High risk of underperformance. Consider reducing idle time, balancing workload, or increasing incentives.*")

        # =========================
        # 📊 PROBABILITY BREAKDOWN
        # =========================
        st.markdown("### 📊 Prediction Breakdown")
        prob_df = pd.DataFrame({
            'Category': ['Low', 'Moderate', 'High'],
            'Probability': probs
        })

        st.bar_chart(prob_df.set_index('Category'))
