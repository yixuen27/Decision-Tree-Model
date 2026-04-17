import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity AI",
    layout="wide",
    page_icon="🏭"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🏭 Garment Factory Productivity AI System")
st.markdown("### 📊 Decision Tree Prediction Dashboard")

st.success("Predict workforce productivity using real factory operational data")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("📘 About Model")
    st.write("""
    - Model: Decision Tree Classifier  
    - Dataset: Garment Employee Productivity  
    - Output: Productivity Tier (Low / Moderate / High)
    """)

    st.markdown("### 📊 Key Drivers")
    st.caption("""
    • Work in Progress (WIP)  
    • SMV (Task Complexity)  
    • Incentive & Overtime  
    • Idle Time & Workers  
    """)

# --- VALIDATION FLAG ---
form_is_invalid = False

# =========================
# INPUT SECTION
# =========================
st.divider()
st.markdown("## 🧾 Input Production Parameters")

col1, col2, col3 = st.columns(3)

# --- COLUMN 1 ---
with col1:
    st.subheader("👥 Workforce")

    team = st.slider("Team Number", 1, 12, 1)

    workers = st.number_input("Number of Workers", value=30)
    if workers < 2 or workers > 90:
        st.error("Workers must be between 2 and 90")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("Max idle workers = 45")
        form_is_invalid = True


# --- COLUMN 2 ---
with col2:
    st.subheader("⚙️ Production")

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv < 2.9 or smv > 55:
        st.error("SMV must be between 2.9 and 54.6")
        form_is_invalid = True

    wip = st.number_input("Work In Progress", value=500)
    if wip > 23122:
        st.error("Max WIP = 23122")
        form_is_invalid = True

    style_change = st.selectbox("Style Change", ["0", "1", "2"])


# --- COLUMN 3 ---
with col3:
    st.subheader("💰 Efficiency")

    incentive = st.number_input("Incentive", value=100)
    if incentive > 3600:
        st.error("Max incentive = 3600")
        form_is_invalid = True

    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

    idle_time = st.number_input("Idle Time", value=0)
    if idle_time > 300:
        st.error("Max idle time = 300")
        form_is_invalid = True


# --- CONTEXT VARIABLES ---
st.markdown("### 📅 Context Information")

col4, col5, col6 = st.columns(3)

with col4:
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])

with col5:
    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])

with col6:
    dept = st.selectbox("Department", ["Sewing","Finishing"])


# =========================
# PREDICTION
# =========================
st.divider()
st.markdown("## 🚀 Prediction Output")

if form_is_invalid:
    st.warning("Please fix input errors before prediction")
    st.button("Predict", disabled=True)

else:
    if st.button("🔍 Predict Productivity", use_container_width=True):

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

        # --- PREDICT ---
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ["Low", "Moderate", "High"]
        result = labels[pred]

        # =========================
        # RESULT DISPLAY
        # =========================
        st.markdown("### 🎯 Prediction Result")

        colA, colB, colC = st.columns(3)

        colA.metric("Low", f"{probs[0]:.2%}")
        colB.metric("Moderate", f"{probs[1]:.2%}")
        colC.metric("High", f"{probs[2]:.2%}")

        st.divider()

        if result == "High":
            st.success(f"✅ HIGH Productivity ({probs[2]:.2%})")
            st.balloons()

        elif result == "Moderate":
            st.warning(f"⚠️ MODERATE Productivity ({probs[1]:.2%})")

        else:
            st.error(f"❌ LOW Productivity ({probs[0]:.2%})")

        # =========================
        # INTERPRETATION
        # =========================
        with st.expander("📊 Model Interpretation"):
            st.write("""
            This prediction is based on Decision Tree rules learned from historical garment production data.

            Key influencing factors:
            - High WIP → reduces efficiency  
            - High idle time → lowers productivity  
            - Incentives → improve performance  
            - SMV → indicates task difficulty  

            Note:
            Decision Tree produces fixed probability values per leaf node,
            so similar inputs may return identical probabilities.
            """)

        # =========================
        # INPUT SUMMARY
        # =========================
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df)
