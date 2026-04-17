import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Intelligence",
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
st.title("🧵 Garment Productivity Intelligence System")
st.markdown("### 📊 Decision Tree Analytics Dashboard")

st.info("""
Predict garment team productivity using a **Decision Tree model**.

**Output Classes:** Low | Moderate | High  
**Note:** Predictions are based on learned production patterns.
""")

form_invalid = False

st.divider()

# =============================
# 🔴 CORE FACTORS
# =============================
st.markdown("## 🔴 Core Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce")

    team = st.slider("Team Number", 1, 12, 1)

    workers = st.number_input("Number of Workers", value=30)
    if workers < 2 or workers > 90:
        st.error("⚠️ Workers must be between 2 and 90")
        form_invalid = True

    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("⚠️ Max idle workers is 45")
        form_invalid = True

with col2:
    st.subheader("⚙️ Workload")

    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv < 2.9 or smv > 55:
        st.error("⚠️ SMV must be between 2.9 and 54.6")
        form_invalid = True

    wip = st.number_input("Work In Progress (WIP)", value=500)
    if wip > 23122:
        st.error("⚠️ Max WIP is 23,122")
        form_invalid = True

    style_change = st.selectbox("Style Changes", ["0", "1", "2"])

# =============================
# 🟡 SUPPORTING FACTORS
# =============================
st.divider()
st.markdown("## 🟡 Supporting Factors")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📅 Production Context")

    day = st.selectbox("Day",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )

    quarter = st.selectbox("Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )

    dept = st.selectbox("Department", ["Sewing", "Finishing"])

with col4:
    st.subheader("💰 Efficiency")

    incentive = st.number_input("Incentive", value=100)
    if incentive > 3600:
        st.error("⚠️ Max incentive is 3600")
        form_invalid = True

    overtime = st.slider("Overtime (minutes)", 0, 25920, 0)

    idle_time = st.number_input("Idle Time (minutes)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max idle time is 300")
        form_invalid = True

# =============================
# 🚀 PREDICTION
# =============================
st.divider()
st.markdown("## 🚀 Prediction Dashboard")

if form_invalid:
    st.warning("Please correct errors before prediction.")
    st.button("Predict", disabled=True)

else:
    if st.button("🔍 Predict Productivity", use_container_width=True):

        # --- INPUT DATA ---
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
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
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        # ✅ FIXED LABEL MAPPING
        class_labels = list(model.classes_)
        result = class_labels[pred]

        # =============================
        # 🎯 RESULT DISPLAY
        # =============================
        st.markdown(f"### 🏷️ Predicted Productivity: **{result}**")

        if result == "High":
            st.success(f"High Productivity ({max(probs):.2%})")
            st.balloons()

        elif result == "Moderate":
            st.warning(f"Moderate Productivity ({max(probs):.2%})")

        else:
            st.error(f"Low Productivity ({max(probs):.2%})")

        # =============================
        # 📊 PROBABILITY BREAKDOWN
        # =============================
        st.subheader("📊 Probability Breakdown")

        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probs
        })

        st.bar_chart(prob_df.set_index("Class"))

        # =============================
        # 🧠 SMART INSIGHT
        # =============================
        st.subheader("🧠 Model Insight")

        if probs[class_labels.index("Low")] > 0.3:
            st.warning("⚠️ Notice: There is a significant probability of LOW productivity.")

        elif probs[class_labels.index("High")] > 0.7:
            st.success("✅ Strong confidence in HIGH productivity.")

        else:
            st.info("ℹ️ Prediction is moderately confident.")

        # =============================
        # 📋 INPUT SUMMARY
        # =============================
        with st.expander("📋 View Input Data"):
            st.dataframe(input_df)

        # =============================
        # ℹ️ NOTE
        # =============================
        st.caption("""
        Decision Tree predictions are based on leaf node distributions. 
        Similar inputs may produce identical probability values.
        """)
