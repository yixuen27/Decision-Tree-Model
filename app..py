import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, "garment_dt_model.pkl"))
    model_columns = joblib.load(os.path.join(BASE_DIR, "garment_dt_columns.pkl"))
    return model, model_columns

model, model_columns = load_assets()

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

df = load_dataset()

# =========================================================
# DATA OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique())
department_options = sorted(df["department"].dropna().unique())
day_options = sorted(df["day"].dropna().unique())
style_change_options = sorted(df["no_of_style_change"].dropna().unique())

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def set_dummy_value(input_df, prefix, value):
    candidates = [
        f"{prefix}_{value}",
        f"{prefix}_{str(value).lower()}",
        f"{prefix}_{str(value).upper()}",
        f"{prefix}_{str(value).capitalize()}",
    ]
    for col in candidates:
        if col in input_df.columns:
            input_df[col] = 1
            return

def normalize_prediction(pred):
    if isinstance(pred, str):
        return pred
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(int(pred), str(pred))

def get_result_message(result):
    if result == "High":
        return "success", "Excellent production performance is expected."
    elif result == "Moderate":
        return "warning", "Production is stable but can be improved."
    else:
        return "error", "There is a risk of low productivity."

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support prototype using Decision Tree.</div>',
    unsafe_allow_html=True
)

st.success("✅ System Status: Decision Tree model loaded successfully.")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📌 Prototype Overview")
    st.write("""
This prototype predicts garment factory productivity into three classes:

- **Low**
- **Moderate**
- **High**
""")

    st.markdown("---")
    st.subheader("🌳 Why Decision Tree?")
    st.write("""
- Easy to interpret  
- Fast prediction  
- Works well for classification tasks  
- Suitable for structured production data  
""")

# =========================================================
# INPUT AREA
# =========================================================
st.markdown("## 📥 Production Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Context")
    quarter = st.selectbox("Quarter", quarter_options)
    department = st.selectbox("Department", department_options)
    day = st.selectbox("Day of the Week", day_options)

with col2:
    st.subheader("⚙️ Production Factors")
    smv = st.number_input("SMV", float(df["smv"].min()), float(df["smv"].max()), float(df["smv"].median()))
    wip = st.number_input("WIP", int(df["wip"].min()), int(df["wip"].max()), int(df["wip"].median()))
    no_of_style_change = st.selectbox("Style Changes", style_change_options)
    no_of_workers = st.number_input("Workers", int(df["no_of_workers"].min()), int(df["no_of_workers"].max()), int(df["no_of_workers"].median()))

with col3:
    st.subheader("💰 Efficiency Metrics")
    over_time = st.number_input("Overtime", 0, 25920, int(df["over_time"].median()))
    incentive = st.number_input("Incentive", int(df["incentive"].min()), int(df["incentive"].max()), int(df["incentive"].median()))
    idle_time = st.number_input("Idle Time", int(df["idle_time"].min()), int(df["idle_time"].max()), int(df["idle_time"].median()))
    idle_men = st.number_input("Idle Workers", int(df["idle_men"].min()), int(df["idle_men"].max()), int(df["idle_men"].median()))

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):

    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # numeric
    for col, val in {
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": no_of_workers
    }.items():
        if col in input_df.columns:
            input_df[col] = val

    # encoding
    set_dummy_value(input_df, "quarter", quarter)
    set_dummy_value(input_df, "department", department)
    set_dummy_value(input_df, "day", day)
    set_dummy_value(input_df, "no_of_style_change", no_of_style_change)

    input_df = input_df[model_columns]

    # predict
    raw_pred = model.predict(input_df)[0]
    result = normalize_prediction(raw_pred)

    # probability
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)[0]
        except:
            pass

    st.markdown("## 📊 Prediction Results")

    if probs is not None:
        confidence = float(np.max(probs))
        st.metric("Predicted Productivity", result)
        st.metric("Confidence", f"{confidence:.2%}")
    else:
        st.metric("Predicted Productivity", result)

    status_type, msg = get_result_message(result)

    if status_type == "success":
        st.success(msg)
    elif status_type == "warning":
        st.warning(msg)
    else:
        st.error(msg)

    # debug (important)
    with st.expander("🔍 Debug Info"):
        st.write("Raw prediction:", raw_pred)
        st.write("Input data:", input_df)
