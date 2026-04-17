# =========================================================
# 📦 IMPORT LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# ⚙️ PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)


# =========================================================
# 🎨 CUSTOM UI STYLING
# =========================================================
def apply_custom_styles():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1rem 0.4rem 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }
    .result-card {
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    .stButton > button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()


# =========================================================
# 🤖 LOAD MODEL & FEATURES
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()
class_values = list(getattr(model, "classes_", [0, 1, 2]))


# =========================================================
# 🧠 HELPER FUNCTIONS
# =========================================================
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
DISPLAY_ORDER = ["Low", "Moderate", "High"]


def get_probability_dict(probabilities, class_values):
    prob_dict = {
        LABEL_MAP.get(cls, str(cls)): float(prob)
        for cls, prob in zip(class_values, probabilities)
    }
    for label in DISPLAY_ORDER:
        prob_dict.setdefault(label, 0.0)
    return prob_dict


def productivity_note(label):
    if label == "High":
        return "Excellent productivity outlook based on the selected production conditions."
    elif label == "Moderate":
        return "Stable productivity is expected, but operational adjustments may still improve output."
    return "There is a risk of underperformance, so closer production monitoring is recommended."


def build_input_dataframe():
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    numeric_values = {
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": no_of_workers,
    }

    for col, val in numeric_values.items():
        if col in input_df.columns:
            input_df[col] = val

    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    set_dummy("quarter", quarter)
    set_dummy("department", department)
    set_dummy("day", day)
    set_dummy("no_of_style_change", no_of_style_change)

    return input_df[model_columns]


# =========================================================
# 🏠 HEADER SECTION
# =========================================================
def render_header():
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin:0; font-size:2rem;">🧵 AI-Powered Garment Factory Productivity Predictor</h1>
        <p style="margin:0.55rem 0 0.2rem 0; font-size:1.02rem;">
            A professional decision-support prototype for forecasting garment production productivity
            based on workforce, operational, and scheduling inputs.
        </p>
        <p style="margin:0.35rem 0 0 0; opacity:0.9;">
            <b>Model:</b> Decision Tree Classifier &nbsp; | &nbsp;
            <b>Output Classes:</b> Low, Moderate, High
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, label, value in zip(
        [m1, m2, m3],
        ["Deployment Status", "Model Type", "Prediction Output"],
        ["Active", "Decision Tree", "3 Levels"]
    ):
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown('</div>', unsafe_allow_html=True)

render_header()

st.markdown('<p class="small-muted">Enter production conditions below to generate a prediction.</p>', unsafe_allow_html=True)
st.divider()


# =========================================================
# 🧾 INPUT SECTION
# =========================================================
form_is_invalid = False

def render_inputs():
    global form_is_invalid
    global no_of_workers, wip, smv, no_of_style_change
    global quarter, department, day
    global incentive, over_time, idle_time, idle_men

    left, right = st.columns(2)

    # LEFT PANEL
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("👥 Workforce & Production Load")

        no_of_workers = st.number_input("Number of Workers", 0, 100, 30)
        if not (2 <= no_of_workers <= 89):
            st.error("Workers must be between 2 and 89.")
            form_is_invalid = True

        wip = st.number_input("WIP", 0, 3000, 500)
        if not (0 <= wip <= 2698):
            st.error("WIP out of range.")
            form_is_invalid = True

        smv = st.number_input("SMV", 0.0, 60.0, 22.0)
        if not (2.9 <= smv <= 54.6):
            st.error("SMV out of range.")
            form_is_invalid = True

        no_of_style_change = st.selectbox("Style Changes", [0, 1, 2])
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT PANEL
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📅 Scheduling & Efficiency")

        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        department = st.selectbox("Department", ["sewing", "finished"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

        incentive = st.number_input("Incentive", 0, 3600, 100)
        over_time = st.slider("Over Time", 0, 25920, 0)
        idle_time = st.number_input("Idle Time", 0, 300, 0)
        idle_men = st.number_input("Idle Workers", 0, 45, 0)
        st.markdown('</div>', unsafe_allow_html=True)

render_inputs()

st.divider()


# =========================================================
# 🚀 PREDICTION SECTION
# =========================================================
st.subheader("🚀 Generate Prediction")

generate = st.button(
    "Generate Productivity Forecast",
    use_container_width=True,
    type="primary",
    disabled=form_is_invalid
)

if form_is_invalid:
    st.warning("Please correct inputs before proceeding.")


# =========================================================
# 📊 OUTPUT SECTION
# =========================================================
if generate:
    input_df = build_input_dataframe()
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    prob_dict = get_probability_dict(probabilities, class_values)
    predicted_label = LABEL_MAP.get(prediction, str(prediction))

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"### Predicted Productivity Level: **{predicted_label}**")
    st.markdown(productivity_note(predicted_label))

    cols = st.columns(3)
    for col, label in zip(cols, DISPLAY_ORDER):
        with col:
            st.metric(f"{label} Probability", f"{prob_dict[label]:.2%}")
            st.progress(prob_dict[label])

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 Input Data"):
        st.dataframe(input_df)

    with st.expander("ℹ️ Interpretation"):
        st.write("Decision Tree may return same probabilities for similar leaf nodes.")
