
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
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

# =========================================================
# LOAD MODEL & FEATURES
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()
class_values = list(getattr(model, "classes_", [0, 1, 2]))

# =========================================================
# HELPERS
# =========================================================
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
DISPLAY_ORDER = ["Low", "Moderate", "High"]

def get_probability_dict(probabilities, class_values):
    prob_dict = {LABEL_MAP.get(cls, str(cls)): float(prob) for cls, prob in zip(class_values, probabilities)}
    for label in DISPLAY_ORDER:
        prob_dict.setdefault(label, 0.0)
    return prob_dict

def productivity_note(label):
    if label == "High":
        return "Excellent productivity outlook based on the selected production conditions."
    if label == "Moderate":
        return "Stable productivity is expected, but operational adjustments may still improve output."
    return "There is a risk of underperformance, so closer production monitoring is recommended."

def build_input_dataframe():
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numerical columns expected by the saved model
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
# HEADER
# =========================================================
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
with m1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Deployment Status", "Active")
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Model Type", "Decision Tree")
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Prediction Output", "3 Levels")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="small-muted">Enter production conditions below to generate a class prediction and compare all class probabilities.</p>', unsafe_allow_html=True)
st.divider()

# =========================================================
# INPUTS
# =========================================================
form_is_invalid = False

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👥 Workforce & Production Load")

    no_of_workers = st.number_input("Number of Workers", min_value=0, max_value=100, value=30, step=1)
    if not (2 <= no_of_workers <= 89):
        st.error("Workers must be between 2 and 89 based on the training data.")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", min_value=0, max_value=3000, value=500, step=10)
    if not (0 <= wip <= 2698):
        st.error("WIP should stay within the observed data range of 0 to 2,698.")
        form_is_invalid = True

    smv = st.number_input("SMV (Standard Minute Value)", min_value=0.0, max_value=60.0, value=22.0, step=0.1)
    if not (2.9 <= smv <= 54.6):
        st.error("SMV must be between 2.9 and 54.6.")
        form_is_invalid = True

    no_of_style_change = st.selectbox("Number of Style Changes", [0, 1, 2], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Scheduling & Efficiency")

    quarter = st.selectbox("Production Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["sewing", "finished"], format_func=lambda x: x.title())
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    incentive = st.number_input("Incentive Amount", min_value=0, max_value=3600, value=100, step=10)
    over_time = st.slider("Over Time (Minutes)", min_value=0, max_value=25920, value=0, step=30)
    idle_time = st.number_input("Idle Time (Minutes)", min_value=0, max_value=300, value=0, step=1)
    idle_men = st.number_input("Idle Workers", min_value=0, max_value=45, value=0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader("🚀 Generate Prediction")

generate = st.button("Generate Productivity Forecast", use_container_width=True, type="primary", disabled=form_is_invalid)

if form_is_invalid:
    st.warning("Please correct the highlighted input values before generating the prediction.")

# =========================================================
# PREDICTION OUTPUT
# =========================================================
if generate:
    input_df = build_input_dataframe()
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = get_probability_dict(probabilities, class_values)
    predicted_label = LABEL_MAP.get(prediction, str(prediction))
    predicted_confidence = prob_dict.get(predicted_label, 0.0)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"### Predicted Productivity Level: **{predicted_label}**")
    st.markdown(productivity_note(predicted_label))

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Low Probability", f"{prob_dict['Low']:.2%}")
        st.progress(min(max(prob_dict["Low"], 0.0), 1.0))
    with r2:
        st.metric("Moderate Probability", f"{prob_dict['Moderate']:.2%}")
        st.progress(min(max(prob_dict["Moderate"], 0.0), 1.0))
    with r3:
        st.metric("High Probability", f"{prob_dict['High']:.2%}")
        st.progress(min(max(prob_dict["High"], 0.0), 1.0))

    if predicted_label == "High":
        st.success(f"Top predicted class confidence: {predicted_confidence:.2%}")
    elif predicted_label == "Moderate":
        st.warning(f"Top predicted class confidence: {predicted_confidence:.2%}")
    else:
        st.error(f"Top predicted class confidence: {predicted_confidence:.2%}")

    # Practical note for this saved model
    if prob_dict["Low"] == 0:
        st.info(
            "Note: this saved Decision Tree model may assign 0% to the Low class for many inputs. "
            "If Low never appears at all, the issue is likely in the trained model file rather than the interface."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View Processed Input Data"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("ℹ️ Interpretation Note"):
        st.write(
            "Decision Tree models can sometimes return repeated probability values for different inputs when those inputs "
            "fall into the same terminal leaf. That is normal behavior for the model. However, if one class never appears "
            "across many very different inputs, the trained model should be reviewed."
        )
