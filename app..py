import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# 🔴 CORE CONFIGURATION (IMPORTANT)
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()
class_values = list(getattr(model, "classes_", [0, 1, 2]))

# =========================================================
# 🔴 CORE CONSTANTS (IMPORTANT)
# =========================================================
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
DISPLAY_ORDER = ["Low", "Moderate", "High"]

# =========================================================
# 🟡 HELPER FUNCTIONS (SUB-IMPORTANT)
# =========================================================
def get_probability_dict(probabilities, class_values):
    prob_dict = {LABEL_MAP.get(cls, str(cls)): float(prob)
                 for cls, prob in zip(class_values, probabilities)}
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
# 🟡 UI STYLING (SUB-IMPORTANT)
# =========================================================
st.markdown("""<style>
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1200px;}
.hero-card {background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
color:white; padding:1.4rem 1.6rem; border-radius:18px;}
.section-card {background:#ffffff; border-radius:16px; padding:1rem;}
.metric-card {background:#f8fafc; border-radius:16px; padding:1rem; text-align:center;}
.result-card {border-radius:18px; padding:1.1rem;}
.small-muted {color:#64748b; font-size:0.92rem;}
</style>""", unsafe_allow_html=True)

# =========================================================
# 🟡 HEADER UI (SUB-IMPORTANT)
# =========================================================
st.markdown("""
<div class="hero-card">
    <h1>🧵 AI-Powered Garment Factory Productivity Predictor</h1>
    <p>Decision-support system for forecasting production productivity.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 🟡 INPUT SECTION (SUB-IMPORTANT UI)
# =========================================================
form_is_invalid = False

left, right = st.columns(2)

with left:
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
    no_of_style_change = st.selectbox("Style Changes", [0, 1, 2])

with right:
    st.subheader("📅 Scheduling & Efficiency")

    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    department = st.selectbox("Department", ["sewing","finished"])
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])

    incentive = st.number_input("Incentive", 0, 3600, 100)
    over_time = st.slider("Overtime", 0, 25920, 0)
    idle_time = st.number_input("Idle Time", 0, 300, 0)
    idle_men = st.number_input("Idle Workers", 0, 45, 0)

# =========================================================
# 🔴 PREDICTION LOGIC (IMPORTANT)
# =========================================================
generate = st.button("Generate Productivity Forecast", disabled=form_is_invalid)

if generate:
    input_df = build_input_dataframe()

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    prob_dict = get_probability_dict(probabilities, class_values)
    predicted_label = LABEL_MAP.get(prediction, str(prediction))
    confidence = prob_dict[predicted_label]

    st.markdown(f"### Prediction: {predicted_label}")
    st.write(productivity_note(predicted_label))

    st.metric("Confidence", f"{confidence:.2%}")

    st.write("### Class Probabilities")
    st.write(prob_dict)
