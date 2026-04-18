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
# 🎨 CUSTOM STYLING (FROM DESIGN 1)
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
    margin-bottom: 1rem;
}
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
.metric-card {
    background: #f8fafc;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}
.result-card {
    border-radius: 18px;
    padding: 1.2rem;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 🤖 LOAD MODEL
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model (1).pkl")
    model_columns = joblib.load("garment_dt_columns (1).pkl")
    return model, model_columns

model, model_columns = load_assets()


# =========================================================
# 🧠 HELPER FUNCTIONS (FROM DESIGN 1)
# =========================================================
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
DISPLAY_ORDER = ["Low", "Moderate", "High"]

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
    return "There is a risk of underperformance."


# =========================================================
# 🏠 HEADER (DESIGN 1)
# =========================================================
st.markdown("""
<div class="hero-card">
<h1>🧵 AI-Powered Garment Factory Productivity Predictor</h1>
<p>Decision-support system using Decision Tree model</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Deployment", "Active")
c2.metric("Model", "Decision Tree")
c3.metric("Output", "3 Classes")


# =========================================================
# 🧾 INPUT SECTION (MERGED DESIGN 1 + 2)
# =========================================================
form_is_invalid = False

st.divider()

# 🔴 IMPORTANT DETAILS (Design 2 structure)
st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce & Workload")

    team = st.slider("Team Number", 1, 12, 1)

    no_of_workers = st.number_input("Number of Workers", value=30)
    if not (2 <= no_of_workers <= 89):
        st.error("Workers must be between 2 and 89")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", value=500)
    if not (0 <= wip <= 2698):
        st.error("WIP out of range")
        form_is_invalid = True

with col2:
    st.subheader("⚙️ Production Complexity")

    smv = st.number_input("SMV", value=22.0)
    if not (2.9 <= smv <= 54.6):
        st.error("SMV out of range")
        form_is_invalid = True

    no_of_style_change = st.selectbox("Style Changes", [0, 1, 2])


# 🟡 SUPPORTING DETAILS
st.divider()
st.markdown("## 🟡 Supporting Operational Details")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📅 Time & Department")

    day = st.selectbox("Day",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )

    quarter = st.selectbox("Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )

    department = st.selectbox("Department", ["sewing", "finished"])

with col4:
    st.subheader("💰 Incentives & Efficiency")

    incentive = st.number_input("Incentive", value=100)

    over_time = st.slider("Over Time (Minutes)", 0, 25920, 0)

    idle_time = st.number_input("Idle Time", value=0)
    idle_men = st.number_input("Idle Workers", value=0)


# =========================================================
# 🚀 PREDICTION BUTTON
# =========================================================
st.divider()
st.subheader("🚀 Generate Prediction")

generate = st.button("Generate Productivity Forecast",
                     use_container_width=True,
                     disabled=form_is_invalid)

if form_is_invalid:
    st.warning("Please fix errors first.")


# =========================================================
# 📊 OUTPUT (DESIGN 1 + 2 COMBINED)
# =========================================================
if generate:

    # Build input
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    input_df['smv'] = smv
    input_df['wip'] = wip
    input_df['over_time'] = over_time
    input_df['incentive'] = incentive
    input_df['idle_time'] = idle_time
    input_df['idle_men'] = idle_men
    input_df['no_of_workers'] = no_of_workers

    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in model_columns:
            input_df[col_name] = 1

    set_dummy("quarter", quarter)
    set_dummy("department", department)
    set_dummy("day", day)
    set_dummy("no_of_style_change", no_of_style_change)

    input_df = input_df[model_columns]

    # Prediction
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    class_values = list(getattr(model, "classes_", [0, 1, 2]))
    prob_dict = get_probability_dict(probs, class_values)

    predicted_label = LABEL_MAP[prediction]

    # 🎯 RESULT CARD
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown(f"### Predicted Productivity Level: **{predicted_label}**")
    st.write(productivity_note(predicted_label))

    # 🎯 PROBABILITY BARS (Design 1)
    c1, c2, c3 = st.columns(3)

    for col, label in zip([c1, c2, c3], DISPLAY_ORDER):
        with col:
            st.metric(label, f"{prob_dict[label]:.2%}")
            st.progress(prob_dict[label])

    # 🎯 FEEDBACK (Design 2 style)
    if predicted_label == "High":
        st.success(f"Confidence: {prob_dict['High']:.2%}")
        st.balloons()
    elif predicted_label == "Moderate":
        st.warning(f"Confidence: {prob_dict['Moderate']:.2%}")
    else:
        st.error(f"Confidence: {prob_dict['Low']:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    # 📋 EXTRA DETAILS
    with st.expander("📋 View Processed Input Data"):
        st.dataframe(input_df)

    with st.expander("ℹ️ Interpretation Note"):
        st.write(
            "Decision Tree may give identical probabilities for similar patterns (same leaf node)."
        )
