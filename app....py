
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --------------------------------------------------
# SMALL CSS STYLING
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #4b5563;
    margin-bottom: 1rem;
}
.card {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 16px;
    background-color: rgba(250, 250, 250, 0.6);
}
.result-card {
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(49, 51, 63, 0.18);
    background: linear-gradient(180deg, rgba(249,250,251,0.95), rgba(255,255,255,0.98));
}
.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
MODEL_PATH = Path("garment_dt_model.pkl")
COLUMNS_PATH = Path("garment_dt_columns.pkl")
DATA_PATH = Path("cleaned_garments_worker_productivity(2).csv")

# --------------------------------------------------
# LOAD ASSETS
# --------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, model_columns

@st.cache_data
def load_reference_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

model, model_columns = load_assets()
reference_df = load_reference_data()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_ORDER = ["Low", "Moderate", "High"]

def get_feature_limits(df):
    # Fallback values if CSV is unavailable
    defaults = {
        "team": (1, 12, 1),
        "no_of_workers": (2, 90, 30),
        "wip": (0, 23122, 500),
        "smv": (2.9, 54.6, 22.0),
        "incentive": (0, 3600, 100),
        "idle_time": (0, 300, 0),
        "idle_men": (0, 45, 0),
        "over_time_scaled": (-2.0, 2.0, 0.0),
    }
    if df is None:
        return defaults

    limits = {}
    for col, fallback in defaults.items():
        if col in df.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            default = fallback[2]
            limits[col] = (col_min, col_max, default)
        else:
            limits[col] = fallback
    return limits

LIMITS = get_feature_limits(reference_df)

def build_input_frame(
    team, smv, wip, incentive, idle_time, idle_men, no_of_workers, over_time_scaled,
    quarter, department, day, no_of_style_change
):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    numeric_values = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": no_of_workers,
        "over_time_scaled": over_time_scaled,
    }

    for col, value in numeric_values.items():
        if col in input_df.columns:
            input_df.at[0, col] = value

    def activate_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1

    activate_dummy("quarter", quarter)
    activate_dummy("department", department)
    activate_dummy("day", day)
    activate_dummy("no_of_style_change", str(no_of_style_change))

    return input_df[model_columns]

def extract_probabilities(input_df):
    # Get probabilities and map them safely to all 3 labels,
    # even if model.classes_ order changes.
    raw_probs = model.predict_proba(input_df)[0]
    prob_map = {int(cls): float(prob) for cls, prob in zip(model.classes_, raw_probs)}

    full_probs = {
        "Low": prob_map.get(0, 0.0),
        "Moderate": prob_map.get(1, 0.0),
        "High": prob_map.get(2, 0.0),
    }
    return full_probs

def detect_low_prediction_issue():
    # Sample broad combinations to see whether the current saved model
    # is able to return any non-zero probability for Low.
    # If not, likely model artifact/version/training issue, not UI issue.
    sample_inputs = [
        (1, 5.0, 20000, 0, 250, 35, 5, -1.8, "Quarter5", "finished", "Sunday", 2),
        (12, 50.0, 15000, 0, 200, 30, 8, -1.5, "Quarter4", "finished", "Saturday", 2),
        (3, 10.0, 18000, 10, 300, 40, 4, -2.0, "Quarter3", "finished", "Thursday", 2),
        (8, 35.0, 500, 2000, 0, 0, 80, 1.2, "Quarter1", "sewing", "Monday", 0),
    ]
    low_found = False
    low_positive_prob = False

    for vals in sample_inputs:
        sample_df = build_input_frame(*vals)
        pred = int(model.predict(sample_df)[0])
        probs = extract_probabilities(sample_df)

        if pred == 0:
            low_found = True
        if probs["Low"] > 0:
            low_positive_prob = True

    return low_found, low_positive_prob

def quality_message(top_label):
    if top_label == "High":
        return "Strong production conditions based on the selected inputs."
    if top_label == "Moderate":
        return "Productivity appears stable, but there is still room for improvement."
    return "This setup shows risk factors that may reduce productivity."

def get_top_risk_flags(wip, idle_time, idle_men, incentive, no_of_workers, smv):
    flags = []
    if wip >= 15000:
        flags.append("Very high WIP may slow production flow.")
    if idle_time >= 120:
        flags.append("High idle time may reduce operational efficiency.")
    if idle_men >= 15:
        flags.append("A larger number of idle workers may indicate underutilization.")
    if incentive == 0:
        flags.append("No incentive is provided in this scenario.")
    if no_of_workers <= 8:
        flags.append("A small workforce may limit output capacity.")
    if smv >= 40:
        flags.append("A high SMV suggests more complex work content.")
    return flags[:3]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">🧵 Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Professional prototype for productivity-level prediction using a Decision Tree model.</div>',
    unsafe_allow_html=True
)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown('<div class="card"><b>Model</b><br>Decision Tree Classifier</div>', unsafe_allow_html=True)
with top2:
    st.markdown('<div class="card"><b>Output Classes</b><br>Low · Moderate · High</div>', unsafe_allow_html=True)
with top3:
    st.markdown('<div class="card"><b>Purpose</b><br>Support operational productivity analysis</div>', unsafe_allow_html=True)

if reference_df is not None:
    class_counts = reference_df["productivity_level"].value_counts().to_dict()
    st.caption(
        f"Reference dataset loaded successfully. Historical class counts — "
        f"Low: {class_counts.get('Low', 0)}, Moderate: {class_counts.get('Moderate', 0)}, High: {class_counts.get('High', 0)}"
    )
else:
    st.caption("Reference dataset not detected. App is running with fallback validation ranges.")

low_found, low_positive_prob = detect_low_prediction_issue()
if not low_positive_prob:
    st.warning(
        "The current saved model appears unable to produce a visible Low probability in tested scenarios. "
        "This usually points to a model artifact or version/training issue rather than a Streamlit display problem."
    )

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### Workforce & Production Load")
    team = st.slider("Team Number", int(LIMITS["team"][0]), int(LIMITS["team"][1]), int(LIMITS["team"][2]))
    no_of_workers = st.number_input(
        "Number of Workers",
        min_value=int(LIMITS["no_of_workers"][0]),
        max_value=int(LIMITS["no_of_workers"][1]),
        value=int(LIMITS["no_of_workers"][2]),
        step=1
    )
    wip = st.number_input(
        "Work in Progress (WIP)",
        min_value=int(LIMITS["wip"][0]),
        max_value=int(LIMITS["wip"][1]),
        value=int(LIMITS["wip"][2]),
        step=1
    )
    smv = st.number_input(
        "SMV (Standard Minute Value)",
        min_value=float(LIMITS["smv"][0]),
        max_value=float(LIMITS["smv"][1]),
        value=float(LIMITS["smv"][2]),
        step=0.1
    )

    st.markdown("### Schedule & Production Context")
    day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    quarter_options = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    department_options = ["finished", "sewing"]

    day = st.selectbox("Day of the Week", day_options)
    quarter = st.selectbox("Production Quarter", quarter_options)
    department = st.selectbox("Department", department_options, format_func=lambda x: x.title())

with right:
    st.markdown("### Efficiency & Operational Conditions")
    incentive = st.number_input(
        "Incentive Amount",
        min_value=int(LIMITS["incentive"][0]),
        max_value=int(LIMITS["incentive"][1]),
        value=int(LIMITS["incentive"][2]),
        step=10
    )
    over_time_scaled = st.slider(
        "Overtime (Scaled)",
        min_value=float(LIMITS["over_time_scaled"][0]),
        max_value=float(LIMITS["over_time_scaled"][1]),
        value=float(LIMITS["over_time_scaled"][2]),
        step=0.1
    )
    idle_time = st.number_input(
        "Idle Time (Minutes)",
        min_value=int(LIMITS["idle_time"][0]),
        max_value=int(LIMITS["idle_time"][1]),
        value=int(LIMITS["idle_time"][2]),
        step=1
    )
    idle_men = st.number_input(
        "Idle Workers",
        min_value=int(LIMITS["idle_men"][0]),
        max_value=int(LIMITS["idle_men"][1]),
        value=int(LIMITS["idle_men"][2]),
        step=1
    )
    no_of_style_change = st.selectbox("Number of Style Changes", [0, 1, 2])

    st.markdown("### Analyst Note")
    st.markdown(
        '<div class="small-note">This interface now displays all class probabilities, '
        'so the result will not look repetitive even when the top predicted class stays the same.</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()
predict_clicked = st.button("Generate Productivity Forecast", use_container_width=True, type="primary")

if predict_clicked:
    input_df = build_input_frame(
        team=team,
        smv=smv,
        wip=wip,
        incentive=incentive,
        idle_time=idle_time,
        idle_men=idle_men,
        no_of_workers=no_of_workers,
        over_time_scaled=over_time_scaled,
        quarter=quarter,
        department=department,
        day=day,
        no_of_style_change=no_of_style_change
    )

    pred_code = int(model.predict(input_df)[0])
    probabilities = extract_probabilities(input_df)
    top_label = LABEL_MAP.get(pred_code, "Unknown")
    top_prob = probabilities.get(top_label, 0.0)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"### Predicted Productivity Level: **{top_label}**")
    st.write(quality_message(top_label))

    c1, c2, c3 = st.columns(3)
    c1.metric("Low", f"{probabilities['Low']:.2%}")
    c2.metric("Moderate", f"{probabilities['Moderate']:.2%}")
    c3.metric("High", f"{probabilities['High']:.2%}")

    st.progress(float(top_prob))
    st.caption(f"Top predicted class confidence: {top_prob:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

    if top_label == "High":
        st.success("This scenario is associated with a high expected productivity level.")
    elif top_label == "Moderate":
        st.warning("This scenario is associated with a moderate productivity level.")
    else:
        st.error("This scenario is associated with a low productivity level.")

    risk_flags = get_top_risk_flags(wip, idle_time, idle_men, incentive, no_of_workers, smv)
    if risk_flags:
        st.markdown("### Operational Interpretation")
        for item in risk_flags:
            st.write(f"• {item}")

    with st.expander("View Encoded Input Used by the Model"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("Why the old app looked repetitive"):
        st.write(
            "Decision Tree models often return the same probability distribution for different inputs that fall into the same terminal leaf. "
            "So repeating percentages can be normal. The bigger issue in the old version was that it only displayed the probability of the predicted class, "
            "which made the output look even more repetitive."
        )
        st.write(
            "This revised version displays all three class probabilities at the same time: Low, Moderate, and High."
        )
