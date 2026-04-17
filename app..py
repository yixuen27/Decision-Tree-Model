import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS (CHANGED)
# =========================================================
@st.cache_resource
def load_model_assets():
    model = joblib.load("garment_dt_model.pkl")   # ✅ Decision Tree
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_model_assets()
df = load_dataset()

# =========================================================
# DATA OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
day_options = sorted(df["day"].dropna().unique().tolist())
style_change_options = sorted(df["no_of_style_change"].dropna().unique().tolist())

smv_min, smv_max = float(df["smv"].min()), float(df["smv"].max())
wip_min, wip_max = int(df["wip"].min()), int(df["wip"].max())
over_time_min, over_time_max = int(df["over_time"].min()), int(df["over_time"].max())
incentive_min, incentive_max = int(df["incentive"].min()), int(df["incentive"].max())
idle_time_min, idle_time_max = int(df["idle_time"].min()), int(df["idle_time"].max())
idle_men_min, idle_men_max = int(df["idle_men"].min()), int(df["idle_men"].max())
workers_min, workers_max = int(df["no_of_workers"].min()), int(df["no_of_workers"].max())

# =========================================================
# HELPER FUNCTIONS (UNCHANGED)
# =========================================================
def set_dummy_value(input_df, prefix, value):
    for col in [
        f"{prefix}_{value}",
        f"{prefix}_{str(value).lower()}",
        f"{prefix}_{str(value).upper()}",
        f"{prefix}_{str(value).capitalize()}",
    ]:
        if col in input_df.columns:
            input_df[col] = 1
            return

def normalize_prediction(pred):
    if isinstance(pred, str):
        return pred
    return {0: "Low", 1: "Moderate", 2: "High"}.get(int(pred), str(pred))

def get_result_message(result):
    return {
        "High": ("success", "The current input pattern suggests strong production performance."),
        "Moderate": ("warning", "The current input pattern suggests average but stable production performance."),
        "Low": ("error", "The current input pattern suggests a risk of lower productivity.")
    }[result]

def get_recommendations(result, wip, over_time, incentive, idle_time, idle_men, workers, style_change):
    if result == "Low":
        return [
            "Reduce idle time and idle workers to improve efficiency.",
            "Rebalance workload with workforce.",
            "Improve incentive strategy.",
            "Minimize unnecessary style changes."
        ]
    elif result == "Moderate":
        return [
            "Production is stable but can be improved.",
            "Optimize workload planning.",
            "Monitor overtime and idle conditions."
        ]
    else:
        return [
            "Production setup is efficient.",
            "Maintain current operational balance.",
            "Use this as a benchmark."
        ]

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Decision Tree-based prediction system for garment factory productivity.</div>',
    unsafe_allow_html=True
)

st.success("✅ System Status: Decision Tree model loaded successfully.")

# =========================================================
# SIDEBAR (CHANGED)
# =========================================================
with st.sidebar:
    st.header("📌 Prototype Overview")
    st.write("""
Predicts productivity:
- Low
- Moderate
- High
""")

    st.markdown("---")
    st.subheader("🌳 Why Decision Tree?")
    st.write("""
Decision Tree was chosen because:
- easy to interpret and explain
- provides clear decision rules
- fast and efficient for prediction
- suitable for structured data
""")

# =========================================================
# INPUT UI (UNCHANGED)
# =========================================================
st.markdown("## 📥 Production Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Time & Context")
    quarter = st.selectbox("Quarter", quarter_options)
    department = st.selectbox("Department", department_options)
    day = st.selectbox("Day", day_options)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Production Factors")
    smv = st.number_input("SMV", smv_min, smv_max, float(df["smv"].median()))
    wip = st.number_input("WIP", wip_min, wip_max, int(df["wip"].median()))
    no_of_style_change = st.selectbox("Style Changes", style_change_options)
    no_of_workers = st.number_input("Workers", workers_min, workers_max, int(df["no_of_workers"].median()))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Efficiency")
    over_time = st.number_input("Overtime", over_time_min, over_time_max, int(df["over_time"].median()))
    incentive = st.number_input("Incentive", incentive_min, incentive_max, int(df["incentive"].median()))
    idle_time = st.number_input("Idle Time", idle_time_min, idle_time_max, int(df["idle_time"].median()))
    idle_men = st.number_input("Idle Workers", idle_men_min, idle_men_max, int(df["idle_men"].median()))
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):

    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    input_df["smv"] = smv
    input_df["wip"] = wip
    input_df["over_time"] = over_time
    input_df["incentive"] = incentive
    input_df["idle_time"] = idle_time
    input_df["idle_men"] = idle_men
    input_df["no_of_style_change"] = no_of_style_change
    input_df["no_of_workers"] = no_of_workers

    set_dummy_value(input_df, "quarter", quarter)
    set_dummy_value(input_df, "department", department)
    set_dummy_value(input_df, "day", day)

    input_df = input_df[model_columns]

    pred = normalize_prediction(model.predict(input_df)[0])
    probs = model.predict_proba(input_df)[0]

    st.markdown("## 📊 Prediction Results")

    st.metric("Predicted Productivity", pred)
    st.metric("Confidence", f"{np.max(probs):.2%}")

    status, msg = get_result_message(pred)
    getattr(st, status)(msg)

    # probabilities
    prob_df = pd.DataFrame({
        "Level": ["Low", "Moderate", "High"],
        "Probability": probs
    })
    st.bar_chart(prob_df.set_index("Level"))

    # recommendations
    st.markdown("### 💡 Recommendations")
    for r in get_recommendations(pred, wip, over_time, incentive, idle_time, idle_men, no_of_workers, no_of_style_change):
        st.write(f"- {r}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown('<div class="small-note">Decision Tree prototype for garment productivity prediction.</div>', unsafe_allow_html=True)
