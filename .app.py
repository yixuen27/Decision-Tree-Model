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
# 🎨 PROFESSIONAL UI STYLING (UPDATED WITH COLORS)
# =========================================================
st.markdown("""
<style>

/* Global */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hero Header */
.hero-card {
    background: linear-gradient(135deg, #1e3a8a, #0f172a);
    color: white;
    padding: 1.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* Section Cards */
.section-card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 1.2rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

/* Result Base Card */
.result-card {
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* RESULT COLORS */
.result-high {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border-left: 6px solid #16a34a;
}

.result-moderate {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-left: 6px solid #d97706;
}

.result-low {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-left: 6px solid #dc2626;
}

/* Badge */
.result-badge {
    font-size: 26px;
    font-weight: 700;
    padding: 10px 20px;
    border-radius: 12px;
    display: inline-block;
    margin-top: 10px;
}

.badge-high {
    background: #16a34a;
    color: white;
}

.badge-moderate {
    background: #d97706;
    color: white;
}

.badge-low {
    background: #dc2626;
    color: white;
}

/* Button */
.stButton>button {
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 🤖 LOAD MODEL
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    model_columns = joblib.load("garment_dt_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()


# =========================================================
# 🧠 HELPER FUNCTIONS
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
# 🏠 HEADER
# =========================================================
st.markdown("""
<div class="hero-card">
<h1>🧵 AI-Powered Garment Factory Productivity Predictor</h1>
<p>Decision-support system using Decision Tree model</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 🧾 INPUT SECTION
# =========================================================
form_is_invalid = False

st.divider()
st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    no_of_workers = st.number_input("Number of Workers", value=30)
    if not (2 <= no_of_workers <= 89):
        st.error("Workers must be between 2 and 89")
        form_is_invalid = True

    wip = st.number_input("WIP", value=500)

with col2:
    smv = st.number_input("SMV", value=22.0)
    no_of_style_change = st.selectbox("Style Changes", [0,1,2])

st.divider()
st.markdown("## 🟡 Supporting Operational Details")

col3, col4 = st.columns(2)

with col3:
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    department = st.selectbox("Department", ["sewing","finished"])

with col4:
    incentive = st.number_input("Incentive", value=100)
    over_time = st.slider("Over Time (Minutes)", 0, 25920, 0)
    idle_time = st.number_input("Idle Time", value=0)
    idle_men = st.number_input("Idle Workers", value=0)


# =========================================================
# 🚀 BUTTON
# =========================================================
generate = st.button("Generate Productivity Forecast", disabled=form_is_invalid)


# =========================================================
# 📊 OUTPUT
# =========================================================
if generate:

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

    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    class_values = list(getattr(model, "classes_", [0,1,2]))
    prob_dict = get_probability_dict(probs, class_values)
    predicted_label = LABEL_MAP[prediction]

    # 🎨 COLOR LOGIC
    if predicted_label == "High":
        result_class = "result-high"
        badge_class = "badge-high"
    elif predicted_label == "Moderate":
        result_class = "result-moderate"
        badge_class = "badge-moderate"
    else:
        result_class = "result-low"
        badge_class = "badge-low"

    st.markdown(f'<div class="result-card {result_class}">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;">
        <h2>🎯 Predicted Productivity</h2>
        <div class="result-badge {badge_class}">
            {predicted_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(productivity_note(predicted_label))

    # 📊 COLORED PROBABILITY BARS
    color_map = {
        "Low": "#dc2626",
        "Moderate": "#d97706",
        "High": "#16a34a"
    }

    c1, c2, c3 = st.columns(3)

    for col, label in zip([c1, c2, c3], DISPLAY_ORDER):
        with col:
            st.metric(label, f"{prob_dict[label]:.2%}")
            st.markdown(f"""
            <div style="background:#e5e7eb; border-radius:10px;">
                <div style="
                    width:{prob_dict[label]*100}%;
                    background:{color_map[label]};
                    padding:6px;
                    border-radius:10px;
                    text-align:right;
                    color:white;">
                    {prob_dict[label]:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
