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
# 🎨 PROFESSIONAL UI STYLING
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
    transition: 0.3s;
}
.section-card:hover {
    transform: translateY(-3px);
}

/* Result Card */
.result-card {
    border-radius: 20px;
    padding: 1.5rem;
    background: linear-gradient(135deg, #ffffff, #f1f5f9);
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #e0f2fe;
    color: #0369a1;
    font-size: 12px;
    margin-right: 6px;
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

<span class="badge">Machine Learning</span>
<span class="badge">Decision Support</span>
<span class="badge">Real-time Prediction</span>

</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Deployment", "Active")
c2.metric("Model", "Decision Tree")
c3.metric("Output", "3 Classes")


# =========================================================
# 🧾 INPUT SECTION
# =========================================================
form_is_invalid = False

st.divider()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## 🔴 Important Production Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Workforce & Workload")

    no_of_workers = st.number_input("Number of Workers", value=30)
    if not (2 <= no_of_workers <= 89):
        st.error("Workers must be between 2 and 89")
        form_is_invalid = True

    wip = st.number_input("Work in Progress (WIP)", value=500)
    if not (0 <= wip <= 23122):
        st.error("WIP's value must be between 7 and 23122")
        form_is_invalid = True

with col2:
    st.subheader("⚙️ Production Complexity")

    smv = st.number_input("SMV", value=22.0)
    if not (2.9 <= smv <= 54.56):
        st.error("SMV's value must be between 2.9 and 54.56")
        form_is_invalid = True

    no_of_style_change = st.selectbox("Style Changes", [0, 1, 2])

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
    if not (0 <= incentive <= 3600):
        st.error("Incentive must be between 0 and 3600")
        form_is_invalid = True
            
    over_time = st.number_input("Over Time (Minutes)",value=0)
    if not (0 <= over_time <= 25920):
        st.error("Over Time's value must be between 0 and 25920 minutes")
        form_is_invalid = True
        
    idle_time = st.number_input("Idle Time", value=0)
    if not (0 <= idle_time <= 300):
        st.error("Idle_time must be between 0 and 300")
        form_is_invalid = True
            
    idle_men = st.number_input("Idle Workers", value=0)
    if not (0 <= idle_men <= 45):
        st.error("Idle_men must be between 0 and 45")
        form_is_invalid = True

st.markdown('</div>', unsafe_allow_html=True)


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

    class_values = list(getattr(model, "classes_", [0, 1, 2]))
    prob_dict = get_probability_dict(probs, class_values)

    predicted_label = LABEL_MAP[prediction]

    # 🎨 COLOR RESULT DESIGN
    color_map = {
        "High": {"bg":"#dcfce7","border":"#16a34a","text":"#14532d","icon":"🟢"},
        "Moderate": {"bg":"#fef9c3","border":"#f59e0b","text":"#78350f","icon":"🟡"},
        "Low": {"bg":"#fee2e2","border":"#dc2626","text":"#7f1d1d","icon":"🔴"}
    }

    style = color_map[predicted_label]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background:{style['bg']};
        border-left:8px solid {style['border']};
        padding:18px;
        border-radius:14px;
        margin-bottom:15px;
    ">
    <h2 style='text-align:center; color:{style["text"]};'>
    {style["icon"]} {predicted_label} Productivity
    </h2>

    <p style='text-align:center; color:{style["text"]}; font-size:16px;'>
    {productivity_note(predicted_label)}
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Prediction Confidence")

    c1, c2, c3 = st.columns(3)

    bar_color = {
        "High": "#16a34a",
        "Moderate": "#f59e0b",
        "Low": "#dc2626"
    }

    for col, label in zip([c1, c2, c3], DISPLAY_ORDER):
        with col:
            st.caption(f"{label} Confidence Level")
            st.metric(label, f"{prob_dict[label]:.2%}")
            st.progress(prob_dict[label])
            st.markdown(
                f"<div style='height:5px;background:{bar_color[label]};border-radius:5px;'></div>",
                unsafe_allow_html=True
            )

    if predicted_label == "High":
        st.success(f"Confidence: {prob_dict['High']:.2%}")
        st.balloons()
    elif predicted_label == "Moderate":
        st.warning(f"Confidence: {prob_dict['Moderate']:.2%}")
    else:
        st.error(f"Confidence: {prob_dict['Low']:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View Processed Input Data"):
        st.dataframe(input_df)

    with st.expander("ℹ️ Interpretation Note"):
        st.write(
            "Decision Tree may give identical probabilities for similar patterns (same leaf node)."
        )


# =========================================================
# 📌 FOOTER
# =========================================================
st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:gray;'>
Built with Streamlit | AI Decision Support System for Garment Industry
</p>
""", unsafe_allow_html=True)
