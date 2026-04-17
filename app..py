import streamlit as st
import pandas as pd
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
# LOAD MODEL & COLUMNS
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_dt_model.pkl")
    columns = joblib.load("garment_dt_columns.pkl")
    return model, columns

model, model_columns = load_assets()

# =========================================================
# HEADER SECTION
# =========================================================
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### 📊 AI-Powered Decision Support System")

st.success("✅ Model successfully loaded and ready for prediction.")

st.info("""
This system predicts **garment production productivity level** using a trained **Decision Tree model**.

**Prediction Classes:**
- 🔴 Low Productivity  
- 🟡 Moderate Productivity  
- 🟢 High Productivity  
""")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def validate_inputs(inputs):
    errors = []

    if not (2 <= inputs['workers'] <= 90):
        errors.append("Workers must be between 2 and 90.")

    if inputs['wip'] > 23122:
        errors.append("WIP exceeds maximum limit (23,122).")

    if not (2.9 <= inputs['smv'] <= 54.6):
        errors.append("SMV must be between 2.9 and 54.6.")

    if inputs['incentive'] > 3600:
        errors.append("Incentive exceeds maximum limit (3,600).")

    if inputs['idle_time'] > 300:
        errors.append("Idle time exceeds 300 minutes.")

    if inputs['idle_men'] > 45:
        errors.append("Idle workers exceed 45.")

    return errors


def create_input_dataframe(inputs):
    df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numerical Features
    df['team'] = inputs['team']
    df['smv'] = inputs['smv']
    df['wip'] = inputs['wip']
    df['incentive'] = inputs['incentive']
    df['idle_time'] = inputs['idle_time']
    df['idle_men'] = inputs['idle_men']
    df['no_of_workers'] = inputs['workers']
    df['over_time_scaled'] = inputs['overtime']

    # One-hot encoding helper
    def encode(feature, value):
        col = f"{feature}_{value}"
        if col in model_columns:
            df[col] = 1

    encode('quarter', inputs['quarter'])
    encode('department', inputs['department'].lower())
    encode('day', inputs['day'])
    encode('no_of_style_change', inputs['style_change'])

    return df[model_columns]


def display_prediction(pred, probs):
    labels = ['Low', 'Moderate', 'High']
    result = labels[pred]

    st.markdown(f"## 🏷️ Prediction: **{result} Productivity**")

    if result == "High":
        st.success(f"Confidence: {probs[2]:.2%} — Excellent operational performance.")
        st.balloons()

    elif result == "Moderate":
        st.warning(f"Confidence: {probs[1]:.2%} — Stable, but improvement possible.")

    else:
        st.error(f"Confidence: {probs[0]:.2%} — High risk of low productivity.")

# =========================================================
# INPUT FORM
# =========================================================
st.divider()
st.header("📝 Enter Production Details")

with st.form("prediction_form"):

    # -------------------------
    # IMPORTANT FACTORS
    # -------------------------
    st.subheader("🔴 Core Production Factors")

    col1, col2 = st.columns(2)

    with col1:
        team = st.slider("Team Number", 1, 12, 1)
        workers = st.number_input("Number of Workers", value=30)
        wip = st.number_input("Work in Progress (WIP)", value=500)

    with col2:
        smv = st.number_input("SMV (Standard Minute Value)", value=22.0)
        style_change = st.selectbox("Style Changes", ["0", "1", "2"])

    # -------------------------
    # SUPPORTING FACTORS
    # -------------------------
    st.subheader("🟡 Operational Details")

    col3, col4 = st.columns(2)

    with col3:
        day = st.selectbox("Day", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
        )
        quarter = st.selectbox("Quarter",
            ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
        )
        department = st.selectbox("Department", ["Sewing", "Finishing"])

    with col4:
        incentive = st.number_input("Incentive", value=100)
        overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
        idle_time = st.number_input("Idle Time (minutes)", value=0)
        idle_men = st.number_input("Idle Workers", value=0)

    submitted = st.form_submit_button("🔍 Generate Prediction")

# =========================================================
# PROCESS PREDICTION
# =========================================================
if submitted:

    user_inputs = {
        "team": team,
        "workers": workers,
        "wip": wip,
        "smv": smv,
        "style_change": style_change,
        "day": day,
        "quarter": quarter,
        "department": department,
        "incentive": incentive,
        "overtime": overtime,
        "idle_time": idle_time,
        "idle_men": idle_men
    }

    # Validate inputs
    errors = validate_inputs(user_inputs)

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    else:
        # Create input data
        input_df = create_input_dataframe(user_inputs)

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Display results
        st.divider()
        display_prediction(prediction, probabilities)

        # Show input summary
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df)
