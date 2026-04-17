import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor (Decision Tree)", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')   # <-- changed
    model_columns = joblib.load('dt_model_columns.pkl')  # <-- changed
    return model, model_columns

model, model_columns = load_assets()

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor (Decision Tree)")
st.info("**Model Info:** Tuned Decision Tree Classifier.")

form_is_invalid = False

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Allocation")
    
    wip = st.number_input("Work in Progress (wip)", value=500)
    if wip > 23122:
        st.error("⚠️ Max wip is 23,122")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Range is 2 to 90")
        form_is_invalid = True

    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    
    smv = st.number_input("SMV (Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ Range is 2.9 to 54.6")
        form_is_invalid = True

with col3:
    st.subheader("💰 Incentives & Metrics")
    
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Max Incentive is 3,600")
        form_is_invalid = True
        
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
    
    idle_time = st.number_input("Idle Time (Mins)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max idle time is 300")
        form_is_invalid = True
        
    idle_men = st.number_input("Idle Workers Count", value=0)
    if idle_men > 45:
        st.error("⚠️ Max idle workers is 45")
        form_is_invalid = True

# --- PREDICTION ---
st.divider()

if form_is_invalid:
    st.warning("Please fix input errors before prediction.")
    st.button("Generate Productivity Forecast", disabled=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        
        # Create input dataframe
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numerical inputs
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

        # Encoding function
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # Align columns
        input_df = input_df[model_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # Display result
        st.markdown(f"## Predicted Tier: **{result}**")

        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%}")
            st.balloons()
        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%}")
        else:
            st.error(f"Confidence: {probs[0]:.2%}")