import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    layout="wide",
    page_icon="🧵"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.markdown("""
# 🧵 Garment Productivity Predictor
### 📊 Decision Tree Model Dashboard
""")

st.info("This system predicts **factory productivity level** based on operational inputs.")

# --- TRACK VALIDATION ---
form_is_invalid = False

# ==============================
# 📥 INPUT SECTION
# ==============================
st.markdown("## 🧾 Input Parameters")

col1, col2, col3 = st.columns(3)

# -------- COLUMN 1 --------
with col1:
    st.markdown("### 📅 Time & Production Context")
    
    day = st.selectbox("Day of Week", 
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    
    quarter = st.selectbox("Production Quarter", 
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    
    team = st.slider("Team Number", 1, 12, 1)

# -------- COLUMN 2 --------
with col2:
    st.markdown("### ⚙️ Resource Allocation")
    
    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("⚠️ Must be ≤ 23,122")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Range: 2 – 90")
        form_is_invalid = True

    style_change = st.selectbox("Style Changes", ["0", "1", "2"])
    
    smv = st.number_input("SMV (Task Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ Range: 2.9 – 54.6")
        form_is_invalid = True

# -------- COLUMN 3 --------
with col3:
    st.markdown("### 💰 Incentives & Performance")
    
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Must be ≤ 3,600")
        form_is_invalid = True
        
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
    
    idle_time = st.number_input("Idle Time (minutes)", value=0)
    if idle_time > 300:
        st.error("⚠️ Must be ≤ 300")
        form_is_invalid = True
        
    idle_men = st.number_input("Idle Workers", value=0)
    if idle_men > 45:
        st.error("⚠️ Must be ≤ 45")
        form_is_invalid = True


# ==============================
# 🔮 PREDICTION SECTION
# ==============================
st.markdown("---")
st.markdown("## 🔮 Productivity Prediction")

if form_is_invalid:
    st.warning("⚠️ Please correct the input errors above.")
    st.button("Generate Prediction", disabled=True)

else:
    if st.button("🚀 Generate Productivity Forecast", use_container_width=True):

        # Prepare input
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime

        # Encoding
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        input_df = input_df[model_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction]

        # ==============================
        # 🎯 RESULT DISPLAY
        # ==============================
        st.markdown("### 🎯 Prediction Result")

        if result == 'High':
            st.markdown(f"""
            ## 🟢 **HIGH PRODUCTIVITY**
            **Confidence:** {probs[2]:.2%}
            
            ✅ Production is operating at an **optimal level**  
            📈 Strong efficiency and resource utilization  
            """)
            st.balloons()

        elif result == 'Moderate':
            st.markdown(f"""
            ## 🟡 **MODERATE PRODUCTIVITY**
            **Confidence:** {probs[1]:.2%}
            
            ⚖️ Performance is **stable but can improve**  
            🔧 Consider optimizing resources or reducing idle time  
            """)

        else:
            st.markdown(f"""
            ## 🔴 **LOW PRODUCTIVITY**
            **Confidence:** {probs[0]:.2%}
            
            ⚠️ High risk of **underperformance**  
            🚨 Immediate action recommended (review workload, workers, incentives)  
            """)
