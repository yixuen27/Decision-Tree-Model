import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --- STYLE ---
st.markdown("""
<style>
.big-title {font-size:42px;font-weight:bold;color:#2E86C1;}
.card {background:#f8f9fa;padding:18px;border-radius:12px;}
.result-box {padding:30px;border-radius:15px;text-align:center;font-size:32px;font-weight:bold;}
.small-text {color:gray;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_dt_model.pkl')
    model_columns = joblib.load('garment_dt_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- TITLE ---
st.markdown('<p class="big-title">🧵 Garment Productivity Predictor</p>', unsafe_allow_html=True)
st.caption("Decision Tree Model • Decision Support Dashboard")

form_is_invalid = False

# --- INPUT LAYOUT ---
col1, col2, col3 = st.columns(3)

# IMPORTANT DETAILS
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📅 Important Details")
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    dept = st.selectbox("Department", ["Sewing","Finishing"])
    team = st.slider("Team",1,12,1)
    st.markdown('</div>', unsafe_allow_html=True)

# RESOURCE DETAILS
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Resource Details")

    wip = st.number_input("WIP",500)
    if wip > 23122:
        st.error("Max: 23122"); form_is_invalid=True

    workers = st.number_input("Workers",30)
    if workers > 90 or workers < 2:
        st.error("Range: 2–90"); form_is_invalid=True

    style_change = st.selectbox("Style Change",["0","1","2"])

    smv = st.number_input("SMV",22.0)
    if smv > 55 or smv < 2.9:
        st.error("Range: 2.9–54.6"); form_is_invalid=True

    st.markdown('</div>', unsafe_allow_html=True)

# PERFORMANCE DETAILS
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Performance Details")

    incentive = st.number_input("Incentive",100)
    if incentive > 3600:
        st.error("Max: 3600"); form_is_invalid=True

    overtime = st.slider("Overtime",-2.0,2.0,0.0)

    idle_time = st.number_input("Idle Time",0)
    if idle_time > 300:
        st.error("Max: 300"); form_is_invalid=True

    idle_men = st.number_input("Idle Workers",0)
    if idle_men > 45:
        st.error("Max: 45"); form_is_invalid=True

    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION ---
st.divider()
center = st.columns([1,2,1])[1]

if form_is_invalid:
    st.warning("Fix errors before prediction")
    center.button("Predict", disabled=True)
else:
    if center.button("🚀 Generate Prediction", use_container_width=True):

        # INPUT DF
        input_df = pd.DataFrame(0,index=[0],columns=model_columns)

        input_df['team']=team
        input_df['smv']=smv
        input_df['wip']=wip
        input_df['incentive']=incentive
        input_df['idle_time']=idle_time
        input_df['idle_men']=idle_men
        input_df['no_of_workers']=workers
        input_df['over_time_scaled']=overtime

        def set_dummy(cat,val):
            col=f"{cat}_{val}"
            if col in model_columns:
                input_df[col]=1

        set_dummy('quarter',quarter)
        set_dummy('department',dept.lower())
        set_dummy('day',day)
        set_dummy('no_of_style_change',style_change)

        input_df=input_df[model_columns]

        # PREDICT
        pred=model.predict(input_df)[0]
        probs=model.predict_proba(input_df)[0]

        labels=['Low','Moderate','High']
        result=labels[pred]

        st.divider()

        # --- RESULT DISPLAY ---
        if result=="High":
            st.markdown('<div class="result-box" style="background:#d4edda;color:#155724;">✅ HIGH PRODUCTIVITY</div>',unsafe_allow_html=True)
            desc="Factory is operating at optimal efficiency."
            idx=2
            st.balloons()

        elif result=="Moderate":
            st.markdown('<div class="result-box" style="background:#fff3cd;color:#856404;">⚠️ MODERATE PRODUCTIVITY</div>',unsafe_allow_html=True)
            desc="Performance is stable but improvement is possible."
            idx=1

        else:
            st.markdown('<div class="result-box" style="background:#f8d7da;color:#721c24;">❌ LOW PRODUCTIVITY</div>',unsafe_allow_html=True)
            desc="High risk of underperformance detected."
            idx=0

        st.metric("Confidence", f"{probs[idx]:.2%}")
        st.write(desc)

        # =========================
        # 📈 PROBABILITY CHART
        # =========================
        st.subheader("📈 Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probability Distribution")
        st.pyplot(fig)

        # =========================
        # 📊 FEATURE IMPORTANCE
        # =========================
        st.subheader("📊 Key Factors Influencing Prediction")

        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": model_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots()
        ax2.barh(feat_imp["Feature"], feat_imp["Importance"])
        ax2.invert_yaxis()
        ax2.set_title("Top 10 Important Features")
        st.pyplot(fig2)

        # =========================
        # 🧠 BUSINESS INSIGHT
        # =========================
        st.subheader("🧠 Smart Insight")

        if result == "High":
            st.success("Maintain current workforce and incentive strategy. Focus on consistency.")
        elif result == "Moderate":
            st.warning("Consider adjusting incentive or reducing idle time to boost performance.")
        else:
            st.error("Re-evaluate worker allocation, reduce idle time, and improve workflow efficiency.")
