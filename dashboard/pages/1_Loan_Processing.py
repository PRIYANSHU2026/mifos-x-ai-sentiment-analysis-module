import streamlit as st
import requests
import time
import os
import sys

# Ensure the parent directory is in the python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.auth import render_sidebar_auth, get_current_role

st.set_page_config(page_title="Loan Processing", page_icon="📝", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

API_URL = "http://127.0.0.1:8000/api"

st.sidebar.markdown("### 🏦 **MIFOS X** AI Platform")
render_sidebar_auth()
role = get_current_role()

# Only Customers and Loan Officers can process new loans
if role not in ["Customer", "Loan Officer", "Administrator"]:
    st.error("Access Denied. You do not have permission to submit loan applications.")
    st.stop()

st.markdown("<h2 style='color:#1F4E79;'>📝 Loan Origination & Processing</h2>", unsafe_allow_html=True)

# ─── Form Inputs ──────────────────────────────────────────
with st.form("loan_application_form"):
    st.markdown("### 👤 Applicant Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    with col2:
        employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
        income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=500)
        region = st.selectbox("Region", ["Urban", "Semiurban", "Rural"])
    with col3:
        credit_score = st.number_input("Credit Score (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        existing_debt = st.number_input("Existing Debt ($)", min_value=0, value=1000, step=100)
        existing_loans = st.number_input("Active Loans Count", min_value=0, value=1)
    with col4:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=100, value=10000, step=500)
        loan_tenure = st.number_input("Loan Tenure (Months)", min_value=1, max_value=360, value=24)
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Home"])
        collateral = st.selectbox("Collateral Offered", ["None", "Vehicle", "Property", "Gold"])
        repayment_history = st.number_input("Repayment History (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    submitted = st.form_submit_button("🚀 Submit to AI Engine")

# ─── Processing ──────────────────────────────────────────
if submitted:
    st.markdown("---")
    
    # ─── Animated Pipeline ─────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("#### ⚙️ Validating Applicant Data...")
    progress_bar.progress(20)
    time.sleep(0.5)
    
    status_text.markdown("#### 🧠 Running Feature Engineering...")
    progress_bar.progress(40)
    time.sleep(0.5)
    
    status_text.markdown("#### 🤖 RL Ensemble Inference (PPO, DQN, DDQN, SAC)...")
    progress_bar.progress(70)
    
    payload = {
        "age": age,
        "gender": gender,
        "employment": employment,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "existing_debt": existing_debt,
        "loan_tenure": loan_tenure,
        "repayment_history": repayment_history,
        "loan_purpose": loan_purpose,
        "region": region,
        "collateral": collateral,
        "existing_loans": existing_loans,
        "education": education
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        status_text.markdown("#### 🦙 Generating AI Explanations...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        if response.status_code == 200:
            result = response.json()
            progress_bar.progress(100)
            status_text.empty()
            
            # Extract main recommendation
            rec = result.get('recommended_pricing', {})
            risk = result.get('risk_analysis', {})
            rate = rec.get('recommended_interest_rate')
            best_model = rec.get('best_model')
            conf = risk.get('confidence', 0)
            
            status_color = "#27AE60" if rate else "#E74C3C"
            decision_text = f"APPROVED @ {rate}%" if rate else "REJECTED"
            
            st.success("✅ Application successfully processed by the AI Engine.")
            
            st.markdown(f"""
            <div style="background:white; padding:30px; border-radius:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08); text-align:center; border-top: 5px solid {status_color};">
                <h3 style="color:#828282; margin:0;">AI Recommendation</h3>
                <h1 style="color:{status_color}; font-size:3.5rem; margin:10px 0;">{decision_text}</h1>
                <p style="font-size:1.1rem; color:#4F4F4F;">Confidence Score: <b>{round(conf*100, 1)}%</b> &nbsp;|&nbsp; Primary Engine: <b>{best_model}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Ensemble Decision Breakdown")
            
            rl = result.get('rl_predictions', {})
            c1, c2, c3, c4 = st.columns(4)
            c1.info(f"**PPO**: {rl.get('PPO')}")
            c2.info(f"**DQN**: {rl.get('DQN')}")
            c3.info(f"**DDQN**: {rl.get('DDQN')}")
            c4.info(f"**SAC**: {rl.get('SAC')}")
            
            st.markdown("### 🔍 Risk Analysis")
            r1, r2, r3 = st.columns(3)
            r1.metric("Calculated Risk Score", risk.get('risk_score', 'N/A'))
            r2.metric("Risk Level", risk.get('risk_level', 'N/A'))
            r3.metric("Expected Profit", f"${rec.get('expected_profit', 0)}")
            
            st.info("ℹ️ **Next Steps:** This application has been queued for Human-in-the-Loop review by a Loan Officer.")
            
        else:
            status_text.error(f"API Error: {response.text}")
    except Exception as e:
        status_text.error(f"Could not connect to the Backend API. Make sure it is running. Error: {e}")
