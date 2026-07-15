import streamlit as st
import requests
import sys
import os

# Ensure the parent directory is in the python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.auth import render_sidebar_auth, get_current_role, api_request, api_request

st.set_page_config(page_title="MIFOS X AI Engine", page_icon="🏦", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

API_URL = "http://127.0.0.1:8000/api"

# ─── Sidebar Authentication ──────────────────────────────────────────
st.sidebar.markdown("### 🏦 **MIFOS X** AI Platform")

if "token" not in st.session_state:
    st.markdown("## 🔐 Login to MIFOS X")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                res = api_request("POST", "token", data={"username": username, "password": password})
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["username"] = username
                    
                    user_res = api_request("GET", "users/me", headers={"Authorization": f"Bearer {data['access_token']}"})
                    if user_res.status_code == 200:
                        st.session_state["current_role"] = user_res.json()["role"]
                        st.rerun()
                else:
                    st.error("Invalid credentials")
                    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["Customer", "Loan Officer", "Administrator"])
            reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                res = api_request("POST", "users/register", json={"username": new_username, "password": new_password, "role": new_role})
                if res.status_code == 200:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(f"Registration failed: {res.text}")
                    
    st.stop()

render_sidebar_auth()
role = get_current_role()
st.sidebar.markdown(f"**Welcome, {role}**")

# ─── Home Dashboard Header ────────────────────────────────────────
st.markdown("<div class='mifos-header'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='color:#1F4E79;'>Enterprise AI Banking Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#828282; font-size: 1.1rem;'>Dynamic Micro-Loan Pricing Engine powered by Reinforcement Learning</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Role-Based Views ──────────────────────────────────────────────
if role == "Customer":
    st.info("👋 Welcome! Please navigate to **Loan Processing** in the sidebar to apply for a loan.")
    st.markdown("### How it works")
    st.markdown("""
    1. **Apply:** Enter your details and upload documents.
    2. **AI Analysis:** Our RL engine determines the best personalized interest rate.
    3. **Approval:** A loan officer reviews the AI recommendation.
    """)
    
else:
    # ─── KPI Cards (For Admin/Officer/Analyst) ──────────────────────
    try:
        res = api_request("GET", "/dashboard")
        if res.status_code == 200:
            data = res.json()
            
            st.markdown("### 📊 Today's Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Applications", data.get('total_applications', 0))
            c2.metric("Approval Rate", f"{data.get('approval_rate_pct', 0)}%")
            c3.metric("Portfolio Value", f"${data.get('total_portfolio_value', 0):,}")
            c4.metric("Active Model", data.get('best_model', 'N/A'))
            
            st.markdown("<br>", unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Avg Interest Rate", f"{data.get('avg_interest_rate', 0)}%")
            c6.metric("Default Probability", "12.4%") # Mocked for now
            c7.metric("AI Confidence", "89%") # Mocked for now
            c8.metric("System Health", "Optimal 🟢")

    except Exception as e:
        st.warning("Backend API is currently unreachable. Start the backend to view live metrics.")
        st.error(str(e))

    # ─── Animated Workflow Visualization ─────────────────────────────
    st.markdown("### 🔄 AI Decision Pipeline")
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:white; padding:20px; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.05); margin-top:20px;">
        <div style="text-align:center;">
            <div style="font-size:2rem;">📄</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">App Data</div>
        </div>
        <div style="color:#2F80ED; font-size:1.5rem;">➔</div>
        <div style="text-align:center;">
            <div style="font-size:2rem;">⚙️</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">Feature Eng</div>
        </div>
        <div style="color:#2F80ED; font-size:1.5rem;">➔</div>
        <div style="text-align:center;">
            <div style="font-size:2rem;">🧠</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">RL Ensemble</div>
        </div>
        <div style="color:#2F80ED; font-size:1.5rem;">➔</div>
        <div style="text-align:center;">
            <div style="font-size:2rem;">🦙</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">Ollama Explain</div>
        </div>
        <div style="color:#2F80ED; font-size:1.5rem;">➔</div>
        <div style="text-align:center;">
            <div style="font-size:2rem;">🧑‍💼</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">HITL Review</div>
        </div>
        <div style="color:#2F80ED; font-size:1.5rem;">➔</div>
        <div style="text-align:center;">
            <div style="font-size:2rem;">✅</div>
            <div style="font-size:0.9rem; font-weight:bold; color:#1F4E79;">Approval</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
