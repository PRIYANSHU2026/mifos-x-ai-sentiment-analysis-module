import streamlit as st
import requests
import pandas as pd
import os
import sys

# Ensure the parent directory is in the python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.auth import render_sidebar_auth, get_current_role

st.set_page_config(page_title="Human Review", page_icon="🧑‍💼", layout="wide")

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

if role not in ["Loan Officer", "Administrator"]:
    st.error("Access Denied. Only Loan Officers and Administrators can access the Human Review dashboard.")
    st.stop()

st.markdown("<h2 style='color:#1F4E79;'>🧑‍💼 Human-in-the-Loop Review</h2>", unsafe_allow_html=True)
st.markdown("Review AI predictions and finalize loan decisions.")

# Fetch applications
try:
    apps_res = requests.get(f"{API_URL}/applications")
    if apps_res.status_code == 200:
        data = apps_res.json()
        if data == "NILL" or not data:
            st.info("No pending applications for review.")
        else:
            # We want to display pending ones. For now, assume all returned are pending 
            # if they don't have a final decision (we'll just list the last 10 for review).
            df = pd.DataFrame(data).tail(10)
            
            # Select an application to review
            app_id = st.selectbox("Select Application ID to Review", df['id'].tolist())
            
            selected_app = df[df['id'] == app_id].iloc[0]
            
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📄 Application Details")
                st.json({
                    "Age": selected_app.get("age"),
                    "Income": selected_app.get("income"),
                    "Credit Score": selected_app.get("credit_score"),
                    "Loan Amount": selected_app.get("loan_amount"),
                    "Tenure": selected_app.get("loan_tenure"),
                    "Existing Debt": selected_app.get("existing_debt"),
                    "Purpose": selected_app.get("loan_purpose")
                })
                
                st.markdown("### 🦙 Ollama AI Explanation")
                explanation = selected_app.get("ollama_explanation", {})
                if isinstance(explanation, dict):
                    st.info(f"**Customer Facing:** {explanation.get('customer_friendly_explanation', 'N/A')}")
                    st.warning(f"**Officer Technical:** {explanation.get('officer_technical_explanation', 'N/A')}")
                    st.success(f"**Suggested Improvements:** {explanation.get('suggested_improvements', 'N/A')}")
                else:
                    st.info(explanation)
                
            with col2:
                st.markdown("### 🤖 RL Prediction Details")
                st.json(selected_app.get("predictions", {}))
                
                st.markdown("### ⚖️ Final Decision")
                with st.form("decision_form"):
                    final_decision = st.selectbox("Decision", ["Approve", "Reject"])
                    final_rate = st.number_input("Final Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0)
                    comments = st.text_area("Officer Comments")
                    
                    if st.form_submit_button("💾 Save Decision"):
                        payload = {
                            "application_id": int(app_id),
                            "decision": final_decision,
                            "approved_interest_rate": final_rate if final_decision == "Approve" else None,
                            "officer_comments": comments
                        }
                        dec_res = requests.post(f"{API_URL}/loan-decisions", json=payload)
                        if dec_res.status_code == 200:
                            st.success("Decision saved successfully!")
                        else:
                            st.error("Failed to save decision.")
    else:
        st.error("Failed to fetch applications.")
except Exception as e:
    st.error(f"Error connecting to backend: {e}")
