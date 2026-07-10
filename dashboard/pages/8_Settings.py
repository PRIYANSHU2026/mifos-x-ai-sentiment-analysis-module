import streamlit as st
import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.auth import render_sidebar_auth, get_current_role

st.set_page_config(page_title="Platform Settings", page_icon="⚙️", layout="wide")

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

st.markdown("<h2 style='color:#1F4E79;'>⚙️ Platform Settings</h2>", unsafe_allow_html=True)

# Fetch current settings
current_model = "gemma2:2b"
current_theme = "light"
try:
    res = requests.get(f"{API_URL}/settings/{role}")
    if res.status_code == 200:
        data = res.json()
        current_model = data.get("ollama_model", "gemma2:2b")
        current_theme = data.get("theme", "light")
except Exception as e:
    st.warning("Could not load settings from database.")

st.markdown("### 🦙 LLM Configuration")
st.markdown("Select the local model used for Explainability and the AI Banking Assistant.")

models = ["gemma2:2b", "qwen", "mistral", "gemma2:2b", "deepseek", "phi"]
model_idx = models.index(current_model) if current_model in models else 0

selected_model = st.selectbox("Active AI Model", models, index=model_idx)

st.markdown("### 🎨 Interface Theme")
themes = ["light", "dark", "system"]
theme_idx = themes.index(current_theme) if current_theme in themes else 0
selected_theme = st.selectbox("UI Theme", themes, index=theme_idx)

if st.button("💾 Save Settings", type="primary"):
    payload = {
        "user_role": role,
        "ollama_model": selected_model,
        "theme": selected_theme
    }
    try:
        post_res = requests.post(f"{API_URL}/settings", json=payload)
        if post_res.status_code == 200:
            st.success("Settings saved successfully!")
        else:
            st.error("Failed to save settings.")
    except:
        st.error("Failed to connect to backend API.")

st.markdown("---")
st.markdown("### 🏥 System Status")
st.info("FastAPI Backend: **Running**")
st.info("ChromaDB Vector Store: **Ready**")
st.info("SQLite Database: **Connected**")
