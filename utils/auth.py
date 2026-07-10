import streamlit as st

ROLES = ["Customer", "Loan Officer", "Risk Analyst", "Administrator"]

def require_role(allowed_roles: list):
    """Decorator or function to check if the current user has access to a page."""
    current_role = st.session_state.get('current_role', "Customer")
    if current_role not in allowed_roles:
        st.error(f"Access Denied. This page requires one of the following roles: {', '.join(allowed_roles)}")
        st.stop()

def render_sidebar_auth():
    """Renders the role switcher in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 User Session")
    
    current = st.session_state.get('current_role', "Customer")
    idx = ROLES.index(current) if current in ROLES else 0
    
    selected_role = st.sidebar.selectbox("Active Role", ROLES, index=idx)
    
    if selected_role != current:
        st.session_state['current_role'] = selected_role
        st.rerun()

def get_current_role():
    return st.session_state.get('current_role', "Customer")
