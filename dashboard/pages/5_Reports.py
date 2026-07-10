import streamlit as st
import requests
import os
import pandas as pd
from io import BytesIO
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

st.set_page_config(page_title="Reports & Exports", page_icon="📄", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

API_URL = "http://127.0.0.1:8000/api"

st.markdown("<h2 style='color:#1F4E79;'>📄 Generate Reports</h2>", unsafe_allow_html=True)
st.markdown("Download comprehensive portfolio and decision reports in PDF, CSV, or Excel formats.")

def fetch_data():
    try:
        apps = requests.get(f"{API_URL}/applications").json()
        decs = requests.get(f"{API_URL}/loan-decisions").json()
        return apps, decs
    except:
        return [], []

def create_excel(df):
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Loan Data')
    except ImportError:
        # Fallback to csv if openpyxl is not installed
        df.to_csv(output, index=False)
    return output.getvalue()

def create_pdf(df):
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="MIFOS X - Dynamic Loan Pricing Report", ln=1, align='C')
    pdf.ln(10)
    
    for i, row in df.iterrows():
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(200, 8, txt=f"Application ID: {row.get('Application ID', 'N/A')}", ln=1)
        pdf.set_font("Arial", '', 10)
        details = f"Amount: ${row.get('Loan Amount', 0)} | Income: ${row.get('Income', 0)} | Risk: {row.get('Risk Score', 'N/A')}"
        pdf.cell(200, 8, txt=details, ln=1)
        decision = f"Decision: {row.get('Decision', 'Pending')} | Rate: {row.get('Interest Rate', 'N/A')}%"
        pdf.cell(200, 8, txt=decision, ln=1)
        pdf.ln(5)
    
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')

with st.container():
    st.markdown("<div class='mifos-container'>", unsafe_allow_html=True)
    apps, decs = fetch_data()
    
    if apps and apps != "NILL":
        df_list = []
        for a in apps:
            raw_pred = a.get("prediction", {})
            pred = raw_pred if isinstance(raw_pred, dict) else {}
            
            raw_dec = a.get("decision", {})
            dec = raw_dec if isinstance(raw_dec, dict) else {}
            
            status = 'Pending'
            if dec.get('approved') == True: status = 'Approved'
            elif dec.get('approved') == False: status = 'Rejected'
            
            df_list.append({
                "Application ID": a.get("id"),
                "Age": a.get("age"),
                "Income": a.get("income"),
                "Credit Score": a.get("credit_score"),
                "Loan Amount": a.get("loan_amount"),
                "Purpose": a.get("loan_purpose"),
                "Risk Score": pred.get("risk_score", "N/A"),
                "Recommended Model": pred.get("best_model", "N/A"),
                "Interest Rate": pred.get("recommended_rate", "N/A"),
                "Decision": status,
                "Officer": dec.get("officer_name", "N/A"),
                "Timestamp": a.get("timestamp")
            })
            
        df = pd.DataFrame(df_list)
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Export Options")
        col1, col2, col3 = st.columns(3)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        excel_data = create_excel(df)
        pdf_data = create_pdf(df)
        
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="loan_portfolio_report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col2:
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name="loan_portfolio_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        with col3:
            if pdf_data:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_data,
                    file_name="loan_portfolio_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("PDF generation requires 'fpdf' library.")
                
    else:
        st.info("No application data available to generate reports.")
    st.markdown("</div>", unsafe_allow_html=True)
