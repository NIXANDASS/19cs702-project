import streamlit as st
import plotly.graph_objects as go
import pandas as pd
# --- Import the ML-driven scoring service ---
from scoring_service import predict_score

st.set_page_config(
    page_title="Pro Loan Approval Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- CSS STYLING ---
def load_css():
    common_css = """
    <style>
        .stApp { background-color: var(--bg-color); }
        h1, h3, p, li, .st-emotion-cache-1r702v p { color: var(--text-color) !important; }
        .st-emotion-cache-16txtl3 { background-color: var(--secondary-bg-color); }
        .st-emotion-cache-1v0mbdj { border-color: var(--border-color); }
    </style>
    """
    if theme == "Dark":
        theme_css = """
        <style>
            :root {
                --bg-color: #0E1117; --secondary-bg-color: #262730;
                --text-color: #FAFAFA; --border-color: #31333F;
            }
        </style>
        """
    else: # Light Theme
        theme_css = """
        <style>
            :root {
                --bg-color: #FFFFFF; --secondary-bg-color: #F0F2F6;
                --text-color: #31333F; --border-color: #CCCCCC;
            }
        </style>
        """
    st.markdown(theme_css, unsafe_allow_html=True)
    st.markdown(common_css, unsafe_allow_html=True)

# --- RISK BAND LOGIC ---
def determine_risk_band(score, monthly_income):
    # This logic remains as it is a business rule, not part of the ML prediction
    risk = "Low Risk" if score >= 650 else "High Risk"
    need = "High Need" if monthly_income < 25000 else "Low Need"
    return f"{risk} – {need}"

# --- SIDEBAR INPUTS ---
st.sidebar.header("Display Options")
theme = st.sidebar.radio("Choose a Theme", ["Dark", "Light"])

st.sidebar.header("Beneficiary Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)

# --- Financial Inputs ---
income = st.sidebar.slider("Monthly Income (RS)", min_value=5000, max_value=100000, value=25000, step=5000)
loan_amount = st.sidebar.slider("Requested Loan Amount (RS)", min_value=10000, max_value=2000000, value=400000, step=50000)

loan_tenure = st.sidebar.selectbox("Loan Tenure (months)", [6, 12, 18, 24, 36], index=2)
repayment_history = st.sidebar.selectbox("Repayment History", ["Good", "Average", "Poor"], index=0)

total_credit_limit = st.sidebar.number_input("Total Credit Limit (RS)", min_value=0, max_value=500000, value=100000, step=10000)
outstanding_debt = st.sidebar.number_input("Current Outstanding Debt (RS)", min_value=0, max_value=500000, value=5000, step=1000)
# --- END MODIFIED SECTION ---

# --- Apply CSS ---
load_css()

# --- CALCULATIONS (NOW USING ML SERVICE) ---
try:
    # 1. Call the ML prediction function imported from scoring_service.py
    credit_score, score_breakdown = predict_score(
        age=age, 
        income=income, 
        loan_amount=loan_amount, 
        loan_tenure=loan_tenure, 
        repayment_history=repayment_history, 
        credit_limit=total_credit_limit, 
        debt=outstanding_debt, 
        consumption_data=None # Placeholder
    )
except Exception as e:
    # Fallback if the scoring service fails after loading
    st.error(f"Error calling scoring service: {e}. Defaulting to score 300.")
    credit_score = 300
    score_breakdown = {'Score Source': 'Error', 'Details': str(e)}

loan_eligibility = "Approved" if credit_score >= 650 else "Rejected"
risk_band = determine_risk_band(credit_score, income)

# --- MAIN DASHBOARD LAYOUT ---
st.markdown(f'<h1 style="color: var(--text-color);">Beneficiary Loan Dashboard</h1>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([2, 3])

    with col1: # Gauge Chart
        gauge_font_color = "#FAFAFA" if theme == "Dark" else "#31333F"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=credit_score,
            title={'text': "Credit Score", 'font': {'size': 24}},
            gauge={'axis': {'range': [300, 850]}, 'bar': {'color': "#28a745" if loan_eligibility == "Approved" else "#dc3545"}}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': gauge_font_color}, height=300, margin=dict(l=20, r=20, t=50, b=20))
        # use_container_width=True is kept for Plotly chart as it's generally safe
        st.plotly_chart(fig, use_container_width=True) 

    with col2: # Main Results
        st.subheader("Loan Assessment")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Risk Band", value=risk_band)
        with res_col2:
            if loan_eligibility == "Approved": st.success("Eligibility: Approved ✔️")
            else: st.error("Eligibility: Rejected ❌")
            
        # Score Breakdown Expander (Now showing ML Model Indicators)
        with st.expander("See Model Breakdown"):
            # Converted to simple Indicator/Value DataFrame. Replaced use_container_width=True
            st.dataframe(pd.DataFrame(score_breakdown.items(), columns=['Indicator', 'Value']), width='stretch')

# --- Additional Details Section ---
st.subheader("Key Financial Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

dti_ratio = "N/A"
if income > 0 and loan_tenure > 0:
    monthly_payment = loan_amount / loan_tenure
    dti_val = (monthly_payment / income) * 100
    dti_ratio = f"{dti_val:.2f}%"

cur_ratio = "N/A"
if total_credit_limit > 0:
    cur_val = (outstanding_debt / total_credit_limit) * 100
    cur_ratio = f"{cur_val:.2f}%"

kpi_col1.metric("Repayment History", repayment_history)
kpi_col2.metric("Debt-to-Income Ratio", dti_ratio)
kpi_col3.metric("Credit Utilization", cur_ratio)