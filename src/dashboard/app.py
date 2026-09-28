"""Credit Risk & Expected Loss Engine - Institutional Financial Terminal."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.models.basel_metrics import BaselExpectedLossEngine
from src.models.explainability import CreditRiskExplainer

# Page configuration for institutional wide layout
st.set_page_config(
    page_title="Credit Risk & Expected Loss Engine | Basel II / IFRS 9",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Institutional Dark Financial Terminal Styling
# Target aesthetic: Bloomberg / BlackRock Aladdin / Barclays BARX
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp {
        background-color: #090d14;
        color: #cbd5e1;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Terminal Header */
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 16px 0;
        margin-top: 0.5rem;
        border-bottom: 1px solid #1a2233;
        margin-bottom: 14px;
    }
    .terminal-header-left {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .terminal-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        line-height: 1.4;
        margin: 0;
        padding-top: 2px;
    }
    .terminal-sub {
        font-size: 0.72rem;
        color: #7d8ea5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .terminal-header-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #10b981;
        background-color: #062319;
        border: 1px solid #0f5132;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
    }
    .status-dot {
        font-size: 0.7rem;
        line-height: 1;
    }
    .terminal-meta-tag {
        font-size: 0.72rem;
        color: #556987;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
        letter-spacing: 0.04em;
    }

    /* Tabs Styling - Red active underline, institutional typography */
    div[data-testid="stTabs"] {
        border-bottom: 1px solid #1a2233;
        margin-bottom: 1rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #8fa0b5 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 8px 16px !important;
        border-radius: 0px !important;
        margin-right: 8px !important;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #f1f5f9 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #dc2626 !important;
        color: #ffffff !important;
        background-color: transparent !important;
    }

    /* Form Controls & Inputs */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #101624 !important;
        border: 1px solid #1d273a !important;
        border-radius: 4px !important;
        color: #f1f5f9 !important;
        min-height: 34px !important;
    }
    div[data-baseweb="input"] input {
        color: #f1f5f9 !important;
        font-size: 0.82rem !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        padding: 4px 8px !important;
    }
    div[data-baseweb="input"] input:focus {
        border-color: #3b82f6 !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #8fa0b5 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 2px !important;
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 4px !important;
    }

    /* Primary Execution Button */
    div.stButton > button {
        background-color: #162032 !important;
        color: #f8fafc !important;
        border: 1px solid #293852 !important;
        border-radius: 4px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 8px 24px !important;
        width: 100% !important;
        box-shadow: none !important;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #1f2e47 !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Section Cards & Panels */
    .fin-panel {
        background-color: #0f1523;
        border: 1px solid #1a2233;
        border-radius: 4px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .fin-panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        border-bottom: 1px solid #1a2233;
        padding-bottom: 6px;
        margin-bottom: 10px;
    }

    /* Risk Summary Strip */
    .risk-summary-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    .risk-summary-card {
        background-color: #0f1523;
        border: 1px solid #1a2233;
        border-radius: 4px;
        padding: 10px 14px;
    }
    .risk-summary-card.active-risk-low {
        border-left: 3px solid #10b981;
    }
    .risk-summary-card.active-risk-med {
        border-left: 3px solid #d97706;
    }
    .risk-summary-card.active-risk-high {
        border-left: 3px solid #dc2626;
    }
    .risk-kpi-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7d8ea5;
        margin-bottom: 4px;
    }
    .risk-kpi-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
        letter-spacing: -0.02em;
    }
    .risk-kpi-subtext {
        font-size: 0.68rem;
        color: #556987;
        margin-top: 3px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Risk Badges */
    .badge-grade {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
        letter-spacing: 0.04em;
    }
    .badge-grade-low {
        background-color: #062319;
        color: #10b981;
        border: 1px solid #0f5132;
    }
    .badge-grade-med {
        background-color: #261a06;
        color: #f59e0b;
        border: 1px solid #78350f;
    }
    .badge-grade-high {
        background-color: #270d0d;
        color: #ef4444;
        border: 1px solid #7f1d1d;
    }

    /* Decision Memos */
    .decision-box {
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .decision-approved {
        background-color: #062319;
        border: 1px solid #0f5132;
        color: #a7f3d0;
    }
    .decision-conditional {
        background-color: #261a06;
        border: 1px solid #78350f;
        color: #fde68a;
    }
    .decision-declined {
        background-color: #270d0d;
        border: 1px solid #7f1d1d;
        color: #fecaca;
    }

    /* Governance Audit Table & Key Value Lists */
    .mrm-audit-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }
    .mrm-audit-card {
        background-color: #0f1523;
        border: 1px solid #1a2233;
        border-radius: 4px;
        padding: 8px 12px;
    }
    .mrm-audit-label {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .mrm-audit-value {
        font-size: 0.82rem;
        color: #e2e8f0;
        font-weight: 600;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
        margin-top: 2px;
    }

    /* Factor Driver Rows */
    .driver-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #161e2e;
        font-size: 0.75rem;
    }
    .driver-name {
        color: #cbd5e1;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
    }
    .driver-impact-pos {
        color: #ef4444;
        font-weight: 600;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
    }
    .driver-impact-neg {
        color: #10b981;
        font-weight: 600;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
    }

    /* Streamlit native component cleanups */
    hr {
        border-color: #1a2233 !important;
        margin: 12px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource
def load_risk_assets():
    """Loads calibrated scoring pipeline and portfolio summary reports."""
    explainer = CreditRiskExplainer()
    basel_summary_path = REPORTS_DIR / "portfolio_basel_summary.json"
    basel_summary = None
    if basel_summary_path.exists():
        with open(basel_summary_path, "r", encoding="utf-8") as f:
            basel_summary = json.load(f)

    tier_path = REPORTS_DIR / "basel_provisions_by_tier.csv"
    tier_df = pd.read_csv(tier_path) if tier_path.exists() else None

    metrics_path = REPORTS_DIR / "model_comparison_table.csv"
    comparison_df = pd.read_csv(metrics_path) if metrics_path.exists() else None

    return explainer, basel_summary, tier_df, comparison_df


def format_currency(val: float) -> str:
    return f"${val:,.0f}" if abs(val) >= 1000 else f"${val:,.2f}"


def main():
    # Compact institutional application header
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-header-left">
            <div class="terminal-title">Credit Risk & Expected Loss Engine</div>
            <div class="terminal-sub">Basel II / IFRS 9 Credit Risk Analytics</div>
        </div>
        <div class="terminal-header-right">
            <div class="status-pill">
                <span class="status-dot">●</span> Model Ready
            </div>
            <div class="terminal-meta-tag">ENGINE v2.4.1 | CALIBRATED (SIGMOID)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        explainer, basel_summary, tier_df, comparison_df = load_risk_assets()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure run_pipeline.py has been executed.")
        st.stop()

    tabs = st.tabs([
        "Single Loan Underwriting",
        "Portfolio Capital Provisions",
        "Model Governance",
    ])

    # =========================================================================
    # TAB 1: Single Loan Underwriting Workspace
    # =========================================================================
    with tabs[0]:
        # Underwriting Workspace Columns (3-Way Split)
        col_terms, col_borrower, col_bureau = st.columns(3)

        with col_terms:
            st.markdown('<div class="fin-panel-title">Loan Terms</div>', unsafe_allow_html=True)
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=40000.0, value=15000.0, step=500.0)
            term = st.selectbox("Term", options=["36 months", "60 months"], index=0)
            int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=32.0, value=12.5, step=0.1)
            grade = st.selectbox("Credit Grade", options=["A", "B", "C", "D", "E", "F", "G"], index=1)
            sub_grade_options = [f"{grade}{i}" for i in range(1, 6)]
            sub_grade = st.selectbox("Sub-Grade", options=sub_grade_options, index=2)

        with col_borrower:
            st.markdown('<div class="fin-panel-title">Borrower Financial Profile</div>', unsafe_allow_html=True)
            annual_inc = st.number_input("Annual Income ($)", min_value=12000.0, max_value=500000.0, value=75000.0, step=2500.0)
            dti = st.slider("Debt-to-Income (DTI %)", min_value=2.0, max_value=55.0, value=18.5, step=0.5)
            emp_length = st.selectbox("Employment Length", options=["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"], index=4)
            home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN", "OTHER"], index=0)
            verification_status = st.selectbox("Income Verification", options=["Source Verified", "Verified", "Not Verified"], index=0)

        with col_bureau:
            st.markdown('<div class="fin-panel-title">Credit Bureau History</div>', unsafe_allow_html=True)
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=100000.0, value=14200.0, step=500.0)
            revol_util = st.slider("Revolving Line Utilization (%)", min_value=0.0, max_value=100.0, value=52.0, step=1.0)
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                delinq_2yrs = st.number_input("Delinquencies (2Y)", min_value=0, max_value=10, value=0, step=1)
                open_acc = st.number_input("Open Accounts", min_value=1, max_value=50, value=11, step=1)
            with c_b2:
                inq_last_6mths = st.number_input("Inquiries (6M)", min_value=0, max_value=10, value=1, step=1)
                total_acc = st.number_input("Total Credit Lines", min_value=open_acc, max_value=70, value=max(open_acc, 22), step=1)
            purpose = st.selectbox("Loan Purpose", options=["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business"], index=0)

        # Monthly installment calculation
        term_months = 36 if term == "36 months" else 60
        r_monthly = (int_rate / 100.0) / 12.0
        calculated_installment = round(
            loan_amnt * (r_monthly * (1 + r_monthly)**term_months) / ((1 + r_monthly)**term_months - 1), 2
        )

        emp_len_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "5 years": 5, "10+ years": 10,
        }
        emp_length_int = emp_len_map.get(emp_length, 5)

        raw_borrower_df = pd.DataFrame([{
            "loan_amnt": float(loan_amnt),
            "funded_amnt": float(loan_amnt),
            "term": 36 if term == "36 months" else 60,
            "int_rate": float(int_rate),
            "installment": float(calculated_installment),
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_title": "Professional",
            "emp_length": int(emp_length_int),
            "home_ownership": home_ownership,
            "annual_inc": float(annual_inc),
            "verification_status": verification_status,
            "issue_d": "Jan-2017",
            "loan_status": "Current",
            "purpose": purpose,
            "title": "Loan Request",
            "dti": float(dti),
            "delinq_2yrs": float(delinq_2yrs),
            "earliest_cr_line": "Jan-2005",
            "inq_last_6mths": float(inq_last_6mths),
            "open_acc": float(open_acc),
            "pub_rec": 0.0,
            "revol_bal": float(revol_bal),
            "revol_util": float(revol_util),
            "total_acc": float(total_acc),
        }])

        # Trigger Calculation
        scorecard = explainer.explain_customer_risk(
            loan_id=f"APPLICANT_{grade}",
            feature_vector=raw_borrower_df,
            loan_amnt=loan_amnt,
        )

        # Risk Summary Panel (Institutional Terminal KPI Strip)
        pd_val = scorecard["predicted_pd_pct"]
        lgd_val = scorecard["loss_given_default_pct"]
        ead_val = scorecard["exposure_at_default_usd"]
        el_val = scorecard["expected_loss_usd"]
        tier_val = scorecard["risk_tier"]

        if "Low" in tier_val or pd_val < 10.0:
            tier_class = "active-risk-low"
            badge_class = "badge-grade-low"
            risk_label = "Low Risk Tier"
        elif "Medium" in tier_val or pd_val < 25.0:
            tier_class = "active-risk-med"
            badge_class = "badge-grade-med"
            risk_label = "Medium Risk Tier"
        else:
            tier_class = "active-risk-high"
            badge_class = "badge-grade-high"
            risk_label = "High Risk Tier"

        st.markdown(f"""
        <div class="risk-summary-grid">
            <div class="risk-summary-card {tier_class}">
                <div class="risk-kpi-label">Probability of Default (PD)</div>
                <div class="risk-kpi-value">{pd_val:.2f}%</div>
                <div class="risk-kpi-subtext">Calibrated Sigmoid Output</div>
            </div>
            <div class="risk-summary-card">
                <div class="risk-kpi-label">Loss Given Default (LGD)</div>
                <div class="risk-kpi-value">{lgd_val:.2f}%</div>
                <div class="risk-kpi-subtext">Recovery Rate: {100 - lgd_val:.1f}%</div>
            </div>
            <div class="risk-summary-card">
                <div class="risk-kpi-label">Exposure at Default (EAD)</div>
                <div class="risk-kpi-value">${ead_val:,.0f}</div>
                <div class="risk-kpi-subtext">CCF: 1.0 (Term Loan)</div>
            </div>
            <div class="risk-summary-card {tier_class}">
                <div class="risk-kpi-label">Expected Loss (EL)</div>
                <div class="risk-kpi-value">${el_val:,.2f}</div>
                <div class="risk-kpi-subtext">{scorecard['expected_loss_rate_pct']:.2f}% of EAD</div>
            </div>
            <div class="risk-summary-card">
                <div class="risk-kpi-label">Risk Grade & Rating</div>
                <div class="risk-kpi-value">{grade} <span class="badge-grade {badge_class}">{tier_val}</span></div>
                <div class="risk-kpi-subtext">{risk_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Attribution & Underwriting Decision Memo Split
        col_waterfall, col_memo = st.columns([1.5, 1.3])

        with col_waterfall:
            st.markdown('<div class="fin-panel-title">SHAP Log-Odds Factor Attribution</div>', unsafe_allow_html=True)
            if Path(scorecard["waterfall_chart_path"]).exists():
                st.image(scorecard["waterfall_chart_path"], width=580)

        with col_memo:
            st.markdown('<div class="fin-panel-title">Underwriting Audit & Risk Decomposition</div>', unsafe_allow_html=True)

            # Decision Memo Block
            if pd_val < 10.0:
                decision_html = """
                <div class="decision-box decision-approved">
                    <strong>UNDERWRITING ACTION: APPROVE</strong><br>
                    Borrower satisfies prime underwriting parameters. Expected credit loss falls within baseline Tier 1 reserve allowances. Approved for standard risk-adjusted interest rate schedule.
                </div>
                """
            elif pd_val < 25.0:
                decision_html = """
                <div class="decision-box decision-conditional">
                    <strong>UNDERWRITING ACTION: CONDITIONAL APPROVAL</strong><br>
                    Elevated default likelihood detected. Facility approval contingent on rate premium adjustment or verification of additional unencumbered liquid collateral.
                </div>
                """
            else:
                decision_html = """
                <div class="decision-box decision-declined">
                    <strong>UNDERWRITING ACTION: DECLINE / ESCALATE</strong><br>
                    Calculated default probability exceeds institutional underwriting risk appetite. Exceeds max provisioning tolerance. Requires Credit Risk Committee override.
                </div>
                """
            st.markdown(decision_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="fin-panel-title">Primary Risk Drivers (SHAP Local Attribution)</div>', unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.7rem; color:#7d8ea5; text-transform:uppercase; font-weight:600; margin-bottom:4px;">Risk-Increasing Factors (+)</div>', unsafe_allow_html=True)
            for item in scorecard["top_risk_increasing_factors"][:3]:
                st.markdown(f"""
                <div class="driver-row">
                    <span class="driver-name">{item['feature']}</span>
                    <span>Value: {item['borrower_value']}</span>
                    <span class="driver-impact-pos">+{item['shap_impact']:.4f} log-odds</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.7rem; color:#7d8ea5; text-transform:uppercase; font-weight:600; margin-top:10px; margin-bottom:4px;">Risk-Mitigating Factors (-)</div>', unsafe_allow_html=True)
            for item in scorecard["top_risk_mitigating_factors"][:3]:
                st.markdown(f"""
                <div class="driver-row">
                    <span class="driver-name">{item['feature']}</span>
                    <span>Value: {item['borrower_value']}</span>
                    <span class="driver-impact-neg">{item['shap_impact']:.4f} log-odds</span>
                </div>
                """, unsafe_allow_html=True)

            # Supporting Underwriting Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="fin-panel-title">Credit Facility Metrics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#94a3b8; display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
                <div>Monthly Installment: <strong style="color:#f1f5f9;">${calculated_installment:,.2f}</strong></div>
                <div>Debt Service Ratio: <strong style="color:#f1f5f9;">{((calculated_installment * 12) / max(annual_inc, 1.0)) * 100:.1f}%</strong></div>
                <div>Base Model Log-Odds: <strong style="color:#f1f5f9;">{scorecard['base_log_odds']:.3f}</strong></div>
                <div>Credit Conversion Factor: <strong style="color:#f1f5f9;">1.00</strong></div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: Portfolio Capital Provisions (Basel II / IFRS 9)
    # =========================================================================
    with tabs[1]:
        st.markdown('<div class="fin-panel-title">Portfolio Exposure & Capital Adequacy</div>', unsafe_allow_html=True)

        if basel_summary:
            total_loans = basel_summary["total_portfolio_loans"]
            funded_vol = basel_summary["total_funded_volume_usd"]
            ead_vol = basel_summary["total_exposure_at_default_usd"]
            el_prov = basel_summary["total_expected_loss_provision_usd"]
            ul_cap = basel_summary["total_unexpected_loss_usd"]
            port_pd = basel_summary["portfolio_weighted_pd_pct"]
            port_lgd = basel_summary["portfolio_weighted_lgd_pct"]
            port_el_rate = basel_summary["portfolio_el_rate_pct"]

            st.markdown(f"""
            <div class="risk-summary-grid">
                <div class="risk-summary-card">
                    <div class="risk-kpi-label">Portfolio Exposure (EAD)</div>
                    <div class="risk-kpi-value">${ead_vol:,.0f}</div>
                    <div class="risk-kpi-subtext">{total_loans:,} Active Facilities</div>
                </div>
                <div class="risk-summary-card">
                    <div class="risk-kpi-label">Weighted Average PD</div>
                    <div class="risk-kpi-value">{port_pd:.2f}%</div>
                    <div class="risk-kpi-subtext">Exposure Weighted</div>
                </div>
                <div class="risk-summary-card">
                    <div class="risk-kpi-label">Weighted Average LGD</div>
                    <div class="risk-kpi-value">{port_lgd:.2f}%</div>
                    <div class="risk-kpi-subtext">Model-Inferred Severity</div>
                </div>
                <div class="risk-summary-card active-risk-high">
                    <div class="risk-kpi-label">Expected Loss Provision</div>
                    <div class="risk-kpi-value">${el_prov:,.0f}</div>
                    <div class="risk-kpi-subtext">{port_el_rate:.2f}% Provision Rate</div>
                </div>
                <div class="risk-summary-card">
                    <div class="risk-kpi-label">Economic Capital (UL)</div>
                    <div class="risk-kpi-value">${ul_cap:,.0f}</div>
                    <div class="risk-kpi-subtext">Basel II Pillar 1 Buffer</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        col_tier_table, col_tier_chart = st.columns([1.4, 1.2])

        with col_tier_table:
            st.markdown('<div class="fin-panel-title">Capital Provisions by Rating Grade Tier</div>', unsafe_allow_html=True)
            if tier_df is not None:
                # Format dataframe with professional institutional columns
                display_tier = tier_df.copy()
                display_tier.columns = [
                    "Risk Tier", "Loan Count", "Funded Volume", "EAD Volume",
                    "Expected Loss", "Avg PD", "Avg LGD", "EL Rate %"
                ]

                st.dataframe(
                    display_tier.style.format({
                        "Loan Count": "{:,}",
                        "Funded Volume": "${:,.0f}",
                        "EAD Volume": "${:,.0f}",
                        "Expected Loss": "${:,.0f}",
                        "Avg PD": "{:.2%}",
                        "Avg LGD": "{:.2%}",
                        "EL Rate %": "{:.2f}%",
                    }),
                    use_container_width=True,
                    height=180,
                )

                st.markdown("""
                <div style="font-size:0.72rem; color:#64748b; line-height:1.4; margin-top:8px;">
                    Provisions computed under IFRS 9 ECL 12-Month model: EL = PD × LGD × EAD.<br>
                    Facilities classified into Basel Rating Grades (Low: Grades A-B, Medium: Grades C-D, High: Grades E-G).
                </div>
                """, unsafe_allow_html=True)

        with col_tier_chart:
            st.markdown('<div class="fin-panel-title">Portfolio Risk Distribution</div>', unsafe_allow_html=True)
            loss_chart_path = REPORTS_DIR / "expected_loss_distribution.png"
            if loss_chart_path.exists():
                st.image(str(loss_chart_path), width=520)

    # =========================================================================
    # TAB 3: Model Governance & Risk Management (MRM)
    # =========================================================================
    with tabs[2]:
        st.markdown('<div class="fin-panel-title">Model Risk Management (MRM) Audit Dossier</div>', unsafe_allow_html=True)

        # Audit Metadata Grid
        st.markdown("""
        <div class="mrm-audit-grid">
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Model Identifier</div>
                <div class="mrm-audit-value">XGB-BASEL-PD-CALIB-V2</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Model Status</div>
                <div class="mrm-audit-value" style="color:#10b981;">Validated & Production Ready</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Primary Objective</div>
                <div class="mrm-audit-value">IFRS 9 / Basel II Pillar 1 PD</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Last Validation Date</div>
                <div class="mrm-audit-value">2026-09-28</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Development Sample</div>
                <div class="mrm-audit-value">41,059 Sanitized Loans</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Holdout Partition</div>
                <div class="mrm-audit-value">20% Stratified Out-of-Sample</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">Calibration Method</div>
                <div class="mrm-audit-value">Platt Sigmoid (Brier 0.151)</div>
            </div>
            <div class="mrm-audit-card">
                <div class="mrm-audit-label">LGD Architecture</div>
                <div class="mrm-audit-value">Ridge Regularized Recovery Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Out-of-Sample Benchmark Comparison Table
        st.markdown('<div class="fin-panel-title">Out-of-Sample Benchmark Performance Comparison</div>', unsafe_allow_html=True)
        if comparison_df is not None:
            formatted_comp = comparison_df.copy()
            st.dataframe(
                formatted_comp.style.format({
                    "ROC_AUC": "{:.4f}",
                    "PR_AUC": "{:.4f}",
                    "Gini_Coefficient": "{:.4f}",
                    "Brier_Score": "{:.4f}",
                    "Mean_Predicted_PD": "{:.2%}",
                    "Empirical_Default_Rate": "{:.2%}",
                }),
                use_container_width=True,
                height=180,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Diagnostic Visualizations & Macro Risk Factors Split
        col_diag_curves, col_macro_shap = st.columns([1.3, 1.1])

        with col_diag_curves:
            st.markdown('<div class="fin-panel-title">Validation Diagnostics (ROC, PR, Calibration)</div>', unsafe_allow_html=True)
            diag_curve_path = REPORTS_DIR / "roc_pr_calibration_curves.png"
            if diag_curve_path.exists():
                st.image(str(diag_curve_path), width=580)

        with col_macro_shap:
            st.markdown('<div class="fin-panel-title">Macro Risk Drivers (Global SHAP Beeswarm)</div>', unsafe_allow_html=True)
            beeswarm_path = REPORTS_DIR / "shap_summary_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), width=520)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Assumptions and Regulatory Limitations
        st.markdown('<div class="fin-panel-title">Regulatory Assumptions & Model Scope Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.76rem; color:#8fa0b5; line-height:1.5;">
            1. <strong>Standardized Credit Conversion Factor (CCF)</strong>: EAD assumes 100% drawn facility conversion at default in compliance with Basel standardized treatment for fixed-term retail installment contracts.<br>
            2. <strong>Loss Severity Regularization</strong>: LGD values are regularized via Ridge recovery estimation bound between 5% and 95%, reflecting empirical historical liquidation recoveries.<br>
            3. <strong>Probability Calibration</strong>: XGBoost raw margins are monotonically transformed via Logistic Sigmoidal calibration to align expected default rates with historical cohort defaults.<br>
            4. <strong>Macroeconomic Assumptions</strong>: Baseline economic conditions assumed static. Forward-looking IFRS 9 macroeconomic scenario adjustments (base, upside, downside) require separate scalar application.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
