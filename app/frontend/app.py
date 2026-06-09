"""
Streamlit dashboard for the Multi-Agent Financial Analyst.

Run with:  streamlit run app/frontend/app.py
"""

import json
import os
import sys
import time
import logging

import streamlit as st
import requests

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config import get_config

logger = logging.getLogger(__name__)

# ── Load .env ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

config = get_config()

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="FinAnalyst · Multi-Agent Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import and apply CSS ───────────────────────────────────────────────────
from app.frontend.components import (
    apply_custom_css,
    render_comparison_table,
    render_download_button,
    render_kpi_cards,
    render_kpi_charts,
    render_risk_factors,
    render_risk_gauge,
    render_summary_section,
)

apply_custom_css()

# ── LLM banner (only shown once if no key set) ────────────────────────────
if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OLLAMA_HOST"):
    st.warning(
        "**No LLM configured.** Add `GROQ_API_KEY` to your Space secrets — "
        "KPI extraction still works without it.",
        icon="⚠️",
    )


# ============================================================
# Session state
# ============================================================
def _init():
    defaults = {
        "analysis_result": None,
        "job_id":          None,
        "is_processing":   False,
        "upload_history":  [],
        "previous_kpis":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    with st.sidebar:

        # ── Logo ─────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 1rem 0 0.5rem; text-align: center;">
                <div style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.55rem;
                    margin-bottom: 0.3rem;
                ">
                    <div style="
                        width: 38px; height: 38px;
                        background: linear-gradient(135deg, #6366F1, #8B5CF6);
                        border-radius: 10px;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1.3rem;
                        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
                    ">📊</div>
                    <span style="
                        font-size: 1.55rem;
                        font-weight: 800;
                        background: linear-gradient(135deg, #818CF8, #A78BFA, #C4B5FD);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        letter-spacing: -0.02em;
                    ">FinAnalyst</span>
                </div>
                <p style="color: #475569; font-size: 0.78rem; margin: 0; letter-spacing: 0.03em;">
                    MULTI-AGENT FINANCIAL ANALYSIS
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Upload ───────────────────────────────────────────
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.78rem; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            "📄 Upload Report</p>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "upload_label",          # label hidden by CSS
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help="Upload a quarterly earnings report (PDF or plain text, max 200 MB).",
        )

        if uploaded_file:
            st.markdown(
                f"""
                <div style="
                    background: rgba(34,197,94,0.1);
                    border: 1px solid rgba(34,197,94,0.25);
                    border-radius: 8px;
                    padding: 0.5rem 0.8rem;
                    margin-top: 0.4rem;
                    font-size: 0.82rem;
                    color: #4ADE80;
                ">✅ {uploaded_file.name}</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── Report details ───────────────────────────────────
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.78rem; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            "🏢 Report Details</p>",
            unsafe_allow_html=True,
        )
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Apple Inc.",
            label_visibility="collapsed",
        )
        st.caption("Company name (leave blank for auto-detect)")

        quarter = st.text_input(
            "Quarter",
            placeholder="e.g., Q3 2025",
            label_visibility="collapsed",
        )
        st.caption("Quarter (leave blank for auto-detect)")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Previous quarter ─────────────────────────────────
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.78rem; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            "📊 Quarter Comparison</p>",
            unsafe_allow_html=True,
        )
        use_previous = st.checkbox("Enable QoQ comparison")

        previous_kpis = None
        if use_previous:
            prev_json = st.text_area(
                "Previous Quarter KPIs (JSON)",
                placeholder='{"revenue": "$85.5B", "net_income": "$20.7B", "eps": "$1.30"}',
                height=110,
                label_visibility="collapsed",
            )
            if prev_json.strip():
                try:
                    previous_kpis = json.loads(prev_json)
                    st.success("✅ Previous KPIs loaded")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON")
                    previous_kpis = None

        st.divider()

        # ── Run button ───────────────────────────────────────
        btn_label = (
            "⏳ Analyzing…" if st.session_state.is_processing
            else "🚀 Run Analysis"
        )
        if st.button(
            btn_label,
            type="primary",
            disabled=(uploaded_file is None or st.session_state.is_processing),
            use_container_width=True,
        ):
            if uploaded_file:
                run_analysis(uploaded_file, company_name, quarter, previous_kpis)

        # ── System status ─────────────────────────────────────
        st.divider()
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.78rem; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            "⚙️ System Status</p>",
            unsafe_allow_html=True,
        )
        _render_system_status()


def _render_system_status():
    try:
        from app.utils.llm import get_active_provider, get_active_model
        provider = get_active_provider()
        if provider == "groq":
            st.success(f"🟢 Groq · {get_active_model()}")
        elif provider == "ollama":
            from app.utils.llm import check_ollama_running
            if check_ollama_running():
                st.success(f"🟢 Ollama · {get_active_model()}")
            else:
                st.warning("🟡 Ollama · not running")
        else:
            st.error("🔴 No LLM provider")
            st.caption("Set GROQ_API_KEY secret")
    except Exception:
        st.warning("🟡 Status unknown")


# ============================================================
# Analysis runner
# ============================================================
def run_analysis(uploaded_file, company_name, quarter, previous_kpis):
    st.session_state.is_processing    = True
    st.session_state.analysis_result  = None

    try:
        result = _run_via_api(uploaded_file, company_name, quarter, previous_kpis)
        if result:
            st.session_state.analysis_result = result
            st.session_state.is_processing   = False
            st.rerun()
            return
    except Exception:
        pass

    try:
        result = _run_direct(uploaded_file, company_name, quarter, previous_kpis)
        st.session_state.analysis_result = result
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        logger.error("Analysis failed: %s", exc, exc_info=True)
    finally:
        st.session_state.is_processing = False
        st.rerun()


def _run_via_api(uploaded_file, company_name, quarter, previous_kpis):
    api_url = config.frontend.api_base_url
    files   = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    data    = {"company_name": company_name, "quarter": quarter}
    if previous_kpis:
        data["previous_kpis_json"] = json.dumps(previous_kpis)

    resp = requests.post(f"{api_url}/analyze", files=files, data=data, timeout=10)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    progress_bar = st.progress(0, text="Starting analysis…")
    for _ in range(300):
        time.sleep(1)
        status = requests.get(f"{api_url}/status/{job_id}", timeout=5).json()
        progress_bar.progress(min(status.get("progress", 0), 0.99), text=status.get("current_step", "Processing…"))
        if status["status"] == "complete":
            progress_bar.progress(1.0, text="✅ Done!")
            return status.get("result")
        elif status["status"] == "failed":
            raise RuntimeError(status.get("error", "Analysis failed"))

    raise TimeoutError("Timed out after 5 minutes")


def _run_direct(uploaded_file, company_name, quarter, previous_kpis):
    import tempfile

    progress_bar = st.progress(0, text="Saving file…")
    fname  = uploaded_file.name or "report.pdf"
    suffix = ".txt" if fname.lower().endswith(".txt") else ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        progress_bar.progress(0.1, text="Extracting text…")
        from app.graph.workflow import run_analysis as _run

        progress_bar.progress(0.2, text="Running AI agents (≈30s with Groq)…")
        result = _run(
            pdf_path=tmp_path,
            company_name=company_name,
            quarter=quarter,
            previous_kpis=previous_kpis,
        )

        progress_bar.progress(0.9, text="Formatting results…")
        report = _state_to_dict(result)
        progress_bar.progress(1.0, text="Analysis complete!")
        return report
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _state_to_dict(state) -> dict:
    return {
        "company_name":  state.company_name or "Unknown Company",
        "quarter":       state.quarter or "Unknown Quarter",
        "kpis":          state.kpis.model_dump() if hasattr(state.kpis, "model_dump") else {},
        "risk_analysis": state.risk_analysis.model_dump() if hasattr(state.risk_analysis, "model_dump") else {},
        "summary":       state.summary.model_dump() if hasattr(state.summary, "model_dump") else {},
        "comparison":    state.comparison.model_dump() if state.comparison and hasattr(state.comparison, "model_dump") else None,
        "errors":        state.errors,
        "warnings":      getattr(state, "warnings", []),
    }


# ============================================================
# Main content
# ============================================================
def render_main():
    result = st.session_state.analysis_result

    if result is None:
        _render_landing()
        return

    company = result.get("company_name", "Company")
    quarter = result.get("quarter", "")

    # Header banner
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
            border: 1px solid rgba(99,102,241,0.28);
            border-radius: 18px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
        ">
            <h1 style="margin:0; font-size:2rem; color:#F1F5F9; font-weight:800;">{company}</h1>
            <p style="margin:0.3rem 0 0; color:#94A3B8; font-size:1.05rem;">
                {quarter} · Earnings Analysis Report
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Errors / warnings
    for err  in result.get("errors",   []):
        st.error(f"⚠️ {err}")
    for warn in result.get("warnings", []):
        st.warning(f"ℹ️ {warn}")

    # Tabs
    tab_kpi, tab_risk, tab_summary, tab_compare, tab_report = st.tabs([
        "📊 KPI Dashboard",
        "⚠️ Risk Analysis",
        "📝 Earnings Summary",
        "📈 Quarter Comparison",
        "📋 Full Report",
    ])

    with tab_kpi:     _render_kpi_tab(result)
    with tab_risk:    _render_risk_tab(result)
    with tab_summary: _render_summary_tab(result)
    with tab_compare: _render_comparison_tab(result)
    with tab_report:  _render_report_tab(result)


def _render_landing():
    """Premium landing page shown before any upload."""
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem 1rem;">
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
                border: 1px solid rgba(99,102,241,0.2);
                border-radius: 20px;
                padding: 0.6rem 1.4rem;
                font-size: 0.8rem;
                color: #A78BFA;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 1.4rem;
            ">Powered by Groq · LangGraph · Streamlit</div>
            <h1 style="
                font-size: 3.2rem;
                background: linear-gradient(135deg, #818CF8 0%, #A78BFA 50%, #C084FC 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                line-height: 1.15;
                margin: 0 0 1rem;
                letter-spacing: -0.03em;
            ">Multi-Agent<br>Financial Analyst</h1>
            <p style="
                color: #64748B;
                font-size: 1.1rem;
                max-width: 560px;
                margin: 0 auto 2.5rem;
                line-height: 1.6;
            ">
                Upload any earnings report PDF and get instant AI-powered KPI extraction,
                risk assessment, executive summary, and quarter-over-quarter comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    cols = st.columns(4, gap="medium")
    features = [
        ("#6366F1", "📊", "KPI Extraction",
         "Revenue, EPS, margins, cash flow & 8 more metrics — automatically extracted."),
        ("#EF4444", "⚠️", "Risk Analysis",
         "Multi-factor risk scoring with explainable evidence from the report."),
        ("#22C55E", "📝", "AI Summary",
         "Executive summary, growth drivers, outlook — generated by Groq LLM."),
        ("#3B82F6", "📈", "QoQ Comparison",
         "Quarter-over-quarter deltas with trend indicators and exec insights."),
    ]

    for col, (color, emoji, title, desc) in zip(cols, features):
        r, g, b = (
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
        )
        with col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(145deg, rgba({r},{g},{b},0.10), rgba({r},{g},{b},0.04));
                    border: 1px solid rgba({r},{g},{b},0.22);
                    border-radius: 16px;
                    padding: 1.6rem 1.2rem;
                    text-align: center;
                    transition: all 0.3s;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">{emoji}</div>
                    <h3 style="
                        color: #E2E8F0;
                        margin: 0 0 0.5rem;
                        font-size: 0.98rem;
                        font-weight: 700;
                    ">{title}</h3>
                    <p style="
                        color: #64748B;
                        font-size: 0.82rem;
                        margin: 0;
                        line-height: 1.5;
                    ">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Call-to-action footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2.5rem; padding-bottom: 1rem;">
            <p style="color: #334155; font-size: 0.9rem;">
                ⬅️ Upload a PDF or TXT report in the sidebar to get started
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_tab(result: dict):
    kpis = result.get("kpis", {})
    if not kpis or all(
        v is None for k, v in kpis.items()
        if k not in ("confidence", "extraction_method", "raw_values")
    ):
        st.info("No KPIs were extracted from this report.")
        return
    render_kpi_cards(kpis)
    st.divider()
    st.markdown("### 📊 Financial Metrics Overview")
    render_kpi_charts(kpis)


def _render_risk_tab(result: dict):
    risk = result.get("risk_analysis", {})
    if not risk:
        st.info("Risk analysis not available.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Overall Risk Score")
        render_risk_gauge(risk.get("overall_score", 0), risk.get("risk_level", "low"))
    with col2:
        st.markdown("### Risk Summary")
        if risk.get("summary"):
            st.markdown(risk["summary"])
        recs = risk.get("recommendations", [])
        if recs:
            st.markdown("#### 💡 Recommendations")
            for r in recs:
                st.markdown(f"- {r}")

    st.divider()
    factors = risk.get("factors", [])
    if factors:
        st.markdown("### Risk Factor Breakdown")
        render_risk_factors(factors)


def _render_summary_tab(result: dict):
    summary = result.get("summary", {})
    if not summary or not summary.get("executive_summary"):
        st.info("Earnings summary not available.")
        return
    render_summary_section(summary)


def _render_comparison_tab(result: dict):
    comparison = result.get("comparison")
    if not comparison:
        st.info(
            "Quarter comparison is not available. "
            "Enable **QoQ comparison** in the sidebar and provide previous-quarter KPIs."
        )
        return

    current_q  = comparison.get("current_quarter",  "Current")
    previous_q = comparison.get("previous_quarter", "Previous")
    trend      = comparison.get("overall_trend", "stable")
    emoji      = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend, "➡️")

    st.markdown(f"### {emoji} {previous_q} → {current_q}  ·  Overall: **{trend.title()}**")

    if comparison.get("comparisons"):
        render_comparison_table(comparison["comparisons"])

    if comparison.get("executive_insights"):
        st.divider()
        st.markdown("### 💡 Executive Insights")
        st.markdown(comparison["executive_insights"])


def _render_report_tab(result: dict):
    st.markdown("### 📋 Full Analysis Report")
    render_download_button(result, result.get("company_name", "Company"))
    st.divider()
    with st.expander("📄 Raw JSON Report", expanded=False):
        st.json(result)


# ============================================================
# Entry point
# ============================================================
render_sidebar()
render_main()
