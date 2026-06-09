"""
Streamlit dashboard for the Multi-Agent Financial Analyst.

This is the main frontend application that provides:
- PDF upload interface
- KPI dashboard with metric cards and charts
- Risk analysis dashboard with gauge and factors
- Earnings summary view
- Quarter comparison table
- Downloadable full report

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

# ============================================================
# Page config
# ============================================================
# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

config = get_config()

st.set_page_config(
    page_title=config.frontend.page_title,
    page_icon=config.frontend.page_icon,
    layout=config.frontend.layout,
    initial_sidebar_state="expanded",
)

# ── Groq API key banner ────────────────────────────────────────────────────
import os as _os
if not _os.environ.get("GROQ_API_KEY") and not _os.environ.get("OLLAMA_HOST"):
    st.warning(
        "**No LLM provider configured.** "
        "Set the `GROQ_API_KEY` secret in your Space settings, or run Ollama locally. "
        "KPI extraction (regex-based) will still work without an LLM.",
        icon="⚠️",
    )

# ============================================================
# Import UI components
# ============================================================
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

# Apply custom styling
apply_custom_css()


# ============================================================
# Session state initialization
# ============================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "analysis_result": None,
        "job_id": None,
        "is_processing": False,
        "upload_history": [],
        "previous_kpis": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """Render the sidebar with upload and settings."""
    with st.sidebar:
        # Logo / header
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="
                    background: linear-gradient(135deg, #6366F1, #8B5CF6, #A78BFA);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 1.8rem;
                    font-weight: 800;
                    margin-bottom: 0.2rem;
                ">📊 FinAnalyst</h1>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Multi-Agent Financial Analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # --- File upload ---
        st.markdown("### 📄 Upload Report")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT earnings report",
            type=["pdf", "txt"],
            help="Upload a company's quarterly earnings report (PDF or plain text).",
        )

        # --- Company info ---
        st.markdown("### 🏢 Report Details")
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Apple Inc.",
            help="Leave blank for auto-detection",
        )
        quarter = st.text_input(
            "Quarter",
            placeholder="e.g., Q3 2025",
            help="Leave blank for auto-detection",
        )

        # --- Previous quarter (for comparison) ---
        st.markdown("### 📊 Previous Quarter Data")
        use_previous = st.checkbox(
            "Enable Quarter Comparison",
            help="Provide previous quarter KPIs for QoQ comparison",
        )

        previous_kpis = None
        if use_previous:
            prev_json = st.text_area(
                "Previous Quarter KPIs (JSON)",
                placeholder='{"revenue": "$85.5B", "net_income": "$20.7B", "eps": "$1.30"}',
                height=120,
            )
            if prev_json.strip():
                try:
                    previous_kpis = json.loads(prev_json)
                    st.success("✅ Previous KPIs loaded")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format")
                    previous_kpis = None

        st.divider()

        # --- Analyse button ---
        analyse_disabled = uploaded_file is None or st.session_state.is_processing
        if st.button(
            "🚀 Run Analysis" if not st.session_state.is_processing else "⏳ Processing...",
            type="primary",
            disabled=analyse_disabled,
            use_container_width=True,
        ):
            if uploaded_file:
                run_analysis(uploaded_file, company_name, quarter, previous_kpis)

        # --- System status ---
        st.divider()
        st.markdown("### ⚙️ System Status")
        _render_system_status()


def _render_system_status():
    """Show LLM provider and connection status."""
    try:
        from app.utils.llm import get_active_provider, get_active_model, _groq_available
        provider = get_active_provider()
        if provider == "groq":
            model = get_active_model()
            st.success(f"🟢 Groq: {model}")
        elif provider == "ollama":
            from app.utils.llm import check_ollama_running
            if check_ollama_running():
                model = get_active_model()
                st.success(f"🟢 Ollama: {model}")
            else:
                st.warning("🟡 Ollama: Not running")
        else:
            st.error("🔴 No LLM provider")
            st.caption("Set GROQ_API_KEY or start Ollama")
    except Exception:
        st.warning("🟡 Status: Unknown")


# ============================================================
# Analysis runner
# ============================================================
def run_analysis(uploaded_file, company_name: str, quarter: str, previous_kpis: dict):
    """Run the analysis pipeline — tries API first, falls back to direct mode."""
    st.session_state.is_processing = True
    st.session_state.analysis_result = None

    try:
        # Try API mode first
        result = _run_via_api(uploaded_file, company_name, quarter, previous_kpis)
        if result:
            st.session_state.analysis_result = result
            st.session_state.is_processing = False
            st.rerun()
            return
    except Exception:
        pass

    # Fallback to direct mode
    try:
        result = _run_direct(uploaded_file, company_name, quarter, previous_kpis)
        st.session_state.analysis_result = result
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        logger.error("Analysis failed: %s", exc, exc_info=True)
    finally:
        st.session_state.is_processing = False
        st.rerun()


def _run_via_api(uploaded_file, company_name, quarter, previous_kpis) -> dict:
    """Submit analysis via the FastAPI backend."""
    api_url = config.frontend.api_base_url

    # Upload file
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    data = {"company_name": company_name, "quarter": quarter}
    if previous_kpis:
        data["previous_kpis_json"] = json.dumps(previous_kpis)

    resp = requests.post(f"{api_url}/analyze", files=files, data=data, timeout=10)
    resp.raise_for_status()
    job_data = resp.json()
    job_id = job_data["job_id"]

    # Poll for results
    progress_bar = st.progress(0, text="Starting analysis...")
    for i in range(300):  # Max 5 minutes
        time.sleep(1)
        status_resp = requests.get(f"{api_url}/status/{job_id}", timeout=5)
        status = status_resp.json()

        progress = status.get("progress", 0)
        step = status.get("current_step", "Processing...")
        progress_bar.progress(min(progress, 0.99), text=step)

        if status["status"] == "complete":
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            return status.get("result")
        elif status["status"] == "failed":
            raise RuntimeError(status.get("error", "Analysis failed"))

    raise TimeoutError("Analysis timed out after 5 minutes")


def _run_direct(uploaded_file, company_name, quarter, previous_kpis) -> dict:
    """Run analysis directly without the API server."""
    import tempfile

    progress_bar = st.progress(0, text="Saving uploaded file...")

    # Determine suffix from file name
    fname = uploaded_file.name or "report.pdf"
    suffix = ".txt" if fname.lower().endswith(".txt") else ".pdf"

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        progress_bar.progress(0.1, text="Extracting text...")
        from app.graph.workflow import run_analysis

        progress_bar.progress(0.2, text="Running agent pipeline (this takes ~30s with Groq)...")
        result = run_analysis(
            pdf_path=tmp_path,
            company_name=company_name,
            quarter=quarter,
            previous_kpis=previous_kpis,
        )

        progress_bar.progress(0.9, text="Formatting results...")
        report = _state_to_dict(result)
        progress_bar.progress(1.0, text="Analysis complete!")
        return report

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _state_to_dict(state) -> dict:
    """Convert PipelineState to a display-friendly dict."""
    result = {
        "company_name": state.company_name or "Unknown Company",
        "quarter": state.quarter or "Unknown Quarter",
        "kpis": state.kpis.model_dump() if hasattr(state.kpis, "model_dump") else {},
        "risk_analysis": state.risk_analysis.model_dump() if hasattr(state.risk_analysis, "model_dump") else {},
        "summary": state.summary.model_dump() if hasattr(state.summary, "model_dump") else {},
        "comparison": state.comparison.model_dump() if state.comparison and hasattr(state.comparison, "model_dump") else None,
        "errors": state.errors,
        "warnings": getattr(state, "warnings", []),
    }
    return result


# ============================================================
# Main content
# ============================================================
def render_main():
    """Render the main content area."""
    result = st.session_state.analysis_result

    if result is None:
        _render_landing()
        return

    # Header
    company = result.get("company_name", "Company")
    quarter = result.get("quarter", "")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
        ">
            <h1 style="margin:0; font-size:2rem; color:#F1F5F9;">
                {company}
            </h1>
            <p style="margin:0.3rem 0 0; color:#94A3B8; font-size:1.1rem;">
                {quarter} Earnings Analysis Report
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show errors/warnings
    errors = result.get("errors", [])
    warnings = result.get("warnings", [])
    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    if warnings:
        for warn in warnings:
            st.warning(f"ℹ️ {warn}")

    # Tabs
    tab_kpi, tab_risk, tab_summary, tab_compare, tab_report = st.tabs([
        "📊 KPI Dashboard",
        "⚠️ Risk Analysis",
        "📝 Earnings Summary",
        "📈 Quarter Comparison",
        "📋 Full Report",
    ])

    with tab_kpi:
        _render_kpi_tab(result)

    with tab_risk:
        _render_risk_tab(result)

    with tab_summary:
        _render_summary_tab(result)

    with tab_compare:
        _render_comparison_tab(result)

    with tab_report:
        _render_report_tab(result)


def _render_landing():
    """Render the landing page when no analysis has been run."""
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 1rem;">
            <h1 style="
                font-size: 3rem;
                background: linear-gradient(135deg, #6366F1, #8B5CF6, #D946EF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                margin-bottom: 1rem;
            ">Multi-Agent Financial Analyst</h1>
            <p style="color: #94A3B8; font-size: 1.2rem; max-width: 600px; margin: 0 auto 2rem;">
                Upload an earnings report PDF to get instant AI-powered analysis
                with KPI extraction, risk assessment, executive summary, and
                quarter-over-quarter comparison.
            </p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 2rem;
                flex-wrap: wrap;
                margin-top: 2rem;
            ">
                <div style="
                    background: rgba(99,102,241,0.1);
                    border: 1px solid rgba(99,102,241,0.2);
                    border-radius: 12px;
                    padding: 1.5rem;
                    width: 200px;
                    text-align: center;
                ">
                    <div style="font-size: 2rem;">📊</div>
                    <h3 style="color: #E2E8F0; margin: 0.5rem 0 0.3rem;">KPI Extraction</h3>
                    <p style="color: #94A3B8; font-size: 0.85rem;">Revenue, EPS, margins, cash flow & more</p>
                </div>
                <div style="
                    background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.2);
                    border-radius: 12px;
                    padding: 1.5rem;
                    width: 200px;
                    text-align: center;
                ">
                    <div style="font-size: 2rem;">⚠️</div>
                    <h3 style="color: #E2E8F0; margin: 0.5rem 0 0.3rem;">Risk Analysis</h3>
                    <p style="color: #94A3B8; font-size: 0.85rem;">Explainable scoring with evidence</p>
                </div>
                <div style="
                    background: rgba(34,197,94,0.1);
                    border: 1px solid rgba(34,197,94,0.2);
                    border-radius: 12px;
                    padding: 1.5rem;
                    width: 200px;
                    text-align: center;
                ">
                    <div style="font-size: 2rem;">📝</div>
                    <h3 style="color: #E2E8F0; margin: 0.5rem 0 0.3rem;">AI Summary</h3>
                    <p style="color: #94A3B8; font-size: 0.85rem;">Executive summaries powered by local LLM</p>
                </div>
                <div style="
                    background: rgba(59,130,246,0.1);
                    border: 1px solid rgba(59,130,246,0.2);
                    border-radius: 12px;
                    padding: 1.5rem;
                    width: 200px;
                    text-align: center;
                ">
                    <div style="font-size: 2rem;">📈</div>
                    <h3 style="color: #E2E8F0; margin: 0.5rem 0 0.3rem;">QoQ Comparison</h3>
                    <p style="color: #94A3B8; font-size: 0.85rem;">Track performance trends over time</p>
                </div>
            </div>
            <p style="color: #64748B; margin-top: 3rem; font-size: 0.9rem;">
                ⬅️ Upload a PDF in the sidebar to get started
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_tab(result: dict):
    """Render the KPI dashboard tab."""
    kpis = result.get("kpis", {})

    if not kpis or all(v is None for k, v in kpis.items() if k not in ("confidence", "extraction_method", "raw_values")):
        st.info("No KPIs were extracted from this report.")
        return

    # KPI cards
    render_kpi_cards(kpis)

    st.divider()

    # KPI chart
    st.markdown("### 📊 Financial Metrics Overview")
    render_kpi_charts(kpis)


def _render_risk_tab(result: dict):
    """Render the risk analysis tab."""
    risk = result.get("risk_analysis", {})

    if not risk:
        st.info("Risk analysis not available.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Overall Risk Score")
        render_risk_gauge(
            risk.get("overall_score", 0),
            risk.get("risk_level", "low"),
        )

    with col2:
        st.markdown("### Risk Summary")
        summary = risk.get("summary", "")
        if summary:
            st.markdown(summary)

        recommendations = risk.get("recommendations", [])
        if recommendations:
            st.markdown("#### 💡 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")

    st.divider()

    factors = risk.get("factors", [])
    if factors:
        st.markdown("### Risk Factor Breakdown")
        render_risk_factors(factors)


def _render_summary_tab(result: dict):
    """Render the earnings summary tab."""
    summary = result.get("summary", {})

    if not summary or not summary.get("executive_summary"):
        st.info("Earnings summary not available.")
        return

    render_summary_section(summary)


def _render_comparison_tab(result: dict):
    """Render the quarter comparison tab."""
    comparison = result.get("comparison")

    if not comparison:
        st.info(
            "Quarter comparison is not available. "
            "To enable comparison, provide previous quarter KPIs "
            "in the sidebar before running analysis."
        )
        return

    # Header
    current_q = comparison.get("current_quarter", "Current")
    previous_q = comparison.get("previous_quarter", "Previous")
    overall_trend = comparison.get("overall_trend", "stable")

    trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(overall_trend, "➡️")

    st.markdown(
        f"### {trend_emoji} {previous_q} → {current_q}  |  Overall: **{overall_trend.title()}**"
    )

    comparisons = comparison.get("comparisons", [])
    if comparisons:
        render_comparison_table(comparisons)

    insights = comparison.get("executive_insights", "")
    if insights:
        st.divider()
        st.markdown("### 💡 Executive Insights")
        st.markdown(insights)


def _render_report_tab(result: dict):
    """Render the full report tab with download options."""
    st.markdown("### 📋 Full Analysis Report")

    render_download_button(
        result,
        result.get("company_name", "Company"),
    )

    st.divider()

    # Show raw JSON
    with st.expander("📄 Raw JSON Report", expanded=False):
        st.json(result)


# ============================================================
# Main
# ============================================================
render_sidebar()
render_main()
