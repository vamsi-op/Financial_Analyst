"""
Reusable Streamlit UI components for the Financial Analyst dashboard.

Provides premium-styled cards, charts, gauges, and tables
with a dark-theme glassmorphism aesthetic.
"""

import json
import logging
from typing import Any, Optional

import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================
# Custom CSS
# ============================================================

def apply_custom_css():
    """Inject custom CSS for premium dark-theme styling with glassmorphism."""
    st.markdown(
        """
        <style>
        /* --- Google Fonts --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* --- Global Reset --- */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* --- Dark background --- */
        .stApp {
            background: #0A0E1A;
        }

        /* --- Main container --- */
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1300px !important;
        }

        /* ── Sidebar ──────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1117 0%, #111827 100%) !important;
            border-right: 1px solid rgba(99,102,241,0.15) !important;
            min-width: 300px !important;
            max-width: 320px !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding: 1rem 1.2rem !important;
        }

        /* Hide the redundant "collapse" arrow label text */
        [data-testid="collapsedControl"] span { display: none !important; }

        /* ── File uploader – full override ───────────────────── */
        /* Hide the label element entirely */
        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] label {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            overflow: hidden !important;
        }

        /* Style the dropzone container */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(99,102,241,0.06) !important;
            border: 1.5px dashed rgba(99,102,241,0.35) !important;
            border-radius: 12px !important;
            padding: 1.2rem !important;
            transition: border-color 0.25s, background 0.25s;
            text-align: center;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            background: rgba(99,102,241,0.12) !important;
            border-color: rgba(139,92,246,0.6) !important;
        }

        /* Hide the native Browse Files button text & re-inject it */
        [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
            border: 1px solid rgba(99,102,241,0.35) !important;
            border-radius: 8px !important;
            color: transparent !important;   /* hide original text */
            font-size: 0 !important;         /* hide original text */
            padding: 0.45rem 1.1rem !important;
            cursor: pointer;
            position: relative;
            min-width: 130px;
        }
        [data-testid="stFileUploaderDropzone"] button::after {
            content: "📂  Choose File";
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            color: #C4B5FD !important;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            white-space: nowrap;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: rgba(99,102,241,0.3) !important;
            border-color: rgba(139,92,246,0.6) !important;
        }

        /* Dropzone instruction text */
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #475569 !important;
            font-size: 0.78rem !important;
        }
        /* Hide the default drag-here text (we rely on the button) */
        [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
            display: none !important;
        }


        /* ── Sidebar text inputs ───────────────────────────────── */
        [data-testid="stSidebar"] input[type="text"] {
            background: rgba(99,102,241,0.08) !important;
            border: 1px solid rgba(99,102,241,0.25) !important;
            border-radius: 8px !important;
            color: #E2E8F0 !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stSidebar"] input[type="text"]:focus {
            border-color: rgba(99,102,241,0.6) !important;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
        }
        [data-testid="stSidebar"] label {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }

        /* ── Primary button ────────────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            color: #fff !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.65rem 1.2rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
        }
        .stButton > button[kind="primary"]:disabled {
            opacity: 0.45 !important;
            transform: none !important;
            box-shadow: none !important;
        }

        /* ── Secondary buttons ─────────────────────────────────── */
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {
            background: rgba(99,102,241,0.1) !important;
            border: 1px solid rgba(99,102,241,0.3) !important;
            border-radius: 10px !important;
            color: #A78BFA !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover {
            background: rgba(99,102,241,0.2) !important;
            border-color: rgba(99,102,241,0.5) !important;
        }

        /* ── Divider ───────────────────────────────────────────── */
        hr {
            border: none !important;
            border-top: 1px solid rgba(99,102,241,0.15) !important;
            margin: 1rem 0 !important;
        }

        /* ── Tabs ──────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            background: rgba(15,23,42,0.6) !important;
            border-radius: 12px !important;
            padding: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px !important;
            padding: 8px 18px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            color: #94A3B8 !important;
            background: transparent !important;
            border: none !important;
            transition: all 0.2s !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(99,102,241,0.25) !important;
            color: #C4B5FD !important;
        }

        /* ── Success / warning / error banners ─────────────────── */
        .stAlert {
            border-radius: 10px !important;
            font-size: 0.88rem !important;
        }

        /* ── Expander ───────────────────────────────────────────── */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: #C4B5FD !important;
        }
        [data-testid="stExpander"] {
            background: rgba(15,23,42,0.5) !important;
            border: 1px solid rgba(99,102,241,0.15) !important;
            border-radius: 12px !important;
        }

        /* ── Section headings in main ───────────────────────────── */
        .main h3 {
            color: #C4B5FD !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.6rem !important;
        }

        /* ── Glassmorphism card ─────────────────────────────────── */
        .glass-card {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(99, 102, 241, 0.18);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.38);
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
            transform: translateY(-2px);
        }

        /* ── Metric card ────────────────────────────────────────── */
        .metric-card {
            background: linear-gradient(135deg, rgba(20,28,48,0.95), rgba(10,14,26,0.98));
            border: 1px solid rgba(99, 102, 241, 0.22);
            border-radius: 14px;
            padding: 1.2rem 1rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366F1, #8B5CF6, #A78BFA);
            border-radius: 14px 14px 0 0;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.45);
            box-shadow: 0 0 24px rgba(99, 102, 241, 0.18);
            transform: translateY(-3px);
        }
        .metric-label {
            color: #64748B;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            color: #F1F5F9;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-confidence {
            font-size: 0.68rem;
            padding: 2px 8px;
            border-radius: 10px;
            display: inline-block;
            font-weight: 600;
        }
        .conf-high   { background: rgba(34,197,94,0.18);  color: #4ADE80; }
        .conf-medium { background: rgba(250,204,21,0.18); color: #FACC15; }
        .conf-low    { background: rgba(239,68,68,0.18);  color: #F87171; }

        /* ── Risk bars ──────────────────────────────────────────── */
        .risk-bar-container {
            background: rgba(30,41,59,0.5);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .risk-bar { height: 100%; border-radius: 8px; transition: width 0.5s ease; }
        .risk-low      { background: linear-gradient(90deg, #22C55E, #4ADE80); }
        .risk-medium   { background: linear-gradient(90deg, #EAB308, #FACC15); }
        .risk-high     { background: linear-gradient(90deg, #F97316, #FB923C); }
        .risk-critical { background: linear-gradient(90deg, #EF4444, #F87171); }

        /* ── Severity badge ─────────────────────────────────────── */
        .severity-badge {
            font-size: 0.72rem;
            padding: 3px 10px;
            border-radius: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .badge-low      { background: rgba(34,197,94,0.18);  color: #4ADE80; }
        .badge-medium   { background: rgba(234,179,8,0.18);  color: #FACC15; }
        .badge-high     { background: rgba(249,115,22,0.18); color: #FB923C; }
        .badge-critical { background: rgba(239,68,68,0.18);  color: #F87171; }

        /* ── Summary block ──────────────────────────────────────── */
        .summary-block {
            background: rgba(15,23,42,0.6);
            border-left: 3px solid #6366F1;
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.5rem;
            margin: 0.8rem 0;
            color: #CBD5E1;
            line-height: 1.7;
        }

        /* ── Trend arrows ───────────────────────────────────────── */
        .trend-up   { color: #4ADE80; font-weight: 700; }
        .trend-down { color: #F87171; font-weight: 700; }
        .trend-flat { color: #94A3B8; font-weight: 700; }

        /* ── Checkbox ───────────────────────────────────────────── */
        [data-testid="stCheckbox"] label {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            text-transform: none !important;
            letter-spacing: 0 !important;
        }

        /* ── Progress bar ───────────────────────────────────────── */
        .stProgress > div > div {
            background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
            border-radius: 4px !important;
        }

        /* ── Scrollbar ──────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0A0E1A; }
        ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# KPI Components
# ============================================================

def render_kpi_cards(kpis: dict):
    """Render KPI metric cards in a responsive grid."""
    confidence = kpis.get("confidence", {})

    # Define display order and labels
    metrics = [
        ("revenue",           "Revenue",           "💰"),
        ("net_income",        "Net Income",         "💵"),
        ("operating_income",  "Operating Income",   "📊"),
        ("eps",               "EPS",                "📈"),
        ("cash_flow",         "Cash Flow",          "🏦"),
        ("free_cash_flow",    "Free Cash Flow",     "💸"),
        ("gross_margin",      "Gross Margin",       "📐"),
        ("operating_margin",  "Operating Margin",   "📏"),
        ("total_debt",        "Total Debt",         "🏗️"),
        ("total_assets",      "Total Assets",       "🏢"),
        ("total_liabilities", "Total Liabilities",  "📑"),
    ]

    available = [(k, l, e) for k, l, e in metrics if kpis.get(k)]

    if not available:
        st.info("No KPI metrics were extracted from this report.")
        return

    # Grid rows of 4
    for row_start in range(0, len(available), 4):
        row = available[row_start: row_start + 4]
        cols = st.columns(len(row))

        for col, (key, label, emoji) in zip(cols, row):
            value = kpis.get(key, "N/A")
            conf = confidence.get(key, 0)
            conf_class = (
                "conf-high"   if conf >= 0.85 else
                "conf-medium" if conf >= 0.70 else
                "conf-low"
            )
            conf_text = f"{conf:.0%}" if conf > 0 else "—"

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{emoji} {label}</div>
                        <div class="metric-value">{value}</div>
                        <span class="metric-confidence {conf_class}">
                            Conf: {conf_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_kpi_charts(kpis: dict):
    """Create a Plotly bar chart of extracted KPIs."""
    raw_values = kpis.get("raw_values", {})

    if not raw_values:
        st.caption("Chart not available — no numeric values extracted.")
        return

    bar_metrics = {
        k: v for k, v in raw_values.items()
        if k not in ("gross_margin", "operating_margin") and isinstance(v, (int, float))
    }

    if not bar_metrics:
        st.caption("Chart not available — no comparable numeric values.")
        return

    labels_map = {
        "revenue":           "Revenue",
        "net_income":        "Net Income",
        "operating_income":  "Op. Income",
        "eps":               "EPS",
        "cash_flow":         "Cash Flow",
        "free_cash_flow":    "FCF",
        "total_debt":        "Debt",
        "total_assets":      "Assets",
        "total_liabilities": "Liabilities",
    }

    labels = [labels_map.get(k, k) for k in bar_metrics.keys()]
    values = list(bar_metrics.values())

    max_val = max(abs(v) for v in values) if values else 1
    if max_val >= 1e9:
        display_values = [v / 1e9 for v in values]
        unit = "Billions ($)"
    elif max_val >= 1e6:
        display_values = [v / 1e6 for v in values]
        unit = "Millions ($)"
    else:
        display_values = values
        unit = "Value ($)"

    colors = ["#6366F1" if v >= 0 else "#F87171" for v in display_values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=display_values,
                marker_color=colors,
                marker_line_width=0,
                text=[f"{v:.1f}" for v in display_values],
                textposition="outside",
                textfont=dict(color="#F1F5F9", size=12),
            )
        ]
    )

    fig.update_layout(
        yaxis_title=unit,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter"),
        xaxis=dict(tickfont=dict(size=12), gridcolor="rgba(99,102,241,0.08)"),
        yaxis=dict(
            gridcolor="rgba(99,102,241,0.08)",
            zerolinecolor="rgba(99,102,241,0.2)",
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Risk Components
# ============================================================

def render_risk_gauge(score: float, level: str):
    """Render a Plotly gauge chart for the overall risk score."""
    color_map = {
        "low":      "#22C55E",
        "medium":   "#EAB308",
        "high":     "#F97316",
        "critical": "#EF4444",
    }
    bar_color = color_map.get(level, "#6366F1")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(
                font=dict(size=48, color="#F1F5F9", family="Inter"),
                suffix="/100",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor="rgba(99,102,241,0.3)",
                    tickfont=dict(color="#94A3B8"),
                ),
                bar=dict(color=bar_color, thickness=0.7),
                bgcolor="rgba(30,41,59,0.5)",
                borderwidth=2,
                bordercolor="rgba(99,102,241,0.2)",
                steps=[
                    dict(range=[0,  30], color="rgba(34,197,94,0.1)"),
                    dict(range=[30, 60], color="rgba(234,179,8,0.1)"),
                    dict(range=[60, 80], color="rgba(249,115,22,0.1)"),
                    dict(range=[80,100], color="rgba(239,68,68,0.1)"),
                ],
                threshold=dict(
                    line=dict(color="#F1F5F9", width=3),
                    thickness=0.8,
                    value=score,
                ),
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=40, b=20),
        font=dict(family="Inter"),
    )

    st.plotly_chart(fig, use_container_width=True)

    badge_class = f"badge-{level}"
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: -1rem;">
            <span class="severity-badge {badge_class}">
                Risk Level: {level.upper()}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_factors(factors: list):
    """Render risk factors as expandable styled cards."""
    if not factors:
        st.info("No risk factors identified.")
        return

    for factor in factors:
        category    = factor.get("category", "Unknown")
        score       = factor.get("score", 0)
        severity    = factor.get("severity", "low")
        description = factor.get("description", "")
        evidence    = factor.get("evidence", [])

        bar_class   = f"risk-{severity}"
        badge_class = f"badge-{severity}"

        with st.expander(f"**{category}** — Score: {score:.0f}/100", expanded=False):
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span style="color:#E2E8F0; font-weight:600;">{category}</span>
                        <span class="severity-badge {badge_class}">{severity.upper()}</span>
                    </div>
                    <div class="risk-bar-container">
                        <div class="risk-bar {bar_class}" style="width:{min(score, 100)}%;"></div>
                    </div>
                    <p style="color:#CBD5E1; font-size:0.9rem; margin-top:0.8rem;">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if evidence:
                st.markdown("**📌 Evidence:**")
                for e in evidence:
                    st.markdown(f"- {e}")


# ============================================================
# Summary Components
# ============================================================

def render_summary_section(summary: dict):
    """Render the earnings summary with formatted sections."""
    executive = summary.get("executive_summary", "")

    if executive:
        st.markdown("### 📝 Executive Summary")
        st.markdown(
            f"""<div class="summary-block">{executive}</div>""",
            unsafe_allow_html=True,
        )

    highlights = summary.get("management_highlights", [])
    if highlights:
        st.markdown("### 🌟 Management Highlights")
        for h in highlights:
            st.markdown(f"- {h}")

    drivers = summary.get("growth_drivers", [])
    if drivers:
        st.markdown("### 🚀 Growth Drivers")
        for d in drivers:
            st.markdown(f"- {d}")

    challenges = summary.get("challenges", [])
    if challenges:
        st.markdown("### ⚡ Challenges")
        for c in challenges:
            st.markdown(f"- {c}")

    outlook = summary.get("outlook", "")
    if outlook:
        st.markdown("### 🔮 Outlook")
        st.markdown(
            f"""<div class="summary-block">{outlook}</div>""",
            unsafe_allow_html=True,
        )

    quotes = summary.get("key_quotes", [])
    if quotes:
        st.markdown("### 💬 Key Quotes")
        for q in quotes:
            st.markdown(f"> *{q}*")


# ============================================================
# Comparison Components
# ============================================================

def render_comparison_table(comparisons: list):
    """Render QoQ comparison as a styled table."""
    if not comparisons:
        st.info("No comparison data available.")
        return

    rows_html = ""
    for comp in comparisons:
        name       = comp.get("metric_name", "")
        current    = comp.get("current_value",  "N/A") or "N/A"
        previous   = comp.get("previous_value", "N/A") or "N/A"
        change_pct = comp.get("change_percent")
        trend      = comp.get("trend", "→")

        if trend == "↑":
            trend_class  = "trend-up"
            change_color = "#4ADE80"
        elif trend == "↓":
            trend_class  = "trend-down"
            change_color = "#F87171"
        else:
            trend_class  = "trend-flat"
            change_color = "#94A3B8"

        change_str = f"{change_pct:+.1f}%" if change_pct is not None else "—"

        rows_html += f"""
            <tr>
                <td style="padding:12px 16px; color:#E2E8F0; font-weight:500; border-bottom:1px solid rgba(99,102,241,0.1);">{name}</td>
                <td style="padding:12px 16px; color:#94A3B8; text-align:right; border-bottom:1px solid rgba(99,102,241,0.1);">{previous}</td>
                <td style="padding:12px 16px; color:#F1F5F9; text-align:right; font-weight:600; border-bottom:1px solid rgba(99,102,241,0.1);">{current}</td>
                <td style="padding:12px 16px; text-align:right; color:{change_color}; font-weight:600; border-bottom:1px solid rgba(99,102,241,0.1);">{change_str}</td>
                <td style="padding:12px 16px; text-align:center; font-size:1.2rem; border-bottom:1px solid rgba(99,102,241,0.1);"><span class="{trend_class}">{trend}</span></td>
            </tr>
        """

    st.markdown(
        f"""
        <div class="glass-card" style="overflow-x:auto; padding:0;">
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr style="background:rgba(99,102,241,0.1);">
                        <th style="padding:14px 16px; text-align:left;  color:#94A3B8; font-weight:600; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">Metric</th>
                        <th style="padding:14px 16px; text-align:right; color:#94A3B8; font-weight:600; font-size:0.82rem; text-transform:uppercase;">Previous</th>
                        <th style="padding:14px 16px; text-align:right; color:#94A3B8; font-weight:600; font-size:0.82rem; text-transform:uppercase;">Current</th>
                        <th style="padding:14px 16px; text-align:right; color:#94A3B8; font-weight:600; font-size:0.82rem; text-transform:uppercase;">Change</th>
                        <th style="padding:14px 16px; text-align:center;color:#94A3B8; font-weight:600; font-size:0.82rem; text-transform:uppercase;">Trend</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Report Download
# ============================================================

def render_download_button(report: dict, company_name: str):
    """Render download buttons for JSON and Markdown reports."""
    col1, col2 = st.columns(2)

    with col1:
        json_str = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"{company_name.replace(' ', '_')}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        md_str = _build_markdown_report(report)
        st.download_button(
            label="📥 Download Markdown Report",
            data=md_str,
            file_name=f"{company_name.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _build_markdown_report(report: dict) -> str:
    """Build a full Markdown report from the analysis results."""
    company    = report.get("company_name", "Company")
    quarter    = report.get("quarter", "")
    kpis       = report.get("kpis", {})
    risk       = report.get("risk_analysis", {})
    summary    = report.get("summary", {})
    comparison = report.get("comparison")

    lines = [
        f"# {company} — {quarter} Earnings Analysis\n",
        "*Generated by Multi-Agent Financial Analyst*\n",
        "---\n",
    ]

    lines.append("## Financial KPIs\n")
    lines.append("| Metric | Value | Confidence |")
    lines.append("|--------|-------|------------|")

    metric_labels = {
        "revenue":           "Revenue",
        "net_income":        "Net Income",
        "operating_income":  "Operating Income",
        "eps":               "EPS",
        "cash_flow":         "Cash Flow",
        "gross_margin":      "Gross Margin",
        "operating_margin":  "Operating Margin",
        "total_debt":        "Total Debt",
        "total_assets":      "Total Assets",
        "total_liabilities": "Total Liabilities",
    }
    confidence = kpis.get("confidence", {})
    for key, label in metric_labels.items():
        val = kpis.get(key)
        if val:
            conf = confidence.get(key, 0)
            lines.append(f"| {label} | {val} | {conf:.0%} |")
    lines.append("")

    lines.append("## Risk Analysis\n")
    lines.append(f"**Overall Score:** {risk.get('overall_score', 0):.0f}/100\n")
    lines.append(f"**Risk Level:** {risk.get('risk_level', 'N/A')}\n")
    if risk.get("summary"):
        lines.append(f"\n{risk['summary']}\n")

    if summary.get("executive_summary"):
        lines.append("## Executive Summary\n")
        lines.append(summary["executive_summary"])
        lines.append("")

    if comparison:
        lines.append("## Quarter Comparison\n")
        lines.append("| Metric | Previous | Current | Change | Trend |")
        lines.append("|--------|----------|---------|--------|-------|")
        for c in comparison.get("comparisons", []):
            pct = f"{c.get('change_percent', 0):+.1f}%" if c.get("change_percent") is not None else "—"
            lines.append(
                f"| {c.get('metric_name', '')} | {c.get('previous_value', 'N/A')} | "
                f"{c.get('current_value', 'N/A')} | {pct} | {c.get('trend', '→')} |"
            )
        lines.append("")

    return "\n".join(lines)
