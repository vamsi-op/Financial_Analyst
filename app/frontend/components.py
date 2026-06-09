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

        /* --- Global --- */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }

        /* --- Glassmorphism card --- */
        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
            transform: translateY(-2px);
        }

        /* --- Metric card --- */
        .metric-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
            backdrop-filter: blur(12px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366F1, #8B5CF6, #A78BFA);
            border-radius: 14px 14px 0 0;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
            transform: translateY(-3px);
        }
        .metric-label {
            color: #94A3B8;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            color: #F1F5F9;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .metric-confidence {
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 10px;
            display: inline-block;
            font-weight: 600;
        }
        .conf-high { background: rgba(34,197,94,0.2); color: #4ADE80; }
        .conf-medium { background: rgba(250,204,21,0.2); color: #FACC15; }
        .conf-low { background: rgba(239,68,68,0.2); color: #F87171; }

        /* --- Risk factor bar --- */
        .risk-bar-container {
            background: rgba(30,41,59,0.5);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .risk-bar {
            height: 100%;
            border-radius: 8px;
            transition: width 0.5s ease;
        }
        .risk-low { background: linear-gradient(90deg, #22C55E, #4ADE80); }
        .risk-medium { background: linear-gradient(90deg, #EAB308, #FACC15); }
        .risk-high { background: linear-gradient(90deg, #F97316, #FB923C); }
        .risk-critical { background: linear-gradient(90deg, #EF4444, #F87171); }

        /* --- Severity badge --- */
        .severity-badge {
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .badge-low { background: rgba(34,197,94,0.2); color: #4ADE80; }
        .badge-medium { background: rgba(234,179,8,0.2); color: #FACC15; }
        .badge-high { background: rgba(249,115,22,0.2); color: #FB923C; }
        .badge-critical { background: rgba(239,68,68,0.2); color: #F87171; }

        /* --- Summary section --- */
        .summary-block {
            background: rgba(30,41,59,0.5);
            border-left: 3px solid #6366F1;
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.5rem;
            margin: 0.8rem 0;
        }

        /* --- Trend arrows --- */
        .trend-up { color: #4ADE80; font-weight: 700; }
        .trend-down { color: #F87171; font-weight: 700; }
        .trend-flat { color: #94A3B8; font-weight: 700; }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 500;
        }

        /* --- Expanders --- */
        .streamlit-expanderHeader {
            font-weight: 600;
        }

        /* --- Divider --- */
        hr {
            border-color: rgba(99,102,241,0.15) !important;
        }
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
        ("revenue", "Revenue", "💰"),
        ("net_income", "Net Income", "💵"),
        ("operating_income", "Operating Income", "📊"),
        ("eps", "EPS", "📈"),
        ("cash_flow", "Cash Flow", "🏦"),
        ("free_cash_flow", "Free Cash Flow", "💸"),
        ("gross_margin", "Gross Margin", "📐"),
        ("operating_margin", "Operating Margin", "📏"),
        ("total_debt", "Total Debt", "🏗️"),
        ("total_assets", "Total Assets", "🏢"),
        ("total_liabilities", "Total Liabilities", "📑"),
    ]

    # Filter to only metrics that have values
    available = [(k, l, e) for k, l, e in metrics if kpis.get(k)]

    if not available:
        st.info("No KPI metrics were extracted from this report.")
        return

    # Create grid rows of 4
    for row_start in range(0, len(available), 4):
        row = available[row_start : row_start + 4]
        cols = st.columns(len(row))

        for col, (key, label, emoji) in zip(cols, row):
            value = kpis.get(key, "N/A")
            conf = confidence.get(key, 0)
            conf_class = "conf-high" if conf >= 0.85 else "conf-medium" if conf >= 0.7 else "conf-low"
            conf_text = f"{conf:.0%}" if conf > 0 else "—"

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{emoji} {label}</div>
                        <div class="metric-value">{value}</div>
                        <span class="metric-confidence {conf_class}">
                            Confidence: {conf_text}
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

    # Filter out margins (they're percentages, different scale)
    bar_metrics = {
        k: v for k, v in raw_values.items()
        if k not in ("gross_margin", "operating_margin") and isinstance(v, (int, float))
    }

    if not bar_metrics:
        st.caption("Chart not available — no comparable numeric values.")
        return

    # Pretty labels
    labels_map = {
        "revenue": "Revenue",
        "net_income": "Net Income",
        "operating_income": "Op. Income",
        "eps": "EPS",
        "cash_flow": "Cash Flow",
        "free_cash_flow": "FCF",
        "total_debt": "Debt",
        "total_assets": "Assets",
        "total_liabilities": "Liabilities",
    }

    labels = [labels_map.get(k, k) for k in bar_metrics.keys()]
    values = list(bar_metrics.values())

    # Normalize to millions/billions for display
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

    colors = [
        "#6366F1" if v >= 0 else "#F87171" for v in display_values
    ]

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
        xaxis=dict(
            tickfont=dict(size=12),
            gridcolor="rgba(99,102,241,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(99,102,241,0.1)",
            zerolinecolor="rgba(99,102,241,0.2)",
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Risk Components
# ============================================================

def render_risk_gauge(score: float, level: str):
    """Render a Plotly gauge chart for the overall risk score."""
    color_map = {
        "low": "#22C55E",
        "medium": "#EAB308",
        "high": "#F97316",
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
                    dict(range=[0, 30], color="rgba(34,197,94,0.1)"),
                    dict(range=[30, 60], color="rgba(234,179,8,0.1)"),
                    dict(range=[60, 80], color="rgba(249,115,22,0.1)"),
                    dict(range=[80, 100], color="rgba(239,68,68,0.1)"),
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

    # Level badge
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
        category = factor.get("category", "Unknown")
        score = factor.get("score", 0)
        severity = factor.get("severity", "low")
        description = factor.get("description", "")
        evidence = factor.get("evidence", [])

        bar_class = f"risk-{severity}"
        badge_class = f"badge-{severity}"

        with st.expander(f"**{category}** — Score: {score:.0f}/100", expanded=False):
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #E2E8F0; font-weight: 600;">{category}</span>
                        <span class="severity-badge {badge_class}">{severity.upper()}</span>
                    </div>
                    <div class="risk-bar-container">
                        <div class="risk-bar {bar_class}" style="width: {min(score, 100)}%;"></div>
                    </div>
                    <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0.8rem;">
                        {description}
                    </p>
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

    # Highlights
    highlights = summary.get("management_highlights", [])
    if highlights:
        st.markdown("### 🌟 Management Highlights")
        for h in highlights:
            st.markdown(f"- {h}")

    # Growth drivers
    drivers = summary.get("growth_drivers", [])
    if drivers:
        st.markdown("### 🚀 Growth Drivers")
        for d in drivers:
            st.markdown(f"- {d}")

    # Challenges
    challenges = summary.get("challenges", [])
    if challenges:
        st.markdown("### ⚡ Challenges")
        for c in challenges:
            st.markdown(f"- {c}")

    # Outlook
    outlook = summary.get("outlook", "")
    if outlook:
        st.markdown("### 🔮 Outlook")
        st.markdown(
            f"""<div class="summary-block">{outlook}</div>""",
            unsafe_allow_html=True,
        )

    # Key quotes
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

    # Build HTML table
    rows_html = ""
    for comp in comparisons:
        name = comp.get("metric_name", "")
        current = comp.get("current_value", "N/A") or "N/A"
        previous = comp.get("previous_value", "N/A") or "N/A"
        change_pct = comp.get("change_percent")
        trend = comp.get("trend", "→")

        # Trend styling
        if trend == "↑":
            trend_class = "trend-up"
            change_color = "#4ADE80"
        elif trend == "↓":
            trend_class = "trend-down"
            change_color = "#F87171"
        else:
            trend_class = "trend-flat"
            change_color = "#94A3B8"

        change_str = f"{change_pct:+.1f}%" if change_pct is not None else "—"

        rows_html += f"""
            <tr>
                <td style="padding: 12px 16px; color: #E2E8F0; font-weight: 500; border-bottom: 1px solid rgba(99,102,241,0.1);">
                    {name}
                </td>
                <td style="padding: 12px 16px; color: #CBD5E1; text-align: right; border-bottom: 1px solid rgba(99,102,241,0.1);">
                    {previous}
                </td>
                <td style="padding: 12px 16px; color: #F1F5F9; text-align: right; font-weight: 600; border-bottom: 1px solid rgba(99,102,241,0.1);">
                    {current}
                </td>
                <td style="padding: 12px 16px; text-align: right; color: {change_color}; font-weight: 600; border-bottom: 1px solid rgba(99,102,241,0.1);">
                    {change_str}
                </td>
                <td style="padding: 12px 16px; text-align: center; font-size: 1.2rem; border-bottom: 1px solid rgba(99,102,241,0.1);">
                    <span class="{trend_class}">{trend}</span>
                </td>
            </tr>
        """

    st.markdown(
        f"""
        <div class="glass-card" style="overflow-x: auto; padding: 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: rgba(99,102,241,0.1);">
                        <th style="padding: 14px 16px; text-align: left; color: #94A3B8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Metric</th>
                        <th style="padding: 14px 16px; text-align: right; color: #94A3B8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Previous</th>
                        <th style="padding: 14px 16px; text-align: right; color: #94A3B8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Current</th>
                        <th style="padding: 14px 16px; text-align: right; color: #94A3B8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Change</th>
                        <th style="padding: 14px 16px; text-align: center; color: #94A3B8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
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

    # JSON download
    with col1:
        json_str = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"{company_name.replace(' ', '_')}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    # Markdown download
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
    company = report.get("company_name", "Company")
    quarter = report.get("quarter", "")
    kpis = report.get("kpis", {})
    risk = report.get("risk_analysis", {})
    summary = report.get("summary", {})
    comparison = report.get("comparison")

    lines = [
        f"# {company} — {quarter} Earnings Analysis\n",
        f"*Generated by Multi-Agent Financial Analyst*\n",
        "---\n",
    ]

    # KPIs
    lines.append("## Financial KPIs\n")
    lines.append("| Metric | Value | Confidence |")
    lines.append("|--------|-------|------------|")

    metric_labels = {
        "revenue": "Revenue", "net_income": "Net Income",
        "operating_income": "Operating Income", "eps": "EPS",
        "cash_flow": "Cash Flow", "gross_margin": "Gross Margin",
        "operating_margin": "Operating Margin", "total_debt": "Total Debt",
        "total_assets": "Total Assets", "total_liabilities": "Total Liabilities",
    }
    confidence = kpis.get("confidence", {})
    for key, label in metric_labels.items():
        val = kpis.get(key)
        if val:
            conf = confidence.get(key, 0)
            lines.append(f"| {label} | {val} | {conf:.0%} |")
    lines.append("")

    # Risk
    lines.append("## Risk Analysis\n")
    lines.append(f"**Overall Score:** {risk.get('overall_score', 0):.0f}/100\n")
    lines.append(f"**Risk Level:** {risk.get('risk_level', 'N/A')}\n")
    if risk.get("summary"):
        lines.append(f"\n{risk['summary']}\n")

    # Summary
    if summary.get("executive_summary"):
        lines.append("## Executive Summary\n")
        lines.append(summary["executive_summary"])
        lines.append("")

    # Comparison
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
