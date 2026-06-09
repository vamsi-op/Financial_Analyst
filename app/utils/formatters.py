"""
Output formatting utilities for the Financial Analyst application.

Provides helpers for:
- Currency, percentage, and change formatting
- Trend arrow indicators
- Risk level colour codes
- Full report serialisation (JSON and Markdown)
- Financial number parsing (``$1.2B``, ``1,234M``, ``(1,234)``, etc.)
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from app.graph.state import AnalysisReport, MetricComparison

logger = logging.getLogger(__name__)

# ================================================================
# Magnitude thresholds
# ================================================================
_TRILLION = 1_000_000_000_000.0
_BILLION = 1_000_000_000.0
_MILLION = 1_000_000.0
_THOUSAND = 1_000.0


# ================================================================
# Number formatting
# ================================================================

def format_currency(value: float, prefix: str = "$") -> str:
    """
    Format a numeric value as a human-readable currency string.

    Automatically selects the best magnitude suffix:
    - **T** for trillions
    - **B** for billions
    - **M** for millions
    - **K** for thousands
    - plain format for values below 1 000

    Args:
        value: The raw numeric value.
        prefix: Currency symbol (default ``"$"``).

    Returns:
        Formatted string, e.g. ``"$1.23B"`` or ``"$456.78M"``.
    """
    try:
        abs_val = abs(value)
        sign = "-" if value < 0 else ""

        if abs_val >= _TRILLION:
            return f"{sign}{prefix}{abs_val / _TRILLION:,.2f}T"
        if abs_val >= _BILLION:
            return f"{sign}{prefix}{abs_val / _BILLION:,.2f}B"
        if abs_val >= _MILLION:
            return f"{sign}{prefix}{abs_val / _MILLION:,.2f}M"
        if abs_val >= _THOUSAND:
            return f"{sign}{prefix}{abs_val / _THOUSAND:,.2f}K"
        return f"{sign}{prefix}{abs_val:,.2f}"
    except (TypeError, ValueError) as exc:
        logger.warning("format_currency failed for value=%r: %s", value, exc)
        return str(value)


def format_percentage(value: float) -> str:
    """
    Format a numeric value as a percentage string.

    Args:
        value: The numeric value (e.g. ``12.345``).

    Returns:
        Formatted string, e.g. ``"12.3%"``.
    """
    try:
        return f"{value:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def format_change(value: float) -> str:
    """
    Format a change value with sign prefix and colour hint.

    Positive changes are prefixed with ``+``, negative with ``-``.

    Args:
        value: Percentage change.

    Returns:
        Formatted string such as ``"+12.3%"`` or ``"-5.2%"``.
    """
    try:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def trend_arrow(change: float, threshold: float = 1.0) -> str:
    """
    Return a trend arrow character based on the magnitude of *change*.

    Args:
        change: Percentage change value.
        threshold: Minimum absolute change to register a direction
            (default ``1.0``).

    Returns:
        ``"↑"`` for positive, ``"↓"`` for negative, ``"→"`` for flat.
    """
    try:
        if change > threshold:
            return "↑"
        if change < -threshold:
            return "↓"
        return "→"
    except (TypeError, ValueError):
        return "→"


def risk_level_color(level: str) -> str:
    """
    Map a risk level label to a hex colour code.

    Args:
        level: One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.

    Returns:
        A CSS-compatible hex colour string.
    """
    colours = {
        "low": "#22C55E",       # green-500
        "medium": "#F59E0B",    # amber-500
        "high": "#EF4444",      # red-500
        "critical": "#991B1B",  # red-800
    }
    return colours.get(level.lower(), "#6B7280")  # gray-500 fallback


# ================================================================
# Report serialisation
# ================================================================

def format_report_as_json(report: AnalysisReport) -> str:
    """
    Serialise an :class:`AnalysisReport` to a pretty-printed JSON string.

    Args:
        report: The analysis report to serialise.

    Returns:
        Formatted JSON string.
    """
    try:
        return report.model_dump_json(indent=2)
    except Exception as exc:
        logger.error("Failed to serialise report to JSON: %s", exc)
        return json.dumps({"error": str(exc)}, indent=2)


def format_report_as_markdown(report: AnalysisReport) -> str:
    """
    Generate a Markdown document from an :class:`AnalysisReport`.

    The output follows this structure::

        # Financial Analysis Report — <Company> <Quarter>
        ## Key Performance Indicators
        ## Risk Analysis
        ## Earnings Summary
        ## Quarter-over-Quarter Comparison  (if available)

    Args:
        report: The analysis report to render.

    Returns:
        Markdown-formatted string.
    """
    try:
        lines: list[str] = []
        header = f"# 📊 Financial Analysis Report — {report.company_name}"
        if report.quarter:
            header += f" ({report.quarter})"
        lines.append(header)
        lines.append(f"*Generated: {report.report_date or datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        # ---- KPIs ----
        lines.append("## Key Performance Indicators")
        lines.append("")
        kpi = report.kpis
        kpi_rows = [
            ("Revenue", kpi.revenue),
            ("Net Income", kpi.net_income),
            ("Operating Income", kpi.operating_income),
            ("EPS", kpi.eps),
            ("Operating Cash Flow", kpi.cash_flow),
            ("Free Cash Flow", kpi.free_cash_flow),
            ("Gross Margin", kpi.gross_margin),
            ("Operating Margin", kpi.operating_margin),
            ("Total Debt", kpi.total_debt),
            ("Total Assets", kpi.total_assets),
            ("Total Liabilities", kpi.total_liabilities),
        ]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for label, val in kpi_rows:
            display = val if val else "—"
            lines.append(f"| {label} | {display} |")
        lines.append("")

        # ---- Risk ----
        risk = report.risk_analysis
        risk_colour = risk_level_color(risk.risk_level)
        lines.append("## Risk Analysis")
        lines.append("")
        lines.append(
            f"**Overall Risk Score:** {risk.overall_score:.0f}/100 "
            f"(**{risk.risk_level.upper()}**)"
        )
        lines.append("")
        if risk.summary:
            lines.append(risk.summary)
            lines.append("")
        if risk.factors:
            lines.append("| Category | Score | Severity | Description |")
            lines.append("|----------|-------|----------|-------------|")
            for f in risk.factors:
                lines.append(
                    f"| {f.category} | {f.score:.0f} | {f.severity} | {f.description} |"
                )
            lines.append("")
        if risk.recommendations:
            lines.append("### Recommendations")
            for rec in risk.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # ---- Summary ----
        summary = report.summary
        lines.append("## Earnings Summary")
        lines.append("")
        if summary.executive_summary:
            lines.append(summary.executive_summary)
            lines.append("")
        if summary.management_highlights:
            lines.append("### Management Highlights")
            for h in summary.management_highlights:
                lines.append(f"- {h}")
            lines.append("")
        if summary.growth_drivers:
            lines.append("### Growth Drivers")
            for d in summary.growth_drivers:
                lines.append(f"- {d}")
            lines.append("")
        if summary.challenges:
            lines.append("### Challenges")
            for c in summary.challenges:
                lines.append(f"- {c}")
            lines.append("")
        if summary.outlook:
            lines.append("### Outlook")
            lines.append(summary.outlook)
            lines.append("")
        if summary.key_quotes:
            lines.append("### Key Quotes")
            for q in summary.key_quotes:
                lines.append(f'> "{q}"')
            lines.append("")

        # ---- Comparison ----
        comp = report.comparison
        if comp and comp.comparisons:
            lines.append("## Quarter-over-Quarter Comparison")
            lines.append(
                f"*{comp.previous_quarter} → {comp.current_quarter}*"
            )
            lines.append("")
            lines.append("| Metric | Previous | Current | Change | Trend |")
            lines.append("|--------|----------|---------|--------|-------|")
            for m in comp.comparisons:
                chg = format_change(m.change_percent) if m.change_percent is not None else "—"
                lines.append(
                    f"| {m.metric_name} | {m.previous_value or '—'} | "
                    f"{m.current_value or '—'} | {chg} | {m.trend} |"
                )
            lines.append("")
            if comp.executive_insights:
                lines.append("### Insights")
                lines.append(comp.executive_insights)
                lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        logger.error("Failed to render Markdown report: %s", exc)
        return f"# Report Generation Error\n\n{exc}"


# ================================================================
# Financial number parsing
# ================================================================

# Pre-compiled patterns used by parse_financial_number
_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "t": _TRILLION,
    "tr": _TRILLION,
    "trillion": _TRILLION,
    "b": _BILLION,
    "bn": _BILLION,
    "billion": _BILLION,
    "m": _MILLION,
    "mn": _MILLION,
    "mm": _MILLION,
    "mil": _MILLION,
    "million": _MILLION,
    "k": _THOUSAND,
    "thousand": _THOUSAND,
}

_NUMBER_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<negative>[\(\-])?\s*     # optional negative sign: ( or -
    (?P<currency>[^\d\s\(\-])?   # optional currency symbol
    \s*
    (?P<number>[\d,]+            # integer part (with optional commas)
        (?:\.\d+)?               # optional decimal part
    )
    \s*
    (?P<suffix>[A-Za-z]*)        # optional magnitude suffix
    \s*
    (?P<close_paren>\))?         # optional closing parenthesis
    \s*$
    """,
    re.VERBOSE,
)


def parse_financial_number(text: str) -> Optional[float]:
    """
    Parse a financial number string into a float.

    Handles the wide variety of formats found in earnings reports:

    - ``"$1.2B"`` → ``1_200_000_000.0``
    - ``"1,234M"`` → ``1_234_000_000.0``
    - ``"(1,234)"`` → ``-1234.0``  (accounting negative)
    - ``"-$5.3M"`` → ``-5_300_000.0``
    - ``"12.3%"`` → ``12.3``
    - ``"$1,234,567"`` → ``1_234_567.0``
    - ``"45.2"`` → ``45.2``

    Args:
        text: The raw string to parse.

    Returns:
        Parsed float, or ``None`` if the string cannot be interpreted.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # ---- Handle percentage shortcut ----
    if cleaned.endswith("%"):
        try:
            return float(cleaned.rstrip("%").replace(",", "").strip())
        except ValueError:
            return None

    match = _NUMBER_PATTERN.match(cleaned)
    if not match:
        return None

    try:
        negative = match.group("negative") or match.group("close_paren")
        number_str = match.group("number").replace(",", "")
        suffix = (match.group("suffix") or "").lower().strip()

        value = float(number_str)

        # Apply magnitude suffix
        if suffix:
            multiplier = _SUFFIX_MULTIPLIERS.get(suffix)
            if multiplier:
                value *= multiplier
            # Ignore unrecognised suffixes silently (may just be units)

        # Apply sign
        if negative:
            value = -abs(value)

        return value

    except (ValueError, AttributeError) as exc:
        logger.debug("parse_financial_number failed for %r: %s", text, exc)
        return None
