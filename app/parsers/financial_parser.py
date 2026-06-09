"""
Deterministic regex-based financial metric extraction.

This module extracts KPIs (revenue, net income, EPS, margins, etc.) from
earnings-report text and tables **without** relying on an LLM.  Numbers
are parsed with configurable regex patterns that handle common financial
notation: ``$1.2B``, ``$1,234 million``, ``(1,234)``, ``$1.23/share``.

All public functions return structured data suitable for populating
:pyclass:`app.graph.state.KPIData`.
"""

import logging
import re
from typing import Any, Optional

from app.graph.state import KPIData

logger = logging.getLogger(__name__)


# ============================================================
# Number-parsing helpers
# ============================================================

# Multiplier keywords
_MULTIPLIERS: dict[str, float] = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "t": 1e12,
}

# Master pattern that captures a financial number with optional
# currency symbol, commas, decimals, parenthesised negatives, and
# scale suffixes (M / B / million / billion / etc.)
# Pattern order: neg_paren → neg_dash → currency → number → scale → close_paren
# neg_dash is placed BEFORE currency to handle both "-$1.2B" and "$-1.2B" formats.
_FIN_NUMBER_RE = re.compile(
    r"(?P<neg_paren>\()?"                             # optional opening paren
    r"\s*"
    r"(?P<neg_dash>-\s*)?"                             # optional dash negative (before currency)
    r"(?P<currency>[\$€£¥₹])?"                        # optional currency symbol
    r"\s*"
    r"(?P<number>[\d,]+(?:\.\d+)?)"                    # the number itself
    r"\s*"
    r"(?P<scale>[KkMmBbTt](?:illion|illions|housand|housands)?)?"  # scale suffix
    r"\s*"
    r"(?P<neg_paren_close>\))?"                        # optional closing paren
)


def _parse_financial_number(text: str) -> Optional[float]:
    """
    Parse a financial number string into a raw float.

    Handles the following formats:
    * ``$1,234.56``
    * ``$1.2B``, ``$1.2 billion``
    * ``$1,234M``, ``$1,234 million``
    * ``(1,234)`` — accounting negative
    * ``-1,234``
    * ``($1,234)``

    Args:
        text: A string containing a financial number.

    Returns:
        The parsed float, or ``None`` if parsing fails.
    """
    if not text:
        return None

    text = text.strip()
    match = _FIN_NUMBER_RE.search(text)
    if not match:
        return None

    try:
        raw = match.group("number").replace(",", "")
        value = float(raw)

        # Apply scale
        scale_str = (match.group("scale") or "").lower().strip()
        if scale_str:
            for key, mult in _MULTIPLIERS.items():
                if scale_str.startswith(key[0]) and (
                    len(scale_str) == 1 or scale_str == key
                ):
                    value *= mult
                    break

        # Apply negative
        if match.group("neg_paren") and match.group("neg_paren_close"):
            value = -value
        elif match.group("neg_dash"):
            value = -value

        return value
    except (ValueError, AttributeError) as exc:
        logger.debug("Failed to parse financial number '%s': %s", text, exc)
        return None


def _format_number(value: float) -> str:
    """Format a raw number into a human-readable financial string."""
    abs_val = abs(value)
    if abs_val >= 1e12:
        formatted = f"${value / 1e12:,.2f} trillion"
    elif abs_val >= 1e9:
        formatted = f"${value / 1e9:,.2f} billion"
    elif abs_val >= 1e6:
        formatted = f"${value / 1e6:,.2f} million"
    else:
        formatted = f"${value:,.2f}"
    if value < 0:
        formatted = formatted.replace("$", "$(") + ")"
        formatted = formatted.replace("$(-", "$(")
    return formatted


# ============================================================
# Individual KPI extractors
# ============================================================

# Each extractor returns (formatted_str, raw_number) or (None, None).

def _extract_revenue(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract revenue / net sales from text."""
    patterns = [
        re.compile(
            r"(?i)(?:total\s+)?(?:net\s+)?(?:revenue|sales)s?"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)",
        ),
        re.compile(
            r"(?i)(?:revenue|net\s+sales)\s+(?:increased?|decreased?|grew|declined?)"
            r"\s+to\s+([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)",
        ),
    ]
    return _try_patterns(patterns, text, "revenue")


def _extract_net_income(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract net income / net profit."""
    patterns = [
        re.compile(
            r"(?i)net\s+(?:income|profit|earnings)"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
        re.compile(
            r"(?i)net\s+(?:income|profit|earnings)"
            r"\s+(?:increased?|decreased?|grew|declined?)\s+to\s+"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
    ]
    return _try_patterns(patterns, text, "net_income")


def _extract_eps(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract earnings per share (diluted preferred)."""
    patterns = [
        re.compile(
            r"(?i)(?:diluted\s+)?(?:earnings?\s+per\s+share|EPS)"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*\)?)"
        ),
        re.compile(
            r"(?i)(?:diluted\s+)?EPS\s+of\s+"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*\)?)"
        ),
        re.compile(
            r"(?i)(\$\s*[\d]+(?:\.\d+)?)\s*(?:per\s+(?:diluted\s+)?share|/share)"
        ),
    ]
    return _try_patterns(patterns, text, "eps")


def _extract_operating_income(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract operating income / EBIT."""
    patterns = [
        re.compile(
            r"(?i)(?:operating\s+(?:income|profit|earnings)|EBIT(?!DA))"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
    ]
    return _try_patterns(patterns, text, "operating_income")


def _extract_cash_flow(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract operating cash flow."""
    patterns = [
        re.compile(
            r"(?i)(?:operating|net)\s+cash\s+flow"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
        re.compile(
            r"(?i)cash\s+(?:provided\s+by|from)\s+(?:operating\s+)?activities"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
    ]
    return _try_patterns(patterns, text, "cash_flow")


def _extract_debt(text: str) -> tuple[Optional[str], Optional[float]]:
    """Extract total debt / borrowings."""
    patterns = [
        re.compile(
            r"(?i)(?:total\s+)?(?:debt|borrowings|long[\s-]term\s+debt)"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"(\(?\s*[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*\)?)",
        ),
    ]
    return _try_patterns(patterns, text, "total_debt")


def _extract_assets_liabilities(text: str) -> dict[str, tuple[Optional[str], Optional[float]]]:
    """Extract total assets and total liabilities."""
    results: dict[str, tuple[Optional[str], Optional[float]]] = {}

    assets_patterns = [
        re.compile(
            r"(?i)total\s+assets"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)",
        ),
    ]
    results["total_assets"] = _try_patterns(assets_patterns, text, "total_assets")

    liabilities_patterns = [
        re.compile(
            r"(?i)total\s+liabilities"
            r"\s*(?:was|were|of|:|\s)\s*"
            r"([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)",
        ),
    ]
    results["total_liabilities"] = _try_patterns(
        liabilities_patterns, text, "total_liabilities"
    )

    return results


def _extract_margins(
    text: str,
    revenue: Optional[float],
    operating_income: Optional[float],
) -> dict[str, Optional[str]]:
    """
    Extract or calculate margin percentages.

    First attempts regex extraction from text.  If that fails and the
    raw revenue / operating-income values are available, computes the
    operating margin arithmetically.
    """
    margins: dict[str, Optional[str]] = {
        "gross_margin": None,
        "operating_margin": None,
    }

    # Try regex
    gm = re.search(
        r"(?i)gross\s+(?:profit\s+)?margin\s*(?:was|were|of|:|\s)\s*"
        r"([\d]+(?:\.\d+)?)\s*%",
        text,
    )
    if gm:
        margins["gross_margin"] = f"{gm.group(1)}%"

    om = re.search(
        r"(?i)operating\s+(?:profit\s+)?margin\s*(?:was|were|of|:|\s)\s*"
        r"([\d]+(?:\.\d+)?)\s*%",
        text,
    )
    if om:
        margins["operating_margin"] = f"{om.group(1)}%"

    # Compute operating margin from numbers if regex missed it
    if (
        margins["operating_margin"] is None
        and revenue
        and operating_income
        and revenue > 0
    ):
        pct = (operating_income / revenue) * 100
        margins["operating_margin"] = f"{pct:.1f}%"
        logger.debug("Computed operating margin: %s", margins["operating_margin"])

    return margins


# ============================================================
# Table-based extraction
# ============================================================

# Map of canonical KPI label → list of header-cell patterns to look for
_TABLE_HEADER_MAP: dict[str, list[str]] = {
    "revenue": ["revenue", "net sales", "total revenue", "net revenue"],
    "net_income": ["net income", "net profit", "net earnings"],
    "operating_income": ["operating income", "operating profit", "income from operations", "ebit"],
    "eps": ["diluted eps", "earnings per share", "diluted earnings per share", "eps"],
    "cash_flow": ["operating cash flow", "cash from operations", "net cash from operating"],
    "total_debt": ["total debt", "total borrowings", "long-term debt"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
}


def _extract_from_tables(tables: list[list[list[str]]]) -> dict[str, Any]:
    """
    Search extracted tables for financial metrics by header matching.

    Iterates every row of every table, checking if the first non-empty
    cell matches a known KPI label.  When a match is found, the last
    numeric cell in that row is taken as the value (commonly the most
    recent period).

    Args:
        tables: Nested list of tables (list of rows of cell strings).

    Returns:
        Dict mapping KPI names to ``(formatted_str, raw_value)`` tuples.
    """
    found: dict[str, Any] = {}

    for table in tables:
        for row in table:
            if not row:
                continue
            label_cell = (row[0] or "").strip().lower()
            if not label_cell:
                continue

            for kpi_name, header_variants in _TABLE_HEADER_MAP.items():
                if kpi_name in found:
                    continue  # already found this KPI
                if any(variant in label_cell for variant in header_variants):
                    # Grab the last numeric value in the row
                    value = _last_numeric_cell(row[1:])
                    if value is not None:
                        formatted = _format_number(value)
                        found[kpi_name] = (formatted, value)
                        logger.debug(
                            "Table extraction: %s = %s", kpi_name, formatted
                        )
                    break

    return found


def _last_numeric_cell(cells: list[str]) -> Optional[float]:
    """Return the parsed value of the last numeric cell in a row."""
    for cell in reversed(cells):
        val = _parse_financial_number(cell or "")
        if val is not None:
            return val
    return None


# ============================================================
# Helper
# ============================================================

def _try_patterns(
    patterns: list[re.Pattern],
    text: str,
    kpi_name: str,
) -> tuple[Optional[str], Optional[float]]:
    """Try each regex pattern; return the first successful parse."""
    for pat in patterns:
        match = pat.search(text)
        if match:
            raw_str = match.group(1)
            value = _parse_financial_number(raw_str)
            if value is not None:
                formatted = _format_number(value)
                logger.debug("Regex extraction [%s]: %s → %s", kpi_name, raw_str, formatted)
                return formatted, value
    return None, None


# ============================================================
# Main entry point
# ============================================================

def extract_kpis_from_text(
    text: str,
    tables: Optional[list[list[list[str]]]] = None,
) -> KPIData:
    """
    Extract all available KPIs from text and tables.

    Strategy:
        1. Regex extraction from body text (high priority).
        2. Table-header matching (fills gaps).
        3. Computed fields (e.g. operating margin from revenue & EBIT).

    Args:
        text:   Full or cleaned document text.
        tables: Optional list of tables (each a list of rows of strings).

    Returns:
        A populated :pyclass:`KPIData` instance.
    """
    logger.info("Starting deterministic KPI extraction")

    raw_values: dict[str, float] = {}
    confidence: dict[str, float] = {}

    # --- Step 1: regex-based extraction ---
    rev_str, rev_val = _extract_revenue(text)
    ni_str, ni_val = _extract_net_income(text)
    eps_str, eps_val = _extract_eps(text)
    oi_str, oi_val = _extract_operating_income(text)
    cf_str, cf_val = _extract_cash_flow(text)
    debt_str, debt_val = _extract_debt(text)
    al_results = _extract_assets_liabilities(text)
    ta_str, ta_val = al_results.get("total_assets", (None, None))
    tl_str, tl_val = al_results.get("total_liabilities", (None, None))

    # Store regex results with confidence
    regex_results: dict[str, tuple[Optional[str], Optional[float]]] = {
        "revenue": (rev_str, rev_val),
        "net_income": (ni_str, ni_val),
        "eps": (eps_str, eps_val),
        "operating_income": (oi_str, oi_val),
        "cash_flow": (cf_str, cf_val),
        "total_debt": (debt_str, debt_val),
        "total_assets": (ta_str, ta_val),
        "total_liabilities": (tl_str, tl_val),
    }

    for kpi, (_, val) in regex_results.items():
        if val is not None:
            raw_values[kpi] = val
            confidence[kpi] = 0.8  # regex = good confidence

    # --- Step 2: fill gaps from tables ---
    table_results: dict[str, Any] = {}
    if tables:
        table_results = _extract_from_tables(tables)
        for kpi, (tbl_str, tbl_val) in table_results.items():
            if kpi not in raw_values:
                raw_values[kpi] = tbl_val
                regex_results[kpi] = (tbl_str, tbl_val)
                confidence[kpi] = 0.9  # table = high confidence

    # --- Step 3: margins ---
    margins = _extract_margins(
        text,
        raw_values.get("revenue"),
        raw_values.get("operating_income"),
    )

    # --- Assemble KPIData ---
    kpi = KPIData(
        revenue=regex_results["revenue"][0],
        net_income=regex_results["net_income"][0],
        operating_income=regex_results["operating_income"][0],
        eps=regex_results["eps"][0],
        cash_flow=regex_results["cash_flow"][0],
        gross_margin=margins.get("gross_margin"),
        operating_margin=margins.get("operating_margin"),
        total_debt=regex_results["total_debt"][0],
        total_assets=regex_results["total_assets"][0],
        total_liabilities=regex_results["total_liabilities"][0],
        confidence=confidence,
        extraction_method="regex",
        raw_values=raw_values,
    )

    extracted_count = sum(
        1 for v in [kpi.revenue, kpi.net_income, kpi.eps, kpi.operating_income,
                     kpi.cash_flow, kpi.total_debt]
        if v is not None
    )
    logger.info(
        "KPI extraction complete — %d / 6 core metrics found", extracted_count
    )
    return kpi
