"""
Quick demo that runs the full deterministic pipeline on synthetic data.
No LLM / Ollama needed — only regex KPI extraction, rule-based risk scoring,
and formatted output.

Run:  python demo_no_llm.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from app.parsers.financial_parser import extract_kpis_from_text
from app.parsers.text_cleaner import clean_text, detect_sections, extract_company_info
from app.utils.formatters import (
    format_currency, format_percentage, trend_arrow,
    format_report_as_markdown, risk_level_color
)
from app.graph.state import KPIData, RiskAnalysis, RiskFactor, EarningsSummary, AnalysisReport

DIVIDER = "=" * 65


def score_risk(kpis: KPIData) -> RiskAnalysis:
    """Deterministic risk scoring — no LLM needed."""
    factors = []
    raw = kpis.raw_values

    # --- Revenue health ---
    rev = raw.get("revenue", 0)
    rev_score = 20 if rev > 50e9 else 40 if rev > 10e9 else 60
    factors.append(RiskFactor(
        category="Revenue Scale",
        score=rev_score,
        severity="low" if rev_score < 35 else "medium" if rev_score < 60 else "high",
        description=f"Revenue of {kpis.revenue or 'N/A'}",
        evidence=[f"Extracted revenue: {kpis.revenue}"],
    ))

    # --- Profitability ---
    ni = raw.get("net_income", 0)
    rev_val = raw.get("revenue", 1)
    net_margin = (ni / rev_val * 100) if rev_val else 0
    profit_score = 15 if net_margin > 20 else 35 if net_margin > 10 else 55 if net_margin > 0 else 85
    factors.append(RiskFactor(
        category="Profitability",
        score=profit_score,
        severity="low" if profit_score < 35 else "medium" if profit_score < 60 else "high",
        description=f"Net margin: {net_margin:.1f}%",
        evidence=[f"Net income: {kpis.net_income}", f"Revenue: {kpis.revenue}"],
    ))

    # --- Debt level ---
    debt = raw.get("total_debt", 0)
    assets = raw.get("total_assets", 1)
    debt_ratio = (debt / assets * 100) if assets else 0
    debt_score = 20 if debt_ratio < 30 else 45 if debt_ratio < 60 else 70
    factors.append(RiskFactor(
        category="Debt Level",
        score=debt_score,
        severity="low" if debt_score < 35 else "medium" if debt_score < 60 else "high",
        description=f"Debt-to-assets ratio: {debt_ratio:.1f}%",
        evidence=[f"Total debt: {kpis.total_debt}", f"Total assets: {kpis.total_assets}"],
    ))

    # --- Cash flow ---
    cf = raw.get("cash_flow", 0)
    cf_score = 15 if cf > 10e9 else 30 if cf > 1e9 else 60 if cf > 0 else 85
    factors.append(RiskFactor(
        category="Cash Flow Health",
        score=cf_score,
        severity="low" if cf_score < 35 else "medium" if cf_score < 60 else "high",
        description=f"Operating cash flow: {kpis.cash_flow or 'N/A'}",
        evidence=[f"Cash flow from operations: {kpis.cash_flow}"],
    ))

    # --- Overall ---
    weights = [0.30, 0.30, 0.20, 0.20]
    overall = sum(f.score * w for f, w in zip(factors, weights))
    level = "low" if overall < 30 else "medium" if overall < 60 else "high" if overall < 80 else "critical"

    return RiskAnalysis(
        overall_score=round(overall, 1),
        risk_level=level,
        factors=factors,
        summary=f"Deterministic risk assessment (no LLM). Overall score: {overall:.1f}/100.",
        recommendations=[
            "Enable Ollama or a free API (Groq/Gemini) for AI-powered insights.",
            "Review debt levels relative to asset base.",
            "Monitor operating cash flow trends quarter-over-quarter.",
        ],
    )


def print_kpis(kpis: KPIData):
    print(f"\n{'KPI EXTRACTION RESULTS':^65}")
    print(DIVIDER)
    fields = [
        ("Revenue",           kpis.revenue),
        ("Net Income",        kpis.net_income),
        ("Operating Income",  kpis.operating_income),
        ("EPS (Diluted)",     kpis.eps),
        ("Cash Flow (Ops)",   kpis.cash_flow),
        ("Free Cash Flow",    kpis.free_cash_flow),
        ("Gross Margin",      kpis.gross_margin),
        ("Operating Margin",  kpis.operating_margin),
        ("Total Debt",        kpis.total_debt),
        ("Total Assets",      kpis.total_assets),
        ("Total Liabilities", kpis.total_liabilities),
    ]
    for label, val in fields:
        if val:
            conf = kpis.confidence.get(label.lower().replace(" ", "_"), 0)
            conf_str = f"[conf: {conf:.0%}]" if conf else ""
            print(f"  {label:<22} {val:<25} {conf_str}")
    extracted = sum(1 for _, v in fields if v)
    print(f"\n  Extracted {extracted}/{len(fields)} metrics via deterministic regex.")
    print(f"  Extraction method: {kpis.extraction_method}")


def print_risk(risk: RiskAnalysis):
    bar_len = 40
    print(f"\n{'RISK ANALYSIS':^65}")
    print(DIVIDER)
    bar = int(risk.overall_score / 100 * bar_len)
    bar_str = "█" * bar + "░" * (bar_len - bar)
    print(f"\n  Overall Score : {risk.overall_score:.1f}/100")
    print(f"  [{bar_str}]")
    print(f"  Risk Level    : {risk.risk_level.upper()}")
    print(f"\n  Factor Breakdown:")
    for f in risk.factors:
        fbar = int(f.score / 100 * 20)
        fbar_str = "█" * fbar + "░" * (20 - fbar)
        print(f"    {f.category:<22} [{fbar_str}] {f.score:.0f}/100  ({f.severity.upper()})")


def main():
    sample_path = os.path.join("data", "sample", "ACME_Corp_Q3_2025_Earnings.txt")
    q2_path     = os.path.join("data", "sample", "ACME_Corp_Q2_2025_Earnings.txt")
    edge_path   = os.path.join("data", "sample", "Edge_Case_Negative_Report.txt")

    for label, path in [
        ("ACME Corp — Q3 2025 (Healthy Company)", sample_path),
        ("Struggling Industries — Q4 2025 (Edge Case: Negative)", edge_path),
    ]:
        print(f"\n\n{'#'*65}")
        print(f"  REPORT: {label}")
        print(f"{'#'*65}")

        with open(path, encoding="utf-8") as f:
            raw_text = f.read()

        # --- Clean ---
        cleaned = clean_text(raw_text)
        info    = extract_company_info(cleaned)
        print(f"\n  Detected company : {info.get('company_name', 'Unknown')}")
        print(f"  Detected quarter : {info.get('quarter', 'Unknown')}")

        # --- Extract KPIs ---
        kpis = extract_kpis_from_text(cleaned)
        print_kpis(kpis)

        # --- Risk score ---
        risk = score_risk(kpis)
        print_risk(risk)

    # --- QoQ Comparison (Q3 vs Q2) ---
    print(f"\n\n{'#'*65}")
    print(f"  QUARTER COMPARISON: ACME Q3 2025 vs Q2 2025")
    print(f"{'#'*65}")

    with open(sample_path, encoding="utf-8") as f:
        q3_text = f.read()
    with open(q2_path, encoding="utf-8") as f:
        q2_text = f.read()

    q3_kpis = extract_kpis_from_text(clean_text(q3_text))
    q2_kpis = extract_kpis_from_text(clean_text(q2_text))

    metrics = ["revenue", "net_income", "operating_income", "eps", "cash_flow"]
    print(f"\n  {'Metric':<22} {'Q2 2025':<20} {'Q3 2025':<20} {'Change':>10}  Trend")
    print(f"  {'-'*22} {'-'*20} {'-'*20} {'-'*10}  -----")

    for m in metrics:
        prev = q2_kpis.raw_values.get(m)
        curr = q3_kpis.raw_values.get(m)
        prev_str = getattr(q2_kpis, m) or "N/A"
        curr_str = getattr(q3_kpis, m) or "N/A"
        if prev and curr and prev != 0:
            pct = (curr - prev) / abs(prev) * 100
            arrow = trend_arrow(pct)
            change_str = f"{pct:+.1f}%"
        else:
            change_str = "N/A"
            arrow = "?"
        label = m.replace("_", " ").title()
        print(f"  {label:<22} {prev_str:<20} {curr_str:<20} {change_str:>10}  {arrow}")

    print(f"\n{'='*65}")
    print("  Demo complete. No LLM required for the above output!")
    print("  For AI summaries, install Ollama OR add a free Groq/Gemini API key.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
