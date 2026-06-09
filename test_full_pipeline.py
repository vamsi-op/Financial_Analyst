"""
Full end-to-end pipeline test using Groq + synthetic ACME Corp data.
Run:  python test_full_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.graph.workflow import run_analysis
from app.utils.llm import get_active_provider, get_active_model

DIVIDER = "=" * 62

print(DIVIDER)
print(f"  LLM Provider : {get_active_provider().upper()}")
print(f"  Model        : {get_active_model()}")
print(DIVIDER)
print("Running FULL pipeline on: ACME Corp Q3 2025 (synthetic data)")
print("Including: PDF parse → KPI extract → Risk → AI Summary → QoQ Compare")
print()

# Previous quarter KPIs for comparison
prev_kpis = {
    "revenue": "$43.10 billion",
    "net_income": "$10.50 billion",
    "operating_income": "$13.80 billion",
    "eps": "$2.64",
    "cash_flow": "$15.90 billion",
    "gross_margin": "46.5%",
    "operating_margin": "32.0%",
    "total_debt": "$89.20 billion",
    "total_assets": "$280.10 billion",
    "total_liabilities": "$196.70 billion",
    "raw_values": {
        "revenue": 43.1e9, "net_income": 10.5e9,
        "operating_income": 13.8e9, "eps": 2.64,
        "cash_flow": 15.9e9, "total_debt": 89.2e9,
        "total_assets": 280.1e9, "total_liabilities": 196.7e9,
    }
}

state = run_analysis(
    pdf_path="data/sample/ACME_Corp_Q3_2025_Earnings.txt",
    company_name="ACME Corporation",
    quarter="Q3 2025",
    previous_kpis=prev_kpis,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
print(f"\n{'KPI EXTRACTION':^62}")
print(DIVIDER)
kpi_fields = [
    ("revenue", "Revenue"),
    ("net_income", "Net Income"),
    ("operating_income", "Operating Income"),
    ("eps", "EPS (Diluted)"),
    ("cash_flow", "Cash Flow"),
    ("free_cash_flow", "Free Cash Flow"),
    ("gross_margin", "Gross Margin"),
    ("operating_margin", "Operating Margin"),
    ("total_debt", "Total Debt"),
    ("total_assets", "Total Assets"),
    ("total_liabilities", "Total Liabilities"),
]
kpis = state.kpis
found = 0
for attr, label in kpi_fields:
    val = getattr(kpis, attr, None)
    if val:
        conf = kpis.confidence.get(attr, 0)
        conf_str = f"[{conf:.0%}]" if conf else ""
        print(f"  {label:<24} {val:<26} {conf_str}")
        found += 1
print(f"\n  Extracted {found}/{len(kpi_fields)} metrics | method: {kpis.extraction_method}")

# ── Risk ──────────────────────────────────────────────────────────────────────
print(f"\n{'RISK ANALYSIS':^62}")
print(DIVIDER)
risk = state.risk_analysis
bar_len = 40
bar = int(risk.overall_score / 100 * bar_len)
bar_str = chr(9608) * bar + chr(9617) * (bar_len - bar)
print(f"  Score  : {risk.overall_score:.1f}/100  -> {risk.risk_level.upper()}")
print(f"  [{bar_str}]")
if risk.factors:
    print(f"\n  Factor Breakdown:")
    for f in risk.factors:
        fbar = int(f.score / 100 * 20)
        fbar_str = chr(9608) * fbar + chr(9617) * (20 - fbar)
        print(f"    {f.category:<22} [{fbar_str}] {f.score:.0f}/100 ({f.severity.upper()})")

# ── AI Summary ────────────────────────────────────────────────────────────────
print(f"\n{'AI EXECUTIVE SUMMARY  (powered by Groq)':^62}")
print(DIVIDER)
summary = state.summary
if summary.executive_summary:
    # Word-wrap at 60 chars
    words = summary.executive_summary.split()
    line = "  "
    for word in words:
        if len(line) + len(word) > 62:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
else:
    print("  (No summary generated)")

if summary.growth_drivers:
    print(f"\n  Growth Drivers:")
    for d in summary.growth_drivers[:3]:
        print(f"    + {d}")

if summary.challenges:
    print(f"\n  Challenges:")
    for c in summary.challenges[:3]:
        print(f"    - {c}")

if summary.outlook:
    print(f"\n  Outlook: {summary.outlook[:200]}")

# ── QoQ Comparison ────────────────────────────────────────────────────────────
if state.comparison:
    print(f"\n{'QUARTER COMPARISON  Q2 2025 -> Q3 2025':^62}")
    print(DIVIDER)
    comp = state.comparison
    print(f"  Overall Trend: {comp.overall_trend.upper()}")
    print()
    print(f"  {'Metric':<22} {'Q2 2025':<18} {'Q3 2025':<18} {'Change':>8}  Trend")
    print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*8}  -----")
    for c in comp.comparisons[:6]:
        pct = f"{c.change_percent:+.1f}%" if c.change_percent is not None else "N/A"
        prev = str(c.previous_value or "N/A")
        curr = str(c.current_value or "N/A")
        print(f"  {c.metric_name:<22} {prev:<18} {curr:<18} {pct:>8}  {c.trend}")
    if comp.executive_insights:
        print(f"\n  Insights: {comp.executive_insights[:250]}")
else:
    print("\n  (No comparison data)")

# ── Footer ────────────────────────────────────────────────────────────────────
print()
print(DIVIDER)
if state.errors:
    print(f"  Errors ({len(state.errors)}):")
    for e in state.errors:
        print(f"    ! {e}")
else:
    print("  Errors: None")
print(DIVIDER)
