"""
Typed state models for the LangGraph workflow.

These Pydantic models define the data contract between all agents.
Every agent reads from and writes to this shared state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field


# ============================================================
# KPI Models
# ============================================================
class KPIData(BaseModel):
    """Extracted financial KPIs from an earnings report."""

    revenue: Optional[str] = Field(None, description="Total revenue / net sales")
    net_income: Optional[str] = Field(None, description="Net income / net profit")
    operating_income: Optional[str] = Field(None, description="Operating income / EBIT")
    eps: Optional[str] = Field(None, description="Earnings per share (diluted)")
    cash_flow: Optional[str] = Field(None, description="Operating cash flow")
    gross_margin: Optional[str] = Field(None, description="Gross margin percentage")
    operating_margin: Optional[str] = Field(None, description="Operating margin percentage")
    total_debt: Optional[str] = Field(None, description="Total debt / borrowings")
    total_assets: Optional[str] = Field(None, description="Total assets")
    total_liabilities: Optional[str] = Field(None, description="Total liabilities")
    free_cash_flow: Optional[str] = Field(None, description="Free cash flow")

    # Metadata
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score (0-1) for each extracted metric",
    )
    extraction_method: str = Field(
        default="hybrid",
        description="Method used: 'regex', 'llm', or 'hybrid'",
    )
    raw_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw numeric values for calculations",
    )


# ============================================================
# Risk Analysis Models
# ============================================================
class RiskFactor(BaseModel):
    """A single identified risk factor."""

    category: str = Field(..., description="Risk category name")
    score: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    severity: str = Field(..., description="low / medium / high / critical")
    description: str = Field(..., description="Detailed explanation")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence from report")


class RiskAnalysis(BaseModel):
    """Complete risk analysis output."""

    overall_score: float = Field(..., ge=0, le=100, description="Aggregate risk score")
    risk_level: str = Field(..., description="low / medium / high / critical")
    factors: list[RiskFactor] = Field(default_factory=list)
    summary: str = Field(default="", description="Executive risk summary")
    recommendations: list[str] = Field(default_factory=list)


# ============================================================
# Summary Models
# ============================================================
class EarningsSummary(BaseModel):
    """Executive earnings summary output."""

    executive_summary: str = Field(default="", description="200-500 word executive summary")
    management_highlights: list[str] = Field(default_factory=list)
    growth_drivers: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    outlook: str = Field(default="", description="Forward-looking guidance")
    key_quotes: list[str] = Field(default_factory=list, description="Notable management quotes")


# ============================================================
# Comparison Models
# ============================================================
class MetricComparison(BaseModel):
    """Comparison of a single metric between quarters."""

    metric_name: str
    current_value: Optional[str] = None
    previous_value: Optional[str] = None
    change_amount: Optional[str] = None
    change_percent: Optional[float] = None
    trend: str = Field(default="→", description="↑ ↓ → trend indicator")


class QuarterComparison(BaseModel):
    """Full quarter-over-quarter comparison."""

    current_quarter: str = Field(default="", description="e.g., Q3 2025")
    previous_quarter: str = Field(default="", description="e.g., Q2 2025")
    comparisons: list[MetricComparison] = Field(default_factory=list)
    executive_insights: str = Field(default="")
    overall_trend: str = Field(default="stable", description="improving / stable / declining")


# ============================================================
# Report Models
# ============================================================
class AnalysisReport(BaseModel):
    """Final consolidated analysis report."""

    company_name: str = ""
    quarter: str = ""
    report_date: str = ""
    kpis: KPIData = Field(default_factory=KPIData)
    risk_analysis: RiskAnalysis = Field(
        default_factory=lambda: RiskAnalysis(overall_score=0, risk_level="low")
    )
    summary: EarningsSummary = Field(default_factory=EarningsSummary)
    comparison: Optional[QuarterComparison] = None


# ============================================================
# LangGraph Pipeline State
# ============================================================
class PipelineState(BaseModel):
    """
    Shared state that flows through the LangGraph workflow.

    Every node reads from and writes to this state object.
    """

    # --- Input ---
    pdf_path: str = Field(default="", description="Path to uploaded PDF")
    raw_text: str = Field(default="", description="Full extracted text")
    pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-page extraction: [{page_num, text, tables}]",
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted tables as list of dicts",
    )
    chunks: list[str] = Field(
        default_factory=list,
        description="Text chunks for RAG retrieval",
    )

    # --- Agent outputs ---
    kpis: KPIData = Field(default_factory=KPIData)
    risk_analysis: RiskAnalysis = Field(
        default_factory=lambda: RiskAnalysis(overall_score=0, risk_level="low")
    )
    summary: EarningsSummary = Field(default_factory=EarningsSummary)
    comparison: Optional[QuarterComparison] = None

    # --- Metadata ---
    company_name: str = Field(default="", description="Detected or provided company name")
    quarter: str = Field(default="", description="Detected or provided quarter (e.g., Q3 2025)")
    previous_kpis: Optional[KPIData] = Field(
        None, description="Previous quarter KPIs for comparison"
    )

    # --- Final output ---
    final_report: Optional[AnalysisReport] = None

    # --- Error tracking ---
    # Annotated with operator.add so parallel nodes can each append errors
    # without LangGraph raising InvalidUpdateError.
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    warnings: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- Processing flags ---
    ocr_used: bool = Field(default=False, description="Whether OCR was needed")
    processing_complete: bool = Field(default=False)

    model_config = {"arbitrary_types_allowed": True}
