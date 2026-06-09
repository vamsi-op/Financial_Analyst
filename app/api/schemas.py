"""
Pydantic request / response schemas for the FastAPI endpoints.

These models define the API contract.  The frontend and any external
clients depend on these shapes being stable.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# Request models
# ============================================================

class AnalysisRequest(BaseModel):
    """Body for the ``POST /analyze`` endpoint (sent as form data)."""

    company_name: str = Field(default="", description="Override company name detection")
    quarter: str = Field(default="", description="Override quarter detection (e.g. Q3 2025)")
    previous_kpis: Optional[dict[str, Any]] = Field(
        default=None,
        description="Previous quarter KPIs for comparison (JSON object)",
    )


# ============================================================
# Response models
# ============================================================

class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = "ok"
    ollama_connected: bool = False
    active_model: str = ""
    available_models: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response after file upload."""

    job_id: str
    filename: str
    status: str = "processing"
    message: str = ""


class KPIResponse(BaseModel):
    """KPI extraction results."""

    revenue: Optional[str] = None
    net_income: Optional[str] = None
    operating_income: Optional[str] = None
    eps: Optional[str] = None
    cash_flow: Optional[str] = None
    gross_margin: Optional[str] = None
    operating_margin: Optional[str] = None
    total_debt: Optional[str] = None
    total_assets: Optional[str] = None
    total_liabilities: Optional[str] = None
    free_cash_flow: Optional[str] = None
    confidence: dict[str, float] = Field(default_factory=dict)
    extraction_method: str = "hybrid"


class RiskFactorResponse(BaseModel):
    """Single risk factor."""

    category: str
    score: float
    severity: str
    description: str
    evidence: list[str] = Field(default_factory=list)


class RiskResponse(BaseModel):
    """Risk analysis results."""

    overall_score: float = 0
    risk_level: str = "low"
    factors: list[RiskFactorResponse] = Field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    """Earnings summary results."""

    executive_summary: str = ""
    management_highlights: list[str] = Field(default_factory=list)
    growth_drivers: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    outlook: str = ""
    key_quotes: list[str] = Field(default_factory=list)


class MetricComparisonResponse(BaseModel):
    """Single metric comparison."""

    metric_name: str
    current_value: Optional[str] = None
    previous_value: Optional[str] = None
    change_amount: Optional[str] = None
    change_percent: Optional[float] = None
    trend: str = "→"


class ComparisonResponse(BaseModel):
    """Quarter comparison results."""

    current_quarter: str = ""
    previous_quarter: str = ""
    comparisons: list[MetricComparisonResponse] = Field(default_factory=list)
    executive_insights: str = ""
    overall_trend: str = "stable"


class FullReportResponse(BaseModel):
    """Complete analysis report."""

    company_name: str = ""
    quarter: str = ""
    report_date: str = ""
    kpis: KPIResponse = Field(default_factory=KPIResponse)
    risk_analysis: RiskResponse = Field(default_factory=RiskResponse)
    summary: SummaryResponse = Field(default_factory=SummaryResponse)
    comparison: Optional[ComparisonResponse] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    """Status of an analysis job."""

    job_id: str
    status: str = "pending"  # pending | processing | complete | failed
    progress: float = 0.0   # 0.0 - 1.0
    current_step: str = ""
    result: Optional[FullReportResponse] = None
    error: Optional[str] = None
