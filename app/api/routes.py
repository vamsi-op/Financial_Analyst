"""
API route handlers for the Financial Analyst.

Endpoints:
    GET  /health          — Server + Ollama health check
    POST /analyze         — Upload PDF and run full analysis
    GET  /status/{job_id} — Check job status
    GET  /results/{job_id} — Get completed results
"""

import logging
import os
import uuid
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    ComparisonResponse,
    FullReportResponse,
    HealthResponse,
    JobStatusResponse,
    KPIResponse,
    MetricComparisonResponse,
    RiskFactorResponse,
    RiskResponse,
    SummaryResponse,
    UploadResponse,
)
from app.config import UPLOAD_DIR, get_config

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job store (production would use Redis/DB)
_jobs: dict[str, dict] = {}


# ============================================================
# Health check
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and LLM provider health."""
    response = HealthResponse()

    try:
        from app.utils.llm import get_active_provider, get_active_model, check_ollama_running, _groq_available

        provider = get_active_provider()
        response.active_model = get_active_model()

        if provider == "groq":
            response.ollama_connected = True   # reuse field — means "LLM available"
            response.available_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
            response.status = "ok"
        elif provider == "ollama" and check_ollama_running():
            response.ollama_connected = True
            from app.utils.llm import list_available_models
            response.available_models = list_available_models()
            response.status = "ok"
        else:
            response.status = "degraded"
    except Exception as e:
        logger.warning("Health check error: %s", e)
        response.status = "degraded"

    return response


# ============================================================
# Upload & Analyse
# ============================================================

@router.post("/analyze", response_model=UploadResponse)
async def analyze_report(
    file: UploadFile = File(..., description="PDF earnings report"),
    company_name: str = Form(default="", description="Company name override"),
    quarter: str = Form(default="", description="Quarter override (e.g. Q3 2025)"),
    previous_kpis_json: str = Form(default="", description="Previous quarter KPIs as JSON string"),
):
    """
    Upload a PDF earnings report and start the analysis pipeline.

    Returns a job ID that can be used to poll for results.
    """
    # Validate file type
    if file.content_type not in ("application/pdf",):
        if not (file.filename and file.filename.lower().endswith(".pdf")):
            raise HTTPException(
                status_code=415,
                detail=f"Only PDF files are supported. Got: {file.content_type}",
            )

    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "report.pdf")[1]
    saved_name = f"{job_id}{ext}"
    file_path = str(UPLOAD_DIR / saved_name)

    try:
        async with aiofiles.open(file_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    # Parse previous KPIs if provided
    previous_kpis = None
    if previous_kpis_json.strip():
        import json
        try:
            previous_kpis = json.loads(previous_kpis_json)
        except json.JSONDecodeError:
            logger.warning("Invalid previous_kpis JSON — ignoring")

    # Initialize job
    _jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "current_step": "Starting analysis...",
        "file_path": file_path,
        "result": None,
        "error": None,
    }

    # Run analysis in background (synchronous for now, async in production)
    import threading
    thread = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, file_path, company_name, quarter, previous_kpis),
        daemon=True,
    )
    thread.start()

    return UploadResponse(
        job_id=job_id,
        filename=file.filename or "report.pdf",
        status="processing",
        message="Analysis started. Use GET /status/{job_id} to check progress.",
    )


def _run_analysis_job(
    job_id: str,
    file_path: str,
    company_name: str,
    quarter: str,
    previous_kpis: Optional[dict],
):
    """Run the analysis pipeline in a background thread."""
    try:
        _jobs[job_id]["current_step"] = "Running analysis pipeline..."
        _jobs[job_id]["progress"] = 0.1

        from app.graph.workflow import run_analysis

        result = run_analysis(
            pdf_path=file_path,
            company_name=company_name,
            quarter=quarter,
            previous_kpis=previous_kpis,
        )

        # Convert state to response model
        report = _state_to_response(result)
        _jobs[job_id]["result"] = report
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["current_step"] = "Complete"
        logger.info("Job %s completed successfully", job_id)

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        _jobs[job_id]["current_step"] = "Failed"


def _state_to_response(state) -> FullReportResponse:
    """Convert a PipelineState to FullReportResponse."""
    # KPIs
    kpi_resp = KPIResponse(
        revenue=state.kpis.revenue,
        net_income=state.kpis.net_income,
        operating_income=state.kpis.operating_income,
        eps=state.kpis.eps,
        cash_flow=state.kpis.cash_flow,
        gross_margin=state.kpis.gross_margin,
        operating_margin=state.kpis.operating_margin,
        total_debt=state.kpis.total_debt,
        total_assets=state.kpis.total_assets,
        total_liabilities=state.kpis.total_liabilities,
        free_cash_flow=state.kpis.free_cash_flow,
        confidence=state.kpis.confidence,
        extraction_method=state.kpis.extraction_method,
    )

    # Risk
    risk_factors = [
        RiskFactorResponse(
            category=f.category,
            score=f.score,
            severity=f.severity,
            description=f.description,
            evidence=f.evidence,
        )
        for f in state.risk_analysis.factors
    ]
    risk_resp = RiskResponse(
        overall_score=state.risk_analysis.overall_score,
        risk_level=state.risk_analysis.risk_level,
        factors=risk_factors,
        summary=state.risk_analysis.summary,
        recommendations=state.risk_analysis.recommendations,
    )

    # Summary
    summary_resp = SummaryResponse(
        executive_summary=state.summary.executive_summary,
        management_highlights=state.summary.management_highlights,
        growth_drivers=state.summary.growth_drivers,
        challenges=state.summary.challenges,
        outlook=state.summary.outlook,
        key_quotes=state.summary.key_quotes,
    )

    # Comparison
    comparison_resp = None
    if state.comparison:
        comp_metrics = [
            MetricComparisonResponse(
                metric_name=m.metric_name,
                current_value=m.current_value,
                previous_value=m.previous_value,
                change_amount=m.change_amount,
                change_percent=m.change_percent,
                trend=m.trend,
            )
            for m in state.comparison.comparisons
        ]
        comparison_resp = ComparisonResponse(
            current_quarter=state.comparison.current_quarter,
            previous_quarter=state.comparison.previous_quarter,
            comparisons=comp_metrics,
            executive_insights=state.comparison.executive_insights,
            overall_trend=state.comparison.overall_trend,
        )

    report_date = ""
    if state.final_report:
        report_date = state.final_report.report_date

    return FullReportResponse(
        company_name=state.company_name,
        quarter=state.quarter,
        report_date=report_date,
        kpis=kpi_resp,
        risk_analysis=risk_resp,
        summary=summary_resp,
        comparison=comparison_resp,
        errors=state.errors,
        warnings=state.warnings,
    )


# ============================================================
# Job status & results
# ============================================================

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of an analysis job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        result=job["result"] if job["status"] == "complete" else None,
        error=job["error"],
    )


@router.get("/results/{job_id}", response_model=FullReportResponse)
async def get_results(job_id: str):
    """Get the completed results of an analysis job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job['status']}. Current step: {job['current_step']}",
        )

    if job["result"] is None:
        raise HTTPException(status_code=500, detail="Job completed but result is missing")

    return job["result"]


@router.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        from app.utils.llm import list_available_models, get_active_model

        return {
            "active_model": get_active_model(),
            "available_models": list_available_models(),
        }
    except Exception as e:
        return {"error": str(e), "available_models": []}
