"""
LangGraph workflow orchestration for the financial analysis pipeline.

Defines the complete StateGraph that coordinates PDF extraction,
text cleaning, vectorisation, parallel agent execution, comparison,
and final report generation.

Workflow::

    START → extract_pdf → clean_text → build_vectors
          → [parallel] extract_kpis, analyse_risks, generate_summary
          → compare_quarters → generate_report → END
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from app.config import get_config
from app.graph.state import (
    AnalysisReport,
    KPIData,
    PipelineState,
    RiskAnalysis,
)

logger = logging.getLogger(__name__)


# ============================================================
# Node functions
# ============================================================

def extract_pdf(state: PipelineState) -> dict[str, Any]:
    """
    Node 1: Extract text and tables from the uploaded PDF.

    Uses pdfplumber (primary) with PyMuPDF fallback, and PaddleOCR
    for scanned pages when needed.
    """
    logger.info("Node [extract_pdf]: Processing %s", state.pdf_path)

    if not state.pdf_path:
        return {"errors": state.errors + ["No PDF path provided"]}

    try:
        from app.parsers.pdf_parser import extract_from_pdf

        result = extract_from_pdf(state.pdf_path)
        pages = result.get("pages", [])
        full_text = result.get("full_text", "")
        all_tables = result.get("all_tables", [])

        # Check if any page needed OCR
        ocr_needed = any(p.get("ocr_used", False) for p in pages)

        if ocr_needed:
            logger.info("OCR was required for some pages — invoking OCR engine")
            try:
                from app.ocr.engine import OCREngine
                from app.config import get_config as _gc

                ocr_engine = OCREngine(_gc().ocr)
                if ocr_engine.is_available():
                    ocr_results = ocr_engine.process_pdf(state.pdf_path)
                    for ocr_page in ocr_results:
                        page_num = ocr_page["page_num"]
                        # Merge OCR text into pages that had insufficient text
                        for page in pages:
                            if page["page_num"] == page_num and len(page.get("text", "")) < 50:
                                page["text"] = ocr_page.get("text", "")
                                page["ocr_used"] = True

                    # Rebuild full text
                    full_text = "\n\n".join(
                        p.get("text", "") for p in pages if p.get("text")
                    )
                else:
                    logger.warning("PaddleOCR not available — skipping OCR")
            except Exception as e:
                logger.warning("OCR processing failed: %s", e)

        logger.info(
            "PDF extraction complete: %d pages, %d chars, %d tables",
            len(pages), len(full_text), len(all_tables),
        )

        return {
            "pages": pages,
            "raw_text": full_text,
            "tables": all_tables,
            "ocr_used": ocr_needed,
        }

    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc, exc_info=True)
        return {"errors": state.errors + [f"PDF extraction failed: {exc}"]}


def clean_text(state: PipelineState) -> dict[str, Any]:
    """
    Node 2: Clean and preprocess the extracted text.

    Normalises whitespace, removes boilerplate, detects sections,
    and extracts company metadata.
    """
    logger.info("Node [clean_text]: Cleaning %d chars of text", len(state.raw_text))

    if not state.raw_text:
        return {"warnings": state.warnings + ["No text to clean"]}

    try:
        from app.parsers.text_cleaner import (
            clean_text as _clean,
            chunk_text,
            detect_sections,
            extract_company_info,
        )

        config = get_config()

        cleaned = _clean(state.raw_text)

        # Extract company info
        company_info = extract_company_info(cleaned)
        company_name = company_info.get("company_name", state.company_name or "Unknown")
        quarter = company_info.get("quarter", state.quarter or "Unknown")

        # Chunk for RAG
        chunks = chunk_text(
            cleaned,
            chunk_size=config.embedding.chunk_size,
            overlap=config.embedding.chunk_overlap,
        )

        logger.info(
            "Text cleaning complete: %d chars → %d chunks, company=%s, quarter=%s",
            len(cleaned), len(chunks), company_name, quarter,
        )

        return {
            "raw_text": cleaned,
            "chunks": chunks,
            "company_name": company_name,
            "quarter": quarter,
        }

    except Exception as exc:
        logger.error("Text cleaning failed: %s", exc, exc_info=True)
        return {"warnings": state.warnings + [f"Text cleaning failed: {exc}"]}


def build_vectors(state: PipelineState) -> dict[str, Any]:
    """
    Node 3: Build FAISS vector index from text chunks.

    The index is stored in memory on the state for downstream
    retrieval by the summary agent.
    """
    logger.info("Node [build_vectors]: Indexing %d chunks", len(state.chunks))

    if not state.chunks:
        logger.warning("No chunks to vectorise")
        return {}

    try:
        from app.vectorstore.faiss_store import FAISSStore

        config = get_config()
        store = FAISSStore(config.embedding)
        store.add_documents(
            texts=state.chunks,
            metadata=[{"chunk_idx": i} for i in range(len(state.chunks))],
        )

        logger.info("Vector index built: %d vectors", len(store))
        # We cannot store the FAISS object on Pydantic state directly,
        # so we signal that vectors are ready via a flag on chunks.
        return {}

    except Exception as exc:
        logger.warning("Vector store creation failed: %s — RAG retrieval will be limited", exc)
        return {"warnings": state.warnings + [f"Vector store build failed: {exc}"]}


def extract_kpis(state: PipelineState) -> dict[str, Any]:
    """Node 4a: Run the KPI extraction agent."""
    logger.info("Node [extract_kpis]: Starting KPI extraction")

    try:
        from app.agents.kpi_agent import KPIAgent

        agent = KPIAgent()
        updated_state = agent.invoke(state)
        return {
            "kpis": updated_state.kpis,
            "errors": updated_state.errors,
        }
    except Exception as exc:
        logger.error("KPI extraction node failed: %s", exc, exc_info=True)
        return {"errors": state.errors + [f"KPI extraction failed: {exc}"]}


def analyse_risks(state: PipelineState) -> dict[str, Any]:
    """Node 4b: Run the risk analysis agent."""
    logger.info("Node [analyse_risks]: Starting risk analysis")

    try:
        from app.agents.risk_agent import RiskAgent

        agent = RiskAgent()
        updated_state = agent.invoke(state)
        return {
            "risk_analysis": updated_state.risk_analysis,
            "errors": updated_state.errors,
        }
    except Exception as exc:
        logger.error("Risk analysis node failed: %s", exc, exc_info=True)
        return {"errors": state.errors + [f"Risk analysis failed: {exc}"]}


def generate_summary(state: PipelineState) -> dict[str, Any]:
    """Node 4c: Run the earnings summary agent."""
    logger.info("Node [generate_summary]: Starting summary generation")

    try:
        from app.agents.summary_agent import SummaryAgent

        agent = SummaryAgent()
        updated_state = agent.invoke(state)
        return {
            "summary": updated_state.summary,
            "errors": updated_state.errors,
        }
    except Exception as exc:
        logger.error("Summary generation node failed: %s", exc, exc_info=True)
        return {"errors": state.errors + [f"Summary generation failed: {exc}"]}


def compare_quarters(state: PipelineState) -> dict[str, Any]:
    """
    Node 5: Run the quarter comparison agent.

    Only runs if ``previous_kpis`` is available on the state.
    """
    logger.info("Node [compare_quarters]: Starting quarter comparison")

    if state.previous_kpis is None:
        logger.info("No previous quarter data — skipping comparison")
        return {}

    try:
        from app.agents.comparison_agent import ComparisonAgent

        agent = ComparisonAgent()
        updated_state = agent.invoke(state)
        return {
            "comparison": updated_state.comparison,
            "errors": updated_state.errors,
        }
    except Exception as exc:
        logger.error("Quarter comparison failed: %s", exc, exc_info=True)
        return {"errors": state.errors + [f"Quarter comparison failed: {exc}"]}


def generate_report(state: PipelineState) -> dict[str, Any]:
    """
    Node 6: Assemble the final consolidated report from all agent outputs.
    """
    logger.info("Node [generate_report]: Assembling final report")

    try:
        from datetime import datetime

        report = AnalysisReport(
            company_name=state.company_name or "Unknown Company",
            quarter=state.quarter or "Unknown Quarter",
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            kpis=state.kpis,
            risk_analysis=state.risk_analysis,
            summary=state.summary,
            comparison=state.comparison,
        )

        logger.info("Final report assembled for %s %s", report.company_name, report.quarter)

        return {
            "final_report": report,
            "processing_complete": True,
        }

    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        return {
            "errors": state.errors + [f"Report generation failed: {exc}"],
            "processing_complete": True,
        }


# ============================================================
# Conditional edge: should we run comparison?
# ============================================================

def _should_compare(state: PipelineState) -> str:
    """Route to comparison if previous KPI data is available."""
    if state.previous_kpis is not None:
        return "compare_quarters"
    return "generate_report"


# ============================================================
# Graph builder
# ============================================================

def build_workflow() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()`` or ``.stream()``.

    Workflow::

        START
          │
          ▼
        extract_pdf
          │
          ▼
        clean_text
          │
          ▼
        build_vectors
          │
          ├──────────────────┬─────────────────┐
          ▼                  ▼                  ▼
        extract_kpis    analyse_risks    generate_summary
          │                  │                  │
          ├──────────────────┴─────────────────┘
          ▼
        [conditional] ─── has previous_kpis? ──┐
          │ yes                                  │ no
          ▼                                      │
        compare_quarters                         │
          │                                      │
          ├──────────────────────────────────────┘
          ▼
        generate_report
          │
          ▼
         END
    """
    builder = StateGraph(PipelineState)

    # --- Add nodes ---
    builder.add_node("extract_pdf", extract_pdf)
    builder.add_node("clean_text", clean_text)
    builder.add_node("build_vectors", build_vectors)
    builder.add_node("extract_kpis", extract_kpis)
    builder.add_node("analyse_risks", analyse_risks)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("compare_quarters", compare_quarters)
    builder.add_node("generate_report", generate_report)

    # --- Sequential preprocessing ---
    builder.add_edge(START, "extract_pdf")
    builder.add_edge("extract_pdf", "clean_text")
    builder.add_edge("clean_text", "build_vectors")

    # --- Fan-out: parallel agent execution ---
    builder.add_edge("build_vectors", "extract_kpis")
    builder.add_edge("build_vectors", "analyse_risks")
    builder.add_edge("build_vectors", "generate_summary")

    # --- Fan-in: all three agents converge ---
    # Conditional: go to comparison or skip to report
    builder.add_conditional_edges(
        "extract_kpis",
        _should_compare,
        {"compare_quarters": "compare_quarters", "generate_report": "generate_report"},
    )
    builder.add_edge("analyse_risks", "generate_report")
    builder.add_edge("generate_summary", "generate_report")

    # --- Comparison → report ---
    builder.add_edge("compare_quarters", "generate_report")

    # --- Report → END ---
    builder.add_edge("generate_report", END)

    logger.info("LangGraph workflow built successfully")
    return builder.compile()


# Singleton compiled workflow
_workflow = None


def get_workflow():
    """Get or create the compiled workflow singleton."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_analysis(
    pdf_path: str,
    company_name: str = "",
    quarter: str = "",
    previous_kpis: dict = None,
) -> PipelineState:
    """
    Run the full analysis pipeline on a PDF.

    This is the main entry point for programmatic use.

    Args:
        pdf_path: Path to the earnings report PDF.
        company_name: Optional company name override.
        quarter: Optional quarter identifier (e.g., "Q3 2025").
        previous_kpis: Optional previous quarter KPI dict for comparison.

    Returns:
        The final ``PipelineState`` with all agent outputs populated.
    """
    logger.info("Starting analysis pipeline for: %s", pdf_path)

    # Build initial state
    initial_state = PipelineState(
        pdf_path=pdf_path,
        company_name=company_name,
        quarter=quarter,
    )

    # Attach previous KPIs if provided
    if previous_kpis:
        if isinstance(previous_kpis, dict):
            initial_state.previous_kpis = KPIData(**previous_kpis)
        elif isinstance(previous_kpis, KPIData):
            initial_state.previous_kpis = previous_kpis

    # Run workflow
    workflow = get_workflow()
    final_state = workflow.invoke(initial_state)

    # Handle both dict and PipelineState return types
    if isinstance(final_state, dict):
        final_state = PipelineState(**final_state)

    logger.info("Analysis pipeline complete. Errors: %d", len(final_state.errors))
    return final_state
