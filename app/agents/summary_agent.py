"""
Earnings Summary Agent — RAG-enhanced executive summary generation.

Uses retrieved text chunks (or a document excerpt) together with extracted
KPIs as context when prompting the LLM for a structured summary.
"""

import json
import logging
from typing import Optional

from app.agents.base_agent import BaseAgent
from app.config import AppConfig
from app.graph.state import EarningsSummary, KPIData, PipelineState

logger = logging.getLogger(__name__)

# Maximum characters of raw text to use when no RAG chunks are available.
_FALLBACK_EXCERPT_LENGTH = 4000


class SummaryAgent(BaseAgent):
    """
    Generate an executive earnings summary using the LLM with RAG context.

    If vector-store chunks are available on the pipeline state they are
    used as context; otherwise the first ~4 000 characters of the raw
    text are used as a fallback.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__(name="summary_agent", config=config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Generate the earnings summary and write it to the state.

        Args:
            state: Must contain ``raw_text`` (and ideally ``kpis``
                and ``chunks``).

        Returns:
            Updated state with populated ``summary`` field.
        """
        self.logger.info("Summary generation started")

        if not state.raw_text:
            self.logger.warning("No raw text — skipping summary generation")
            state.errors.append("SummaryAgent: no raw text available")
            return state

        try:
            summary = self._generate_summary(
                text=state.raw_text,
                kpis=state.kpis,
                chunks=state.chunks,
            )
            state.summary = summary
            self.logger.info(
                "Summary generation complete — executive summary length=%d",
                len(summary.executive_summary),
            )
        except Exception as exc:
            self.logger.error("Summary generation failed: %s", exc, exc_info=True)
            state.errors.append(f"SummaryAgent error: {exc}")

        return state

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def _generate_summary(
        self,
        text: str,
        kpis: KPIData,
        chunks: list[str],
    ) -> EarningsSummary:
        """
        Produce a structured earnings summary.

        Args:
            text: Full document text.
            kpis: Extracted KPI data for context enrichment.
            chunks: RAG-retrieved text chunks.

        Returns:
            Populated ``EarningsSummary``.
        """
        # Build context from RAG chunks or raw text excerpt
        if chunks:
            context = "\n\n---\n\n".join(chunks[: self.config.embedding.top_k])
            self.logger.debug(
                "Using %d RAG chunks as context", min(len(chunks), self.config.embedding.top_k)
            )
        else:
            context = text[:_FALLBACK_EXCERPT_LENGTH]
            self.logger.debug(
                "No RAG chunks — using first %d chars of raw text",
                _FALLBACK_EXCERPT_LENGTH,
            )

        prompt = self._build_summary_prompt(context, kpis)

        system_prompt = (
            "You are a senior financial analyst writing an executive summary "
            "of a company's quarterly earnings report. "
            "Your analysis should be data-driven, concise, and insightful. "
            "Return ONLY a JSON object matching the schema below — no "
            "explanations or markdown outside the JSON.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "executive_summary": "200-500 word summary",\n'
            '  "management_highlights": ["highlight 1", ...],\n'
            '  "growth_drivers": ["driver 1", ...],\n'
            '  "challenges": ["challenge 1", ...],\n'
            '  "outlook": "forward-looking guidance paragraph",\n'
            '  "key_quotes": ["notable quote 1", ...]\n'
            "}"
        )

        response = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json_response(response)

        return self._build_summary_model(parsed)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_summary_prompt(self, context: str, kpis: KPIData) -> str:
        """
        Build the user prompt containing context and KPIs.

        Args:
            context: Retrieved/excerpted text.
            kpis: Extracted KPIs.

        Returns:
            Formatted prompt string.
        """
        kpi_section = self._format_kpis_for_prompt(kpis)

        prompt = (
            "Based on the following earnings report excerpt and extracted KPIs, "
            "produce a comprehensive executive summary.\n\n"
            "--- EXTRACTED KPIs ---\n"
            f"{kpi_section}\n"
            "--- END KPIs ---\n\n"
            "--- REPORT EXCERPT ---\n"
            f"{context}\n"
            "--- END EXCERPT ---\n\n"
            "Instructions:\n"
            "1. Write an executive summary (200-500 words) covering key "
            "financial results, strategic highlights, and outlook.\n"
            "2. List 3-7 management highlights.\n"
            "3. List 2-5 growth drivers.\n"
            "4. List 2-5 challenges or risks.\n"
            "5. Provide a forward-looking outlook paragraph.\n"
            "6. Include 2-4 notable management quotes if present in the text.\n"
            "Return ONLY the JSON object."
        )
        return prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_kpis_for_prompt(kpis: KPIData) -> str:
        """
        Render extracted KPIs as a readable bullet list for the prompt.
        """
        lines: list[str] = []
        field_labels = {
            "revenue": "Revenue",
            "net_income": "Net Income",
            "operating_income": "Operating Income",
            "eps": "EPS",
            "cash_flow": "Operating Cash Flow",
            "free_cash_flow": "Free Cash Flow",
            "gross_margin": "Gross Margin",
            "operating_margin": "Operating Margin",
            "total_debt": "Total Debt",
            "total_assets": "Total Assets",
            "total_liabilities": "Total Liabilities",
        }
        for field, label in field_labels.items():
            value = getattr(kpis, field, None)
            if value:
                conf = kpis.confidence.get(field, 0)
                lines.append(f"- {label}: {value} (confidence: {conf:.0%})")

        return "\n".join(lines) if lines else "No KPIs extracted."

    @staticmethod
    def _build_summary_model(parsed: dict) -> EarningsSummary:
        """
        Construct an ``EarningsSummary`` from parsed LLM JSON output.

        Handles missing or malformed fields gracefully.
        """
        def _ensure_list(val) -> list[str]:
            if isinstance(val, list):
                return [str(v) for v in val]
            if isinstance(val, str):
                return [val]
            return []

        return EarningsSummary(
            executive_summary=str(parsed.get("executive_summary", "")),
            management_highlights=_ensure_list(
                parsed.get("management_highlights", [])
            ),
            growth_drivers=_ensure_list(parsed.get("growth_drivers", [])),
            challenges=_ensure_list(parsed.get("challenges", [])),
            outlook=str(parsed.get("outlook", "")),
            key_quotes=_ensure_list(parsed.get("key_quotes", [])),
        )
