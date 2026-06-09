"""
KPI Extraction Agent — hybrid deterministic + LLM approach.

Pipeline:
1. **Deterministic pass** – regex-based extraction from the raw text and
   tables via ``app.parsers.financial_parser``.  Confidence: **0.9**.
2. **LLM pass** – asks the LLM to fill in any metrics the regex could not
   find.  Confidence: **0.7**.
3. **Merge** – when both sources agree on a value the confidence is raised
   to **0.95**; otherwise the regex value is preferred.
"""

import logging
from typing import Any, Optional

from app.agents.base_agent import BaseAgent
from app.config import AppConfig
from app.graph.state import KPIData, PipelineState
from app.utils.formatters import parse_financial_number

logger = logging.getLogger(__name__)

# ---- Confidence tiers ----
CONFIDENCE_REGEX = 0.9
CONFIDENCE_LLM = 0.7
CONFIDENCE_BOTH_AGREE = 0.95


class KPIAgent(BaseAgent):
    """
    Extract financial KPIs from earnings report text.

    Uses a **hybrid** strategy:

    * First, a deterministic regex pass is attempted (fast, high
      confidence).
    * Then, the LLM is asked to extract only the metrics that the
      regex could *not* find, keeping cost and latency low.
    * Results are merged with regex values taking precedence when
      both sources produce a result; an agreement bumps confidence
      to 0.95.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__(name="kpi_agent", config=config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Run the full KPI-extraction pipeline.

        Args:
            state: Current pipeline state (must contain ``raw_text``).

        Returns:
            Updated state with populated ``kpis`` field.
        """
        self.logger.info("KPI extraction started")

        if not state.raw_text:
            self.logger.warning("No raw text available – skipping KPI extraction")
            state.errors.append("KPIAgent: no raw text available")
            return state

        try:
            # Step 1 – deterministic extraction
            regex_kpis = self._deterministic_extract(state.raw_text, state.tables)
            filled_regex = self._count_filled(regex_kpis)
            self.logger.info("Regex extraction found %d KPIs", filled_regex)

            # Step 2 – LLM extraction for gaps
            llm_kpis = self._llm_extract(state.raw_text, regex_kpis)
            filled_llm = self._count_filled(llm_kpis)
            self.logger.info("LLM extraction found %d additional KPIs", filled_llm)

            # Step 3 – merge
            merged = self._merge_results(regex_kpis, llm_kpis)
            merged.extraction_method = "hybrid"

            state.kpis = merged
            self.logger.info(
                "KPI extraction complete – %d metrics populated",
                self._count_filled(merged),
            )

        except Exception as exc:
            self.logger.error("KPI extraction failed: %s", exc, exc_info=True)
            state.errors.append(f"KPIAgent error: {exc}")

        return state

    # ------------------------------------------------------------------
    # Deterministic extraction
    # ------------------------------------------------------------------
    def _deterministic_extract(
        self, text: str, tables: list[dict[str, Any]]
    ) -> KPIData:
        """
        Use the financial parser's regex engine to extract KPIs.

        Falls back gracefully if the parser module is not yet implemented.

        Args:
            text: Full report text.
            tables: Extracted table dicts.

        Returns:
            ``KPIData`` populated with whatever the regex found.
        """
        kpis = KPIData(extraction_method="regex")

        try:
            from app.parsers.financial_parser import extract_kpis_from_text

            # extract_kpis_from_text returns a KPIData Pydantic model (not a dict)
            parsed: KPIData = extract_kpis_from_text(text, tables)

            _KPI_FIELDS = [
                "revenue", "net_income", "operating_income", "eps",
                "cash_flow", "gross_margin", "operating_margin",
                "total_debt", "total_assets", "total_liabilities",
                "free_cash_flow",
            ]

            for field_name in _KPI_FIELDS:
                # Use getattr since parsed is a KPIData object, not a dict
                value = getattr(parsed, field_name, None)
                if value is not None:
                    setattr(kpis, field_name, str(value))
                    kpis.confidence[field_name] = CONFIDENCE_REGEX

                    # Store numeric form if possible
                    numeric = parse_financial_number(str(value))
                    if numeric is not None:
                        kpis.raw_values[field_name] = numeric

            # Propagate raw_values and confidence from parser if available
            for k, v in parsed.raw_values.items():
                if k not in kpis.raw_values:
                    kpis.raw_values[k] = v

            self.logger.debug("Regex extraction populated: %s", list(kpis.confidence.keys()))

        except ImportError:
            self.logger.warning(
                "financial_parser not available – falling back to LLM-only extraction"
            )
        except Exception as exc:
            self.logger.warning("Deterministic extraction failed: %s", exc)

        return kpis

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------
    def _llm_extract(self, text: str, existing_kpis: KPIData) -> KPIData:
        """
        Ask the LLM to extract the KPI fields that regex did not find.

        The prompt explicitly lists which metrics are still missing so the
        LLM focuses its effort and the response stays small.

        Args:
            text: Full report text.
            existing_kpis: KPIs already found by the deterministic pass.

        Returns:
            ``KPIData`` containing only the LLM-sourced values.
        """
        kpis = KPIData(extraction_method="llm")

        # Build list of missing metrics
        all_fields = [
            "revenue", "net_income", "operating_income", "eps",
            "cash_flow", "gross_margin", "operating_margin",
            "total_debt", "total_assets", "total_liabilities",
            "free_cash_flow",
        ]
        missing = [f for f in all_fields if getattr(existing_kpis, f) is None]

        if not missing:
            self.logger.info("All KPIs found by regex – LLM pass skipped")
            return kpis

        # Truncate text to fit context window comfortably
        max_chars = min(len(text), 6000)
        excerpt = text[:max_chars]

        system_prompt = (
            "You are a financial data extraction specialist. "
            "Extract ONLY the requested metrics from the earnings report text. "
            "Return a JSON object with metric names as keys and their values "
            "as strings exactly as they appear in the report. "
            "If a metric is not found in the text, set its value to null. "
            "Do NOT invent values. Return ONLY valid JSON, no explanations."
        )

        prompt = (
            f"Extract the following financial metrics from the earnings report:\n\n"
            f"METRICS NEEDED: {', '.join(missing)}\n\n"
            f"--- REPORT TEXT ---\n{excerpt}\n--- END ---\n\n"
            f"Return JSON with the metric names as keys. Example:\n"
            f'{{"revenue": "$1.2B", "net_income": "$340M"}}'
        )

        try:
            response = self._call_llm(prompt, system_prompt)
            parsed = self._parse_json_response(response)

            for field_name in all_fields:
                value = parsed.get(field_name)
                if value and value != "null" and str(value).strip():
                    str_value = str(value).strip()
                    setattr(kpis, field_name, str_value)
                    kpis.confidence[field_name] = CONFIDENCE_LLM

                    numeric = parse_financial_number(str_value)
                    if numeric is not None:
                        kpis.raw_values[field_name] = numeric

        except RuntimeError as exc:
            self.logger.error("LLM extraction failed: %s", exc)
            state_warning = f"KPIAgent LLM fallback failed: {exc}"
            # Caller will see the warning via the returned (sparse) kpis

        return kpis

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_results(
        self, regex_kpis: KPIData, llm_kpis: KPIData
    ) -> KPIData:
        """
        Merge regex-sourced and LLM-sourced KPIs.

        Priority rules:
        - If only regex has the value → use it (confidence 0.9).
        - If only LLM has the value → use it (confidence 0.7).
        - If both have the value and agree → use regex value (confidence 0.95).
        - If both have the value but disagree → use regex value (confidence 0.9).

        Args:
            regex_kpis: Results from the deterministic pass.
            llm_kpis: Results from the LLM pass.

        Returns:
            Merged ``KPIData``.
        """
        merged = KPIData(extraction_method="hybrid")

        fields = [
            "revenue", "net_income", "operating_income", "eps",
            "cash_flow", "gross_margin", "operating_margin",
            "total_debt", "total_assets", "total_liabilities",
            "free_cash_flow",
        ]

        for field in fields:
            regex_val = getattr(regex_kpis, field)
            llm_val = getattr(llm_kpis, field)

            if regex_val and llm_val:
                # Both found — check agreement
                setattr(merged, field, regex_val)
                if self._values_agree(regex_val, llm_val):
                    merged.confidence[field] = CONFIDENCE_BOTH_AGREE
                    self.logger.debug("KPI '%s' — both agree: %s", field, regex_val)
                else:
                    merged.confidence[field] = CONFIDENCE_REGEX
                    self.logger.debug(
                        "KPI '%s' — disagreement (regex=%s, llm=%s), using regex",
                        field, regex_val, llm_val,
                    )
            elif regex_val:
                setattr(merged, field, regex_val)
                merged.confidence[field] = CONFIDENCE_REGEX
            elif llm_val:
                setattr(merged, field, llm_val)
                merged.confidence[field] = CONFIDENCE_LLM

            # Propagate raw numeric values
            if field in regex_kpis.raw_values:
                merged.raw_values[field] = regex_kpis.raw_values[field]
            elif field in llm_kpis.raw_values:
                merged.raw_values[field] = llm_kpis.raw_values[field]

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _values_agree(a: str, b: str) -> bool:
        """
        Determine whether two string-form financial values are equivalent.

        Compares the parsed numeric forms so that ``"$1.2B"`` and
        ``"$1,200M"`` are considered equal.
        """
        num_a = parse_financial_number(a)
        num_b = parse_financial_number(b)

        if num_a is not None and num_b is not None:
            # Allow 1 % tolerance for rounding differences
            if num_a == 0 and num_b == 0:
                return True
            avg = (abs(num_a) + abs(num_b)) / 2
            if avg == 0:
                return True
            return abs(num_a - num_b) / avg < 0.01

        # Fall back to exact string comparison
        return a.strip().lower() == b.strip().lower()

    @staticmethod
    def _count_filled(kpis: KPIData) -> int:
        """Count how many of the standard KPI fields are non-None."""
        fields = [
            "revenue", "net_income", "operating_income", "eps",
            "cash_flow", "gross_margin", "operating_margin",
            "total_debt", "total_assets", "total_liabilities",
            "free_cash_flow",
        ]
        return sum(1 for f in fields if getattr(kpis, f) is not None)
