"""
Quarter-over-Quarter Comparison Agent.

Compares current-quarter KPIs against previous-quarter KPIs, computes
percentage changes, assigns trend arrows, and asks the LLM for a
narrative insight paragraph.
"""

import logging
from typing import Optional

from app.agents.base_agent import BaseAgent
from app.config import AppConfig
from app.graph.state import (
    KPIData,
    MetricComparison,
    PipelineState,
    QuarterComparison,
)
from app.utils.formatters import (
    format_change,
    parse_financial_number,
    trend_arrow,
)

logger = logging.getLogger(__name__)

# Minimum absolute percentage change to register a directional move.
_FLAT_THRESHOLD = 1.0


class ComparisonAgent(BaseAgent):
    """
    Compare KPIs between the current and previous quarter.

    The agent is **skipped** when ``previous_kpis`` is ``None`` on the
    pipeline state (i.e. this is the first report for the company).
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__(name="comparison_agent", config=config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Run the quarter comparison (or skip if no previous data).

        Args:
            state: Must contain ``kpis`` and optionally ``previous_kpis``.

        Returns:
            Updated state with ``comparison`` field (or ``None`` if skipped).
        """
        if state.previous_kpis is None:
            self.logger.info(
                "No previous-quarter KPIs available — skipping comparison"
            )
            state.warnings.append(
                "ComparisonAgent: skipped (no previous quarter data)"
            )
            return state

        self.logger.info("Quarter comparison started")

        try:
            comparison = self._compare_quarters(state.kpis, state.previous_kpis)

            # Fill quarter labels if available on state
            comparison.current_quarter = state.quarter or "Current"
            comparison.previous_quarter = (
                comparison.previous_quarter or "Previous"
            )

            # Generate narrative insights via LLM
            insights = self._generate_insights(comparison)
            comparison.executive_insights = insights

            # Determine overall trend
            comparison.overall_trend = self._determine_overall_trend(comparison)

            state.comparison = comparison
            self.logger.info(
                "Quarter comparison complete — %d metrics compared, trend=%s",
                len(comparison.comparisons),
                comparison.overall_trend,
            )

        except Exception as exc:
            self.logger.error(
                "Quarter comparison failed: %s", exc, exc_info=True
            )
            state.errors.append(f"ComparisonAgent error: {exc}")

        return state

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------
    def _compare_quarters(
        self, current: KPIData, previous: KPIData
    ) -> QuarterComparison:
        """
        Build per-metric comparisons for every KPI field.

        Only metrics where at least one quarter has data are included.

        Args:
            current: Current-quarter KPIs.
            previous: Previous-quarter KPIs.

        Returns:
            A ``QuarterComparison`` with a list of ``MetricComparison`` items.
        """
        fields = {
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

        comparisons: list[MetricComparison] = []

        for field_key, display_name in fields.items():
            cur_str = getattr(current, field_key, None)
            prev_str = getattr(previous, field_key, None)

            # Skip if neither quarter has data
            if cur_str is None and prev_str is None:
                continue

            mc = self._compare_metric(display_name, cur_str, prev_str)
            comparisons.append(mc)

        return QuarterComparison(comparisons=comparisons)

    def _compare_metric(
        self,
        name: str,
        current_val: Optional[str],
        previous_val: Optional[str],
    ) -> MetricComparison:
        """
        Compare a single metric between quarters.

        Args:
            name: Human-readable metric name.
            current_val: String value from the current quarter.
            previous_val: String value from the previous quarter.

        Returns:
            A ``MetricComparison`` with computed change and trend.
        """
        cur_num = parse_financial_number(current_val) if current_val else None
        prev_num = parse_financial_number(previous_val) if previous_val else None

        change_pct: Optional[float] = None
        change_amount: Optional[str] = None
        trend_indicator = "→"

        if cur_num is not None and prev_num is not None:
            pct, arrow = self._calculate_change(cur_num, prev_num)
            change_pct = pct
            trend_indicator = arrow

            diff = cur_num - prev_num
            change_amount = f"{diff:+,.2f}"

        return MetricComparison(
            metric_name=name,
            current_value=current_val,
            previous_value=previous_val,
            change_amount=change_amount,
            change_percent=change_pct,
            trend=trend_indicator,
        )

    @staticmethod
    def _calculate_change(
        current: float, previous: float
    ) -> tuple[float, str]:
        """
        Calculate percentage change and determine the trend arrow.

        Args:
            current: Current quarter numeric value.
            previous: Previous quarter numeric value.

        Returns:
            Tuple of ``(percentage_change, trend_arrow)``.
            Percentage change is ``None``-safe for zero denominators.
        """
        if previous == 0:
            if current == 0:
                return 0.0, "→"
            # Previous was zero, current isn't — infinite % change capped
            return 100.0 if current > 0 else -100.0, (
                "↑" if current > 0 else "↓"
            )

        pct = ((current - previous) / abs(previous)) * 100.0
        arrow = trend_arrow(pct, threshold=_FLAT_THRESHOLD)
        return round(pct, 2), arrow

    # ------------------------------------------------------------------
    # LLM narrative insights
    # ------------------------------------------------------------------
    def _generate_insights(self, comparison: QuarterComparison) -> str:
        """
        Ask the LLM for a brief narrative summarising the comparison.

        Args:
            comparison: The computed quarter comparison.

        Returns:
            A narrative paragraph, or a fallback message on failure.
        """
        if not comparison.comparisons:
            return "No comparable metrics available."

        # Build a compact table for the prompt
        rows: list[str] = []
        for mc in comparison.comparisons:
            chg = format_change(mc.change_percent) if mc.change_percent is not None else "N/A"
            rows.append(
                f"- {mc.metric_name}: {mc.previous_value or 'N/A'} → "
                f"{mc.current_value or 'N/A'} ({chg} {mc.trend})"
            )
        metrics_text = "\n".join(rows)

        system_prompt = (
            "You are a financial analyst. Given the quarter-over-quarter "
            "metric changes below, write a concise 3-5 sentence insight "
            "paragraph. Highlight the most significant changes, whether "
            "the overall trend is improving or declining, and any areas of "
            "concern. Return ONLY the paragraph text, no JSON."
        )

        prompt = (
            f"Quarter comparison "
            f"({comparison.previous_quarter} → {comparison.current_quarter}):\n\n"
            f"{metrics_text}"
        )

        try:
            insights = self._call_llm(prompt, system_prompt)
            return insights.strip()
        except RuntimeError as exc:
            self.logger.warning("Insight generation failed: %s", exc)
            return "Insight generation unavailable."

    # ------------------------------------------------------------------
    # Overall trend determination
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_overall_trend(comparison: QuarterComparison) -> str:
        """
        Determine the overall trend across all compared metrics.

        Uses a simple vote: count metrics trending up, down, or flat.

        Returns:
            One of ``"improving"``, ``"declining"``, or ``"stable"``.
        """
        up = 0
        down = 0
        flat = 0

        for mc in comparison.comparisons:
            if mc.trend == "↑":
                up += 1
            elif mc.trend == "↓":
                down += 1
            else:
                flat += 1

        if up > down and up > flat:
            return "improving"
        if down > up and down > flat:
            return "declining"
        return "stable"
