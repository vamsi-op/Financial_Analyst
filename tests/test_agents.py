"""
Tests for the financial analysis agents.

Uses mock LLM responses to test agent logic without requiring Ollama.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.graph.state import (
    KPIData,
    PipelineState,
    RiskAnalysis,
    RiskFactor,
    EarningsSummary,
    QuarterComparison,
    MetricComparison,
)


# ============================================================
# KPI Agent tests
# ============================================================

class TestKPIAgent:
    """Test the KPI extraction agent."""

    def _make_state(self, text: str = "", tables: list = None) -> PipelineState:
        return PipelineState(
            raw_text=text,
            tables=tables or [],
            pdf_path="test.pdf",
        )

    def test_invoke_with_empty_text(self):
        from app.agents.kpi_agent import KPIAgent

        agent = KPIAgent()
        state = self._make_state(text="")
        result = agent.invoke(state)

        assert "no raw text" in result.errors[0].lower() or len(result.errors) > 0

    @patch("app.agents.base_agent.invoke_with_fallback")
    def test_invoke_with_financial_text(self, mock_llm):
        """Test KPI extraction from realistic financial text."""
        mock_llm.return_value = '{"free_cash_flow": "$25.5B"}'

        from app.agents.kpi_agent import KPIAgent

        text = """
        Revenue was $85.5 billion for the quarter.
        Net income of $21.4 billion, or $1.40 per diluted share.
        Operating income was $26.0 billion.
        Operating cash flow of $28.6 billion.
        Gross margin was 46.3%.
        """

        agent = KPIAgent()
        state = self._make_state(text=text)
        result = agent.invoke(state)

        # Should have extracted some metrics via regex
        assert result.kpis.revenue is not None
        assert result.kpis.eps is not None

    def test_merge_prefers_regex(self):
        from app.agents.kpi_agent import KPIAgent

        agent = KPIAgent()

        # Use values >1% apart so _values_agree() returns False (disagreement path)
        # $85.5B vs $92.0B differ by ~7.6% — clearly disagrees
        regex_kpis = KPIData(revenue="$85.5 billion", confidence={"revenue": 0.9})
        llm_kpis = KPIData(revenue="$92.0 billion", confidence={"revenue": 0.7})

        merged = agent._merge_results(regex_kpis, llm_kpis)
        assert merged.revenue == "$85.5 billion"  # regex preferred
        assert merged.confidence["revenue"] == 0.9  # disagreement → regex conf

    def test_values_agree(self):
        from app.agents.kpi_agent import KPIAgent

        # Same value, different format
        assert KPIAgent._values_agree("$1.2B", "$1,200M") is True
        assert KPIAgent._values_agree("$85.5 billion", "$85.5B") is True

    def test_values_disagree(self):
        from app.agents.kpi_agent import KPIAgent

        assert KPIAgent._values_agree("$85.5B", "$90.0B") is False


# ============================================================
# Risk Agent tests
# ============================================================

class TestRiskAgent:
    """Test the risk analysis agent."""

    def test_risk_level_classification(self):
        from app.agents.risk_agent import RiskAgent

        agent = RiskAgent()
        assert agent._classify_risk_level(15) == "low"
        assert agent._classify_risk_level(45) == "medium"
        assert agent._classify_risk_level(70) == "high"
        assert agent._classify_risk_level(90) == "critical"

    @patch("app.agents.base_agent.invoke_with_fallback")
    def test_invoke_with_kpis(self, mock_llm):
        """Test risk analysis with pre-populated KPIs."""
        mock_llm.return_value = '{"sentiment_score": 30, "concerns": ["Macro uncertainty"], "positive_signals": ["Strong demand"]}'

        from app.agents.risk_agent import RiskAgent

        state = PipelineState(
            raw_text="Revenue declined 10% year over year. Debt increased significantly.",
            kpis=KPIData(
                revenue="$80B",
                net_income="$15B",
                cash_flow="$20B",
                total_debt="$120B",
                total_assets="$300B",
                raw_values={
                    "revenue": 80e9,
                    "net_income": 15e9,
                    "cash_flow": 20e9,
                    "total_debt": 120e9,
                    "total_assets": 300e9,
                },
            ),
        )

        agent = RiskAgent()
        result = agent.invoke(state)

        assert result.risk_analysis.overall_score >= 0
        assert result.risk_analysis.overall_score <= 100
        assert result.risk_analysis.risk_level in ("low", "medium", "high", "critical")


# ============================================================
# Summary Agent tests
# ============================================================

class TestSummaryAgent:
    """Test the earnings summary agent."""

    @patch("app.agents.base_agent.invoke_with_fallback")
    def test_invoke_generates_summary(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "executive_summary": "The company reported strong results...",
            "management_highlights": ["Revenue growth of 15%"],
            "growth_drivers": ["Cloud services expansion"],
            "challenges": ["Supply chain constraints"],
            "outlook": "Management expects continued growth.",
            "key_quotes": ["CEO: We are pleased with the results."],
        })

        from app.agents.summary_agent import SummaryAgent

        state = PipelineState(
            raw_text="Apple reported revenue of $85.5 billion..." * 10,
            kpis=KPIData(revenue="$85.5B"),
            chunks=["Apple reported revenue of $85.5 billion for the quarter."],
        )

        agent = SummaryAgent()
        result = agent.invoke(state)

        assert result.summary.executive_summary != ""


# ============================================================
# Comparison Agent tests
# ============================================================

class TestComparisonAgent:
    """Test the quarter comparison agent."""

    def test_skip_without_previous_kpis(self):
        from app.agents.comparison_agent import ComparisonAgent

        state = PipelineState(
            raw_text="Some text",
            kpis=KPIData(revenue="$85.5B"),
            previous_kpis=None,
        )

        agent = ComparisonAgent()
        result = agent.invoke(state)

        assert result.comparison is None

    @patch("app.agents.base_agent.invoke_with_fallback")
    def test_compare_with_previous_kpis(self, mock_llm):
        mock_llm.return_value = "Revenue showed solid growth of 4.5% QoQ."

        from app.agents.comparison_agent import ComparisonAgent

        state = PipelineState(
            raw_text="Quarterly results",
            kpis=KPIData(
                revenue="$85.5B",
                net_income="$21.4B",
                raw_values={"revenue": 85.5e9, "net_income": 21.4e9},
            ),
            previous_kpis=KPIData(
                revenue="$81.8B",
                net_income="$20.7B",
                raw_values={"revenue": 81.8e9, "net_income": 20.7e9},
            ),
            quarter="Q3 2025",
        )

        agent = ComparisonAgent()
        result = agent.invoke(state)

        assert result.comparison is not None
        assert len(result.comparison.comparisons) > 0

    def test_calculate_change(self):
        from app.agents.comparison_agent import ComparisonAgent

        agent = ComparisonAgent()
        pct, arrow = agent._calculate_change(110, 100)
        assert pct == pytest.approx(10.0, rel=0.01)
        assert arrow == "↑"

    def test_calculate_change_decrease(self):
        from app.agents.comparison_agent import ComparisonAgent

        agent = ComparisonAgent()
        pct, arrow = agent._calculate_change(90, 100)
        assert pct == pytest.approx(-10.0, rel=0.01)
        assert arrow == "↓"

    def test_calculate_change_flat(self):
        from app.agents.comparison_agent import ComparisonAgent

        agent = ComparisonAgent()
        pct, arrow = agent._calculate_change(100.5, 100)
        assert arrow == "→"


# Required import for JSON
import json
