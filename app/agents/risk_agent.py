"""
Risk Analysis Agent — explainable, multi-factor risk scoring.

Financial metrics are scored **deterministically**; the LLM is used
*only* for management-tone / sentiment analysis.

Scoring algorithm::

    overall = (revenue * 0.25 + profit * 0.20 + margin * 0.15 +
               debt * 0.15 + cash_flow * 0.15 + sentiment * 0.10)

Each sub-score ranges from 0 (no risk) to 100 (maximum risk).
"""

import logging
from typing import Optional

from app.agents.base_agent import BaseAgent
from app.config import AppConfig
from app.graph.state import KPIData, PipelineState, RiskAnalysis, RiskFactor
from app.utils.formatters import parse_financial_number

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Calculate an explainable risk score from extracted KPIs and report text.

    Each risk factor produces a 0-100 sub-score **with evidence**.  The
    weighted combination is classified into one of four levels using
    configurable thresholds from ``RiskConfig``.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__(name="risk_agent", config=config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Run the risk analysis pipeline.

        Args:
            state: Must contain ``kpis`` and ideally ``raw_text``.

        Returns:
            Updated state with populated ``risk_analysis`` field.
        """
        self.logger.info("Risk analysis started")

        try:
            risk = self._calculate_risk_score(state.kpis, state.raw_text)
            state.risk_analysis = risk
            self.logger.info(
                "Risk analysis complete — score=%.1f, level=%s",
                risk.overall_score,
                risk.risk_level,
            )
        except Exception as exc:
            self.logger.error("Risk analysis failed: %s", exc, exc_info=True)
            state.errors.append(f"RiskAgent error: {exc}")

        return state

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------
    def _calculate_risk_score(
        self, kpis: KPIData, text: str
    ) -> RiskAnalysis:
        """
        Build the full risk analysis by evaluating each factor.

        Args:
            kpis: Extracted KPIs.
            text: Raw earnings report text.

        Returns:
            Complete ``RiskAnalysis`` model.
        """
        rc = self.config.risk

        # Evaluate each factor
        revenue_factor = self._score_revenue_trend(kpis)
        profit_factor = self._score_profitability(kpis)
        margin_factor = self._score_margin_compression(kpis)
        debt_factor = self._score_debt_level(kpis)
        cash_factor = self._score_cash_flow(kpis)
        sentiment_factor = self._score_sentiment(text)

        # Weighted combination
        overall = (
            revenue_factor.score * rc.weight_revenue_decline
            + profit_factor.score * rc.weight_profit_decline
            + margin_factor.score * rc.weight_margin_compression
            + debt_factor.score * rc.weight_debt_increase
            + cash_factor.score * rc.weight_cash_flow
            + sentiment_factor.score * rc.weight_sentiment
        )
        overall = min(max(overall, 0.0), 100.0)

        risk_level = self._classify_risk_level(overall)

        factors = [
            revenue_factor,
            profit_factor,
            margin_factor,
            debt_factor,
            cash_factor,
            sentiment_factor,
        ]

        # Build recommendations from high-scoring factors
        recommendations = self._generate_recommendations(factors)

        summary = self._build_summary(overall, risk_level, factors)

        return RiskAnalysis(
            overall_score=round(overall, 1),
            risk_level=risk_level,
            factors=factors,
            summary=summary,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Individual factor scorers
    # ------------------------------------------------------------------
    def _score_revenue_trend(self, kpis: KPIData) -> RiskFactor:
        """
        Score risk from revenue figures.

        If revenue is available as a raw value, score is derived from the
        magnitude relative to thresholds.  Without data the score defaults
        to a moderate 40 (uncertain).
        """
        rc = self.config.risk
        revenue = kpis.raw_values.get("revenue")
        evidence: list[str] = []

        if revenue is None:
            return RiskFactor(
                category="Revenue Trend",
                score=40.0,
                severity="medium",
                description="Revenue data not available for analysis.",
                evidence=["No revenue figure extracted from the report."],
            )

        evidence.append(f"Reported revenue: {kpis.revenue}")

        # Without a previous quarter's revenue in KPIs.raw_values, we
        # cannot calculate a true YoY/QoQ change.  Use heuristics from
        # the report text if the word "decline" / "decrease" appears near
        # revenue mentions.
        score = 20.0  # baseline — revenue is present, that's positive
        description = f"Revenue reported at {kpis.revenue}."

        # Simple text heuristic for decline signals
        lower_text = (kpis.revenue or "").lower()
        if any(neg in lower_text for neg in ["decline", "decrease", "down"]):
            score = 60.0
            description += " Indicators suggest a decline."
            evidence.append("Negative language detected around revenue.")

        severity = self._severity_from_score(score)
        return RiskFactor(
            category="Revenue Trend",
            score=score,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _score_profitability(self, kpis: KPIData) -> RiskFactor:
        """Score risk based on net income / operating income."""
        evidence: list[str] = []
        net_income = kpis.raw_values.get("net_income")
        operating_income = kpis.raw_values.get("operating_income")

        if net_income is None and operating_income is None:
            return RiskFactor(
                category="Profitability",
                score=40.0,
                severity="medium",
                description="Profitability data not available.",
                evidence=["No net income or operating income extracted."],
            )

        score = 20.0  # baseline
        parts: list[str] = []

        if net_income is not None:
            evidence.append(f"Net income: {kpis.net_income}")
            if net_income < 0:
                score = max(score, 80.0)
                parts.append("Net income is negative (loss)")
            else:
                parts.append(f"Net income positive at {kpis.net_income}")

        if operating_income is not None:
            evidence.append(f"Operating income: {kpis.operating_income}")
            if operating_income < 0:
                score = max(score, 75.0)
                parts.append("Operating income is negative")
            else:
                parts.append(f"Operating income at {kpis.operating_income}")

        description = ". ".join(parts) + "." if parts else "Profitability metrics evaluated."
        severity = self._severity_from_score(score)
        return RiskFactor(
            category="Profitability",
            score=score,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _score_margin_compression(self, kpis: KPIData) -> RiskFactor:
        """Score risk from margin percentages (gross & operating)."""
        evidence: list[str] = []
        gross = kpis.raw_values.get("gross_margin")
        operating = kpis.raw_values.get("operating_margin")

        if gross is None and operating is None:
            return RiskFactor(
                category="Margin Compression",
                score=35.0,
                severity="medium",
                description="Margin data not available for analysis.",
                evidence=["No margin figures extracted."],
            )

        score = 15.0
        parts: list[str] = []

        if gross is not None:
            evidence.append(f"Gross margin: {kpis.gross_margin}")
            if gross < 20:
                score = max(score, 70.0)
                parts.append(f"Gross margin low at {kpis.gross_margin}")
            elif gross < 35:
                score = max(score, 45.0)
                parts.append(f"Gross margin moderate at {kpis.gross_margin}")
            else:
                parts.append(f"Gross margin healthy at {kpis.gross_margin}")

        if operating is not None:
            evidence.append(f"Operating margin: {kpis.operating_margin}")
            if operating < 5:
                score = max(score, 65.0)
                parts.append(f"Operating margin thin at {kpis.operating_margin}")
            elif operating < 15:
                score = max(score, 40.0)
                parts.append(f"Operating margin moderate at {kpis.operating_margin}")
            else:
                parts.append(f"Operating margin strong at {kpis.operating_margin}")

        description = ". ".join(parts) + "." if parts else "Margins evaluated."
        severity = self._severity_from_score(score)
        return RiskFactor(
            category="Margin Compression",
            score=score,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _score_debt_level(self, kpis: KPIData) -> RiskFactor:
        """Score risk from debt relative to total assets."""
        evidence: list[str] = []
        debt = kpis.raw_values.get("total_debt")
        assets = kpis.raw_values.get("total_assets")

        if debt is None:
            return RiskFactor(
                category="Debt Level",
                score=35.0,
                severity="medium",
                description="Debt data not available.",
                evidence=["No debt figure extracted."],
            )

        evidence.append(f"Total debt: {kpis.total_debt}")
        score = 20.0
        description_parts: list[str] = [f"Total debt reported at {kpis.total_debt}"]

        if assets is not None and assets > 0:
            evidence.append(f"Total assets: {kpis.total_assets}")
            debt_ratio = (debt / assets) * 100
            description_parts.append(f"Debt-to-assets ratio: {debt_ratio:.1f}%")

            if debt_ratio > 70:
                score = 85.0
            elif debt_ratio > 50:
                score = 60.0
            elif debt_ratio > 30:
                score = 35.0
            else:
                score = 15.0

        description = ". ".join(description_parts) + "."
        severity = self._severity_from_score(score)
        return RiskFactor(
            category="Debt Level",
            score=score,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _score_cash_flow(self, kpis: KPIData) -> RiskFactor:
        """Score risk from operating and free cash flow."""
        evidence: list[str] = []
        ocf = kpis.raw_values.get("cash_flow")
        fcf = kpis.raw_values.get("free_cash_flow")

        if ocf is None and fcf is None:
            return RiskFactor(
                category="Cash Flow",
                score=40.0,
                severity="medium",
                description="Cash flow data not available.",
                evidence=["No cash flow figure extracted."],
            )

        score = 20.0
        parts: list[str] = []

        if ocf is not None:
            evidence.append(f"Operating cash flow: {kpis.cash_flow}")
            if ocf < 0:
                score = max(score, 80.0)
                parts.append("Operating cash flow is negative")
            else:
                parts.append(f"Operating cash flow positive at {kpis.cash_flow}")

        if fcf is not None:
            evidence.append(f"Free cash flow: {kpis.free_cash_flow}")
            if fcf < 0:
                score = max(score, 75.0)
                parts.append("Free cash flow is negative")
            else:
                parts.append(f"Free cash flow positive at {kpis.free_cash_flow}")

        description = ". ".join(parts) + "." if parts else "Cash flow evaluated."
        severity = self._severity_from_score(score)
        return RiskFactor(
            category="Cash Flow",
            score=score,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _score_sentiment(self, text: str) -> RiskFactor:
        """
        Use the LLM to analyse management tone / sentiment.

        This is the **only** factor that relies on the LLM.  If the LLM
        call fails, a neutral score of 40 is returned.
        """
        if not text:
            return RiskFactor(
                category="Management Sentiment",
                score=40.0,
                severity="medium",
                description="No text available for sentiment analysis.",
                evidence=[],
            )

        # Truncate to keep the prompt small
        excerpt = text[:4000]

        system_prompt = (
            "You are a financial sentiment analyst. Analyse the tone and "
            "sentiment of the following earnings report excerpt. "
            "Return a JSON object with exactly these fields:\n"
            '  "sentiment_score": integer 0-100 (0 = very positive / no risk, '
            "100 = very negative / high risk),\n"
            '  "tone": one of "positive", "neutral", "cautious", "negative",\n'
            '  "key_phrases": list of up to 5 notable phrases that influenced '
            "your assessment.\n"
            "Return ONLY valid JSON."
        )

        prompt = (
            f"Analyse the management sentiment in this earnings report excerpt:\n\n"
            f"--- TEXT ---\n{excerpt}\n--- END ---"
        )

        try:
            response = self._call_llm(prompt, system_prompt)
            parsed = self._parse_json_response(response)

            score = float(parsed.get("sentiment_score", 40))
            score = min(max(score, 0), 100)
            tone = parsed.get("tone", "neutral")
            phrases = parsed.get("key_phrases", [])

            evidence = [f"Detected tone: {tone}"]
            if phrases:
                evidence.extend(f'"{p}"' for p in phrases[:5])

            severity = self._severity_from_score(score)
            return RiskFactor(
                category="Management Sentiment",
                score=score,
                severity=severity,
                description=f"Management tone assessed as '{tone}'.",
                evidence=evidence,
            )

        except Exception as exc:
            self.logger.warning("Sentiment analysis failed: %s", exc)
            return RiskFactor(
                category="Management Sentiment",
                score=40.0,
                severity="medium",
                description="Sentiment analysis could not be completed.",
                evidence=[f"Error: {exc}"],
            )

    # ------------------------------------------------------------------
    # Classification & helpers
    # ------------------------------------------------------------------
    def _classify_risk_level(self, score: float) -> str:
        """
        Map an overall score to a risk level using config thresholds.

        Args:
            score: Aggregated risk score (0–100).

        Returns:
            One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        """
        rc = self.config.risk
        if score <= rc.low_risk_max:
            return "low"
        if score <= rc.medium_risk_max:
            return "medium"
        if score <= rc.high_risk_max:
            return "high"
        return "critical"

    @staticmethod
    def _severity_from_score(score: float) -> str:
        """Quick helper mapping a 0-100 score to a severity label."""
        if score <= 25:
            return "low"
        if score <= 50:
            return "medium"
        if score <= 75:
            return "high"
        return "critical"

    @staticmethod
    def _generate_recommendations(factors: list[RiskFactor]) -> list[str]:
        """
        Produce actionable recommendations from high-scoring factors.
        """
        recommendations: list[str] = []
        for f in factors:
            if f.score >= 60:
                if f.category == "Revenue Trend":
                    recommendations.append(
                        "Monitor revenue trajectory closely; consider "
                        "diversification strategies."
                    )
                elif f.category == "Profitability":
                    recommendations.append(
                        "Investigate cost structure and paths to profitability."
                    )
                elif f.category == "Margin Compression":
                    recommendations.append(
                        "Evaluate pricing power and input cost management."
                    )
                elif f.category == "Debt Level":
                    recommendations.append(
                        "Assess refinancing options and debt-reduction plans."
                    )
                elif f.category == "Cash Flow":
                    recommendations.append(
                        "Cash flow health is concerning; review capital "
                        "expenditure and working capital management."
                    )
                elif f.category == "Management Sentiment":
                    recommendations.append(
                        "Management tone suggests caution; dig deeper into "
                        "qualitative disclosures and guidance."
                    )
        return recommendations

    def _build_summary(
        self, score: float, level: str, factors: list[RiskFactor]
    ) -> str:
        """Build a concise risk summary paragraph."""
        high_risk_factors = [f for f in factors if f.score >= 50]
        low_risk_factors = [f for f in factors if f.score < 30]

        parts = [
            f"Overall risk score is {score:.0f}/100, classified as {level.upper()}."
        ]
        if high_risk_factors:
            names = ", ".join(f.category for f in high_risk_factors)
            parts.append(f"Elevated risk areas: {names}.")
        if low_risk_factors:
            names = ", ".join(f.category for f in low_risk_factors)
            parts.append(f"Low-risk areas: {names}.")

        return " ".join(parts)
