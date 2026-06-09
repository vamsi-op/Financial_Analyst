"""
Tests for the PDF parsing and financial extraction modules.
"""

import pytest
from unittest.mock import patch, MagicMock


# ============================================================
# Financial number parsing tests
# ============================================================

class TestParseFinancialNumber:
    """Test the deterministic financial number parser."""

    def test_basic_dollar_amount(self):
        from app.parsers.financial_parser import _parse_financial_number

        assert _parse_financial_number("$1,234.56") == pytest.approx(1234.56)

    def test_billions(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("$1.2B")
        assert result == pytest.approx(1.2e9)

    def test_billions_word(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("$1.2 billion")
        assert result == pytest.approx(1.2e9)

    def test_millions(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("$1,234M")
        assert result == pytest.approx(1234e6)

    def test_millions_word(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("$1,234 million")
        assert result == pytest.approx(1234e6)

    def test_negative_parentheses(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("(1,234)")
        assert result == pytest.approx(-1234)

    def test_negative_dollar_parentheses(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("($1,234)")
        assert result == pytest.approx(-1234)

    def test_negative_dash(self):
        from app.parsers.financial_parser import _parse_financial_number

        result = _parse_financial_number("-$1,234.56")
        assert result == pytest.approx(-1234.56)

    def test_plain_number(self):
        from app.parsers.financial_parser import _parse_financial_number

        assert _parse_financial_number("42.5") == pytest.approx(42.5)

    def test_none_for_empty(self):
        from app.parsers.financial_parser import _parse_financial_number

        assert _parse_financial_number("") is None
        assert _parse_financial_number(None) is None

    def test_none_for_text(self):
        from app.parsers.financial_parser import _parse_financial_number

        assert _parse_financial_number("no numbers here") is None


# ============================================================
# KPI extraction from text tests
# ============================================================

class TestKPIExtraction:
    """Test regex-based KPI extraction from financial text."""

    SAMPLE_TEXT = """
    Apple Inc. Fiscal Year 2025 Q3 Results

    Revenue was $85.5 billion for the quarter, compared to $81.8 billion
    in the year-ago quarter.

    Net income of $21.4 billion, or $1.40 per diluted share.

    Operating income was $26.0 billion.

    The company generated operating cash flow of $28.6 billion.

    Gross margin was 46.3%.

    Total debt was $98.3 billion. Total assets of $352.6 billion.
    Total liabilities was $274.8 billion.
    """

    def test_extract_revenue(self):
        from app.parsers.financial_parser import _extract_revenue

        formatted, value = _extract_revenue(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(85.5e9, rel=0.01)

    def test_extract_net_income(self):
        from app.parsers.financial_parser import _extract_net_income

        formatted, value = _extract_net_income(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(21.4e9, rel=0.01)

    def test_extract_eps(self):
        from app.parsers.financial_parser import _extract_eps

        formatted, value = _extract_eps(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(1.40, rel=0.01)

    def test_extract_operating_income(self):
        from app.parsers.financial_parser import _extract_operating_income

        formatted, value = _extract_operating_income(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(26.0e9, rel=0.01)

    def test_extract_cash_flow(self):
        from app.parsers.financial_parser import _extract_cash_flow

        formatted, value = _extract_cash_flow(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(28.6e9, rel=0.01)

    def test_extract_debt(self):
        from app.parsers.financial_parser import _extract_debt

        formatted, value = _extract_debt(self.SAMPLE_TEXT)
        assert value is not None
        assert value == pytest.approx(98.3e9, rel=0.01)

    def test_full_extraction(self):
        from app.parsers.financial_parser import extract_kpis_from_text

        kpis = extract_kpis_from_text(self.SAMPLE_TEXT)
        assert kpis.revenue is not None
        assert kpis.net_income is not None
        assert kpis.eps is not None
        assert kpis.gross_margin is not None
        assert kpis.gross_margin == "46.3%"


# ============================================================
# Text cleaner tests
# ============================================================

class TestTextCleaner:
    """Test text cleaning and chunking utilities."""

    def test_clean_removes_page_numbers(self):
        from app.parsers.text_cleaner import clean_text

        text = "Some content\n\nPage 1 of 10\n\nMore content\n\nPage 2 of 10"
        cleaned = clean_text(text)
        assert "Page 1 of 10" not in cleaned

    def test_clean_normalizes_whitespace(self):
        from app.parsers.text_cleaner import clean_text

        text = "Hello   \n\n\n\n   World"
        cleaned = clean_text(text)
        assert "    " not in cleaned

    def test_chunk_text_creates_chunks(self):
        from app.parsers.text_cleaner import chunk_text

        text = "A" * 1500  # 1500 chars
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert len(chunks) >= 2
        assert all(len(c) <= 600 for c in chunks)  # allow some margin

    def test_chunk_text_empty(self):
        from app.parsers.text_cleaner import chunk_text

        chunks = chunk_text("", chunk_size=512, overlap=50)
        assert chunks == []

    def test_extract_company_info(self):
        from app.parsers.text_cleaner import extract_company_info

        text = "Apple Inc.\nQ3 2025 Earnings Report\nFiscal Year 2025"
        info = extract_company_info(text)
        assert info is not None
        assert isinstance(info, dict)

    def test_detect_sections(self):
        from app.parsers.text_cleaner import detect_sections

        text = """
        INCOME STATEMENT
        Revenue: $100M

        BALANCE SHEET
        Total Assets: $500M

        CASH FLOW STATEMENT
        Operating Cash Flow: $50M
        """
        sections = detect_sections(text)
        assert isinstance(sections, dict)


# ============================================================
# Formatters tests
# ============================================================

class TestFormatters:
    """Test output formatting utilities."""

    def test_format_currency_billions(self):
        from app.utils.formatters import format_currency

        result = format_currency(1_500_000_000)
        assert "1.50" in result
        assert "B" in result

    def test_format_currency_millions(self):
        from app.utils.formatters import format_currency

        result = format_currency(250_000_000)
        assert "250" in result
        assert "M" in result

    def test_format_percentage(self):
        from app.utils.formatters import format_percentage

        assert format_percentage(12.345) == "12.3%"

    def test_format_change_positive(self):
        from app.utils.formatters import format_change

        result = format_change(5.0)
        assert "+" in result
        assert "5.0" in result

    def test_format_change_negative(self):
        from app.utils.formatters import format_change

        result = format_change(-3.2)
        assert "-" in result
        assert "3.2" in result

    def test_trend_arrow_up(self):
        from app.utils.formatters import trend_arrow

        assert trend_arrow(5.0) == "↑"

    def test_trend_arrow_down(self):
        from app.utils.formatters import trend_arrow

        assert trend_arrow(-5.0) == "↓"

    def test_trend_arrow_flat(self):
        from app.utils.formatters import trend_arrow

        assert trend_arrow(0.5) == "→"

    def test_parse_financial_number(self):
        from app.utils.formatters import parse_financial_number

        assert parse_financial_number("$1.2B") == pytest.approx(1.2e9)
        assert parse_financial_number("$500M") == pytest.approx(500e6)

    def test_risk_level_color(self):
        from app.utils.formatters import risk_level_color

        assert risk_level_color("low").startswith("#")
        assert risk_level_color("critical").startswith("#")
