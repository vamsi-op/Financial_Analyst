"""
Text preprocessing, chunking, and section detection for financial documents.

Provides utilities to clean raw PDF text, split it into overlapping chunks
for RAG retrieval, detect standard financial report sections, and normalise
financial notation.
"""

import logging
import re
import unicodedata
from typing import Optional

from app.config import get_config

logger = logging.getLogger(__name__)


# ============================================================
# Section heading patterns (compiled once)
# ============================================================

# Canonical section names → list of regex patterns that match headings
_SECTION_PATTERNS: dict[str, list[re.Pattern]] = {
    "Income Statement": [
        re.compile(
            r"(?i)(?:consolidated\s+)?(?:statements?\s+of\s+)?(?:income|operations|earnings|profit\s+(?:and|&)\s+loss)",
        ),
    ],
    "Balance Sheet": [
        re.compile(
            r"(?i)(?:consolidated\s+)?(?:balance\s+sheets?|statements?\s+of\s+financial\s+(?:position|condition))",
        ),
    ],
    "Cash Flow": [
        re.compile(
            r"(?i)(?:consolidated\s+)?(?:statements?\s+of\s+)?cash\s+flows?",
        ),
    ],
    "MD&A": [
        re.compile(
            r"(?i)management['']?s?\s+discussion\s+(?:and|&)\s+analysis",
        ),
    ],
    "Revenue": [
        re.compile(r"(?i)(?:revenue|net\s+sales)\s+(?:breakdown|by\s+segment|discussion)"),
    ],
    "Risk Factors": [
        re.compile(r"(?i)risk\s+factors?"),
    ],
    "Notes to Financial Statements": [
        re.compile(r"(?i)notes\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements"),
    ],
    "Stockholders Equity": [
        re.compile(
            r"(?i)(?:consolidated\s+)?(?:statements?\s+of\s+)?(?:stockholders?|shareholders?)['']?\s+equity",
        ),
    ],
}

# Header / footer noise patterns
_HEADER_FOOTER_RE = re.compile(
    r"(?m)"
    r"(?:^[ \t]*page\s+\d+.*$)"              # "Page 3 of 10"
    r"|(?:^[ \t]*\d+[ \t]*$)"                 # lone page number
    r"|(?:^[ \t]*-\s*\d+\s*-[ \t]*$)"         # "- 3 -"
    r"|(?:^[ \t]*©.*$)"                       # copyright lines
    r"|(?:^[ \t]*(?:confidential|proprietary).*$)",  # confidential tags
    re.IGNORECASE,
)

# Excessive whitespace
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Company / quarter extraction helpers
_QUARTER_RE = re.compile(
    r"(?i)\b(Q[1-4]|(?:first|second|third|fourth)\s+quarter)"
    r"(?:\s+(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(20\d{2}))?"
)
_FISCAL_YEAR_RE = re.compile(
    r"(?i)\b(?:fiscal\s+(?:year\s+)?|fy\s*)(20\d{2})\b"
)
_YEAR_ENDED_RE = re.compile(
    r"(?i)(?:year|quarter|period)\s+ended?\s+(\w+\s+\d{1,2},?\s+20\d{2})"
)

# Financial normalisation patterns
_CURRENCY_SYM_MAP = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "INR": "₹",
}
_ABBREV_MAP = {
    r"\brev(?:enue)?\b": "Revenue",
    r"\bnet\s*inc(?:ome)?\b": "Net Income",
    r"\bop(?:erating)?\s*inc(?:ome)?\b": "Operating Income",
    r"\bEPS\b": "Earnings Per Share",
    r"\bYoY\b": "Year-over-Year",
    r"\bQoQ\b": "Quarter-over-Quarter",
    r"\bBps\b": "Basis Points",
}


# ============================================================
# Public API
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean raw extracted PDF text.

    Steps:
        1. Normalise Unicode (NFKC).
        2. Remove common header / footer noise.
        3. Collapse excessive blank lines and spaces.
        4. Strip leading / trailing whitespace on each line.

    Args:
        text: Raw text from PDF extraction.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove header / footer lines
    text = _HEADER_FOOTER_RE.sub("", text)

    # 3. Collapse whitespace
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 4. Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    cleaned = text.strip()
    logger.debug(
        "Cleaned text: %d → %d chars", len(text), len(cleaned)
    )
    return cleaned


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[str]:
    """
    Split text into overlapping chunks for vector-store ingestion.

    Tries to break on sentence boundaries to keep chunks semantically
    coherent.  Falls back to character splitting when sentences are longer
    than ``chunk_size``.

    Args:
        text:       Input text (should already be cleaned).
        chunk_size: Max characters per chunk.  Defaults to ``config.embedding.chunk_size``.
        overlap:    Character overlap between consecutive chunks.  Defaults
                    to ``config.embedding.chunk_overlap``.

    Returns:
        List of text chunks.
    """
    config = get_config()
    chunk_size = chunk_size if chunk_size is not None else config.embedding.chunk_size
    overlap = overlap if overlap is not None else config.embedding.chunk_overlap

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Sentence-aware splitting
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # Sentence longer than chunk_size → hard split
        if sentence_len > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            # Hard-split the long sentence
            for i in range(0, sentence_len, chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size])
            continue

        if current_length + sentence_len + 1 > chunk_size:
            # Emit the current chunk
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Overlap: keep trailing sentences that fit in overlap window
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s) + 1
            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk) + max(
                len(current_chunk) - 1, 0
            )

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(
        "Chunked text into %d chunks (size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def detect_sections(text: str) -> dict[str, str]:
    """
    Detect standard financial-report sections by scanning for headings.

    Returns a mapping of *canonical section name* → *section body text*.
    If a heading is found but the next heading is absent the body extends
    to the end of the document.

    Args:
        text: Full document text (preferably cleaned).

    Returns:
        Dict mapping section names to their body text.
    """
    if not text:
        return {}

    # Collect (position, section_name) tuples
    found: list[tuple[int, str]] = []
    for section_name, patterns in _SECTION_PATTERNS.items():
        for pat in patterns:
            match = pat.search(text)
            if match:
                found.append((match.start(), section_name))
                break  # first pattern match per section is enough

    if not found:
        logger.debug("No standard sections detected")
        return {}

    # Sort by position
    found.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}
    for i, (pos, name) in enumerate(found):
        end_pos = found[i + 1][0] if i + 1 < len(found) else len(text)
        body = text[pos:end_pos].strip()
        sections[name] = body

    logger.info("Detected %d sections: %s", len(sections), list(sections.keys()))
    return sections


def extract_company_info(text: str) -> dict[str, Optional[str]]:
    """
    Extract company name, quarter, and fiscal year from the report header.

    Heuristic: the company name is typically the first non-empty, non-date
    line in the document that is capitalised or title-cased.

    Args:
        text: Full document text.

    Returns:
        Dict with keys ``company_name``, ``quarter``, ``fiscal_year``.
    """
    info: dict[str, Optional[str]] = {
        "company_name": None,
        "quarter": None,
        "fiscal_year": None,
    }

    if not text:
        return info

    # --- Quarter ---
    qm = _QUARTER_RE.search(text[:2000])  # look only in first ~2 000 chars
    if qm:
        quarter_raw = qm.group(1).upper()
        # Normalise textual quarters
        _text_to_q = {
            "FIRST QUARTER": "Q1",
            "SECOND QUARTER": "Q2",
            "THIRD QUARTER": "Q3",
            "FOURTH QUARTER": "Q4",
        }
        quarter = _text_to_q.get(quarter_raw, quarter_raw)
        year = qm.group(2) if qm.lastindex and qm.lastindex >= 2 else None
        info["quarter"] = f"{quarter} {year}" if year else quarter

    # --- Fiscal year fallback ---
    if not info.get("quarter"):
        fy = _FISCAL_YEAR_RE.search(text[:2000])
        if fy:
            info["fiscal_year"] = fy.group(1)
    else:
        # Try to get fiscal year even if quarter was found
        fy = _FISCAL_YEAR_RE.search(text[:2000])
        if fy:
            info["fiscal_year"] = fy.group(1)

    # --- Year ended ---
    ye = _YEAR_ENDED_RE.search(text[:3000])
    if ye and not info.get("fiscal_year"):
        info["fiscal_year"] = ye.group(1)

    # --- Company name (heuristic) ---
    for line in text[:1500].splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        # Skip dates and numbers-only lines
        if re.match(r"^[\d/\-,.\s]+$", line):
            continue
        # Skip lines that look like section headings we know
        if any(p.match(line) for pats in _SECTION_PATTERNS.values() for p in pats):
            continue
        # Likely company name: first substantial text line
        if line[0].isupper() and len(line) < 100:
            info["company_name"] = line
            break

    logger.info("Extracted company info: %s", info)
    return info


def normalize_financial_text(text: str) -> str:
    """
    Normalise financial notation in text.

    * Spell out common abbreviations (EPS → Earnings Per Share).
    * Normalise currency codes to symbols (USD → $).
    * Standardise whitespace around numbers and symbols.

    Args:
        text: Input text.

    Returns:
        Normalised text.
    """
    if not text:
        return ""

    # Currency codes → symbols
    for code, symbol in _CURRENCY_SYM_MAP.items():
        text = re.sub(rf"\b{code}\s*", symbol, text)

    # Common abbreviations
    for pattern, replacement in _ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Normalise spaces around currency symbols
    text = re.sub(r"(\$|€|£|¥|₹)\s+", r"\1", text)

    # Normalise "million" / "billion" notation
    text = re.sub(r"(?i)\b(\d[\d,.]*)\s*mil(?:lion)?\b", r"\1 million", text)
    text = re.sub(r"(?i)\b(\d[\d,.]*)\s*bil(?:lion)?\b", r"\1 billion", text)
    text = re.sub(r"(?i)\b(\d[\d,.]*)\s*tril(?:lion)?\b", r"\1 trillion", text)

    return text.strip()


# ============================================================
# Internal helpers
# ============================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple rule-based approach.

    Handles common financial abbreviations (e.g., "Inc.", "Corp.", "vs.")
    to avoid false breaks.
    """
    # Protect common abbreviations from splitting
    _abbrevs = [
        "Inc", "Corp", "Ltd", "Co", "Mr", "Mrs", "Dr", "Jr", "Sr",
        "vs", "etc", "approx", "No", "Vol", "Fig", "eq",
    ]
    for abbr in _abbrevs:
        text = text.replace(f"{abbr}.", f"{abbr}∎")

    # Split on sentence-ending punctuation followed by whitespace + uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # Restore abbreviations
    sentences = [s.replace("∎", ".").strip() for s in parts if s.strip()]
    return sentences
