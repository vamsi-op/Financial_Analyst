"""
PDF text and table extraction pipeline.

Uses pdfplumber as the primary extraction engine (superior table handling)
and falls back to PyMuPDF (fitz) when pdfplumber fails. Detects pages that
need OCR based on a configurable minimum text-length threshold.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_config

logger = logging.getLogger(__name__)


# ============================================================
# Public API
# ============================================================

def extract_from_pdf(pdf_path: str) -> dict[str, Any]:
    """
    Main entry point for PDF extraction.

    Tries pdfplumber first; falls back to PyMuPDF on failure.
    Each page result is tagged with a flag indicating whether OCR
    is recommended (see ``_needs_ocr``).

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A dict with keys:
            - ``pages``     – list of per-page dicts (page_num, text, tables, needs_ocr)
            - ``full_text`` – concatenated text of all pages
            - ``all_tables``– flat list of every table found across pages
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Allow plain-text files (e.g., synthetic sample reports for testing)
    if path.suffix.lower() in (".txt", ".text", ".md"):
        logger.info("Reading plain-text file directly: %s", pdf_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        page = {"page_num": 1, "text": text.strip(), "tables": [], "needs_ocr": False}
        return {"pages": [page], "full_text": text.strip(), "all_tables": []}

    if not path.suffix.lower() == ".pdf":
        logger.error("File is not a PDF: %s", pdf_path)
        raise ValueError(f"File is not a PDF: {pdf_path}")

    logger.info("Starting PDF extraction for: %s", pdf_path)

    # Primary: pdfplumber (better tables)
    try:
        pages = _extract_with_pdfplumber(pdf_path)
        logger.info("pdfplumber extraction succeeded (%d pages)", len(pages))
    except Exception as exc:
        logger.warning(
            "pdfplumber extraction failed (%s); falling back to PyMuPDF", exc
        )
        try:
            pages = _extract_with_pymupdf(pdf_path)
            logger.info("PyMuPDF extraction succeeded (%d pages)", len(pages))
        except Exception as fallback_exc:
            logger.error("Both extraction engines failed: %s", fallback_exc)
            raise RuntimeError(
                f"Could not extract text from {pdf_path}. "
                f"pdfplumber error: {exc} | PyMuPDF error: {fallback_exc}"
            ) from fallback_exc

    # Mark pages that might need OCR
    for page in pages:
        page["needs_ocr"] = _needs_ocr(page.get("text", ""))

    full_text = "\n\n".join(p.get("text", "") for p in pages)
    all_tables: list[list[list[str]]] = []
    for p in pages:
        all_tables.extend(p.get("tables", []))

    result = {
        "pages": pages,
        "full_text": full_text,
        "all_tables": all_tables,
    }
    logger.info(
        "Extraction complete – %d pages, %d tables, %d chars of text",
        len(pages),
        len(all_tables),
        len(full_text),
    )
    return result


def extract_tables_as_dataframes(pdf_path: str) -> list[pd.DataFrame]:
    """
    Extract every table in the PDF as a pandas DataFrame.

    The first row of each table is promoted to column headers when it looks
    like a header row (non-numeric strings).

    Args:
        pdf_path: Path to the PDF.

    Returns:
        List of DataFrames, one per table found.
    """
    extraction = extract_from_pdf(pdf_path)
    dataframes: list[pd.DataFrame] = []

    for table in extraction["all_tables"]:
        if not table or len(table) < 2:
            continue
        try:
            header = table[0]
            data = table[1:]
            # Promote first row to header if it looks textual
            if header and all(
                isinstance(c, str) and not _is_numeric(c) for c in header if c
            ):
                df = pd.DataFrame(data, columns=header)
            else:
                df = pd.DataFrame(table)
            dataframes.append(df)
        except Exception as exc:
            logger.warning("Failed to convert table to DataFrame: %s", exc)

    logger.info("Converted %d tables to DataFrames", len(dataframes))
    return dataframes


# ============================================================
# Extraction engines
# ============================================================

def _extract_with_pdfplumber(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extract text and tables per page using **pdfplumber**.

    pdfplumber excels at structured-table extraction from native-text PDFs.
    It returns each table as a list-of-lists (rows × cols).

    Raises:
        ImportError: If pdfplumber is not installed.
        Exception:   On encrypted/corrupted PDFs.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for primary PDF extraction. "
            "Install it with: pip install pdfplumber"
        ) from exc

    pages: list[dict[str, Any]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.debug("Opened PDF with pdfplumber – %d pages", len(pdf.pages))
            for idx, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    tables = page.extract_tables() or []
                    # Clean None values inside table cells
                    clean_tables = _clean_table_cells(tables)
                    pages.append({
                        "page_num": idx,
                        "text": text.strip(),
                        "tables": clean_tables,
                    })
                except Exception as page_exc:
                    logger.warning(
                        "pdfplumber failed on page %d: %s", idx, page_exc
                    )
                    pages.append({
                        "page_num": idx,
                        "text": "",
                        "tables": [],
                    })
    except Exception as exc:
        # Check for encrypted PDF hint
        error_msg = str(exc).lower()
        if "encrypt" in error_msg or "password" in error_msg:
            logger.error("PDF appears to be encrypted: %s", pdf_path)
            raise RuntimeError(
                f"PDF is encrypted and cannot be processed: {pdf_path}"
            ) from exc
        raise

    return pages


def _extract_with_pymupdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Fallback extraction using **PyMuPDF** (``fitz``).

    PyMuPDF is faster and more tolerant of malformed PDFs but does not
    provide structured-table extraction out of the box.

    Raises:
        ImportError: If PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required as a fallback PDF engine. "
            "Install it with: pip install PyMuPDF"
        ) from exc

    pages: list[dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        error_msg = str(exc).lower()
        if "encrypt" in error_msg or "password" in error_msg:
            logger.error("PDF appears to be encrypted: %s", pdf_path)
            raise RuntimeError(
                f"PDF is encrypted and cannot be processed: {pdf_path}"
            ) from exc
        raise

    try:
        logger.debug("Opened PDF with PyMuPDF – %d pages", doc.page_count)
        for idx in range(doc.page_count):
            try:
                page = doc.load_page(idx)
                text = page.get_text("text") or ""
                pages.append({
                    "page_num": idx + 1,
                    "text": text.strip(),
                    "tables": [],  # PyMuPDF doesn't do structured tables
                })
            except Exception as page_exc:
                logger.warning("PyMuPDF failed on page %d: %s", idx + 1, page_exc)
                pages.append({
                    "page_num": idx + 1,
                    "text": "",
                    "tables": [],
                })
    finally:
        doc.close()

    return pages


# ============================================================
# Helpers
# ============================================================

def _needs_ocr(page_text: str) -> bool:
    """
    Decide whether a page likely needs OCR.

    A page is flagged for OCR if its extracted text is shorter than the
    ``min_text_length`` threshold defined in :pyclass:`OCRConfig`.

    Args:
        page_text: The text extracted by pdfplumber / PyMuPDF.

    Returns:
        ``True`` if the page text is too short and OCR should be applied.
    """
    config = get_config()
    threshold = config.ocr.min_text_length
    stripped = page_text.strip()
    needs = len(stripped) < threshold
    if needs:
        logger.debug(
            "Page text length (%d) < threshold (%d) — OCR recommended",
            len(stripped),
            threshold,
        )
    return needs


def _clean_table_cells(tables: list[list[list]]) -> list[list[list[str]]]:
    """
    Replace ``None`` cells with empty strings and strip whitespace.

    Args:
        tables: Raw tables from pdfplumber (list of tables, each a
                list of rows, each a list of cell values).

    Returns:
        Cleaned tables with all cells as strings.
    """
    cleaned: list[list[list[str]]] = []
    for table in tables:
        clean_table: list[list[str]] = []
        for row in table:
            clean_row = [
                str(cell).strip() if cell is not None else "" for cell in row
            ]
            clean_table.append(clean_row)
        cleaned.append(clean_table)
    return cleaned


def _is_numeric(value: str) -> bool:
    """Return ``True`` if *value* looks like a number (including currency)."""
    cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False
