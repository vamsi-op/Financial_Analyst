"""
PaddleOCR engine for scanned and image-heavy PDF pages.

Provides an ``OCREngine`` class that:
* Converts PDF pages to PIL Images via PyMuPDF.
* Runs PaddleOCR on each image and returns structured results.
* Degrades gracefully when PaddleOCR is not installed.
"""

import io
import logging
from typing import Any, Optional

from app.config import OCRConfig, get_config

logger = logging.getLogger(__name__)

# Lazy-loaded PaddleOCR class reference
_PaddleOCR: Optional[type] = None
_PADDLE_AVAILABLE: Optional[bool] = None


def _check_paddle() -> bool:
    """Check whether PaddleOCR can be imported, caching the result."""
    global _PaddleOCR, _PADDLE_AVAILABLE
    if _PADDLE_AVAILABLE is not None:
        return _PADDLE_AVAILABLE
    try:
        from paddleocr import PaddleOCR as _Cls  # type: ignore[import-untyped]
        _PaddleOCR = _Cls
        _PADDLE_AVAILABLE = True
        logger.info("PaddleOCR is available")
    except ImportError:
        _PaddleOCR = None
        _PADDLE_AVAILABLE = False
        logger.warning(
            "PaddleOCR is not installed. OCR features are disabled. "
            "Install with: pip install paddlepaddle paddleocr"
        )
    return _PADDLE_AVAILABLE


class OCREngine:
    """
    Wrapper around PaddleOCR for processing scanned PDFs.

    Usage::

        engine = OCREngine(config.ocr)
        if engine.is_available():
            results = engine.process_pdf("report.pdf")

    Attributes:
        config: The :pyclass:`OCRConfig` instance controlling GPU usage,
                language, and detection thresholds.
    """

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        """
        Initialise the OCR engine.

        Args:
            config: OCR configuration.  Falls back to ``get_config().ocr``
                    if not provided.
        """
        self.config = config or get_config().ocr
        self._ocr_instance: Any = None

        if _check_paddle():
            try:
                self._ocr_instance = _PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.language,
                    use_gpu=self.config.use_gpu,
                    gpu_mem=self.config.gpu_mem,
                    det_db_thresh=self.config.det_db_thresh,
                    rec_batch_num=self.config.rec_batch_num,
                    show_log=self.config.show_log,
                )
                logger.info(
                    "PaddleOCR initialised (lang=%s, gpu=%s)",
                    self.config.language,
                    self.config.use_gpu,
                )
            except Exception as exc:
                logger.error("Failed to initialise PaddleOCR: %s", exc)
                self._ocr_instance = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` if PaddleOCR is installed and initialised."""
        return self._ocr_instance is not None

    def process_pdf(self, pdf_path: str) -> list[dict[str, Any]]:
        """
        OCR every page of a PDF and return structured results.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of dicts, one per page::

                [
                    {"page_num": 1, "text": "...", "confidence": 0.93},
                    ...
                ]

            Returns an empty list if PaddleOCR is not available.
        """
        if not self.is_available():
            logger.warning("OCR engine not available — returning empty results")
            return []

        logger.info("Starting OCR for PDF: %s", pdf_path)
        images = self._pdf_to_images(pdf_path)
        if not images:
            logger.warning("No images extracted from PDF: %s", pdf_path)
            return []

        results: list[dict[str, Any]] = []
        for idx, img in enumerate(images, start=1):
            try:
                text, avg_conf = self._process_single_image(img)
                results.append({
                    "page_num": idx,
                    "text": text,
                    "confidence": round(avg_conf, 4),
                })
                logger.debug(
                    "OCR page %d: %d chars, confidence %.2f",
                    idx,
                    len(text),
                    avg_conf,
                )
            except Exception as exc:
                logger.error("OCR failed on page %d: %s", idx, exc)
                results.append({
                    "page_num": idx,
                    "text": "",
                    "confidence": 0.0,
                })

        total_chars = sum(len(r["text"]) for r in results)
        logger.info(
            "OCR complete — %d pages, %d total chars", len(results), total_chars
        )
        return results

    def process_image(self, image: Any) -> str:
        """
        OCR a single PIL Image and return the extracted text.

        Args:
            image: A ``PIL.Image.Image`` instance.

        Returns:
            Extracted text, or an empty string on failure.
        """
        if not self.is_available():
            logger.warning("OCR engine not available")
            return ""
        try:
            text, _ = self._process_single_image(image)
            return text
        except Exception as exc:
            logger.error("OCR failed on image: %s", exc)
            return ""

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _process_single_image(self, image: Any) -> tuple[str, float]:
        """
        Run PaddleOCR on a single image.

        Args:
            image: PIL Image.

        Returns:
            Tuple of (extracted_text, average_confidence).
        """
        import numpy as np  # local import to avoid top-level dep for non-OCR paths

        # Convert PIL Image → numpy array for PaddleOCR
        img_array = np.array(image)

        result = self._ocr_instance.ocr(img_array, cls=self.config.use_angle_cls)

        if not result or not result[0]:
            return "", 0.0

        lines: list[str] = []
        confidences: list[float] = []

        for line_info in result[0]:
            # Each line_info: [bbox, (text, confidence)]
            if line_info and len(line_info) >= 2:
                text_conf = line_info[1]
                if isinstance(text_conf, (tuple, list)) and len(text_conf) >= 2:
                    lines.append(str(text_conf[0]))
                    confidences.append(float(text_conf[1]))

        text = "\n".join(lines)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return text, avg_confidence

    def _pdf_to_images(self, pdf_path: str) -> list[Any]:
        """
        Convert each PDF page to a PIL Image using PyMuPDF.

        Each page is rendered at 300 DPI for good OCR quality.

        Args:
            pdf_path: Path to the PDF.

        Returns:
            List of ``PIL.Image.Image`` objects, one per page.
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
        except ImportError as exc:
            logger.error(
                "PyMuPDF and Pillow are required for PDF-to-image conversion: %s",
                exc,
            )
            return []

        images: list[Any] = []
        doc = None

        try:
            doc = fitz.open(pdf_path)
            logger.debug("Converting %d PDF pages to images", doc.page_count)

            for page_idx in range(doc.page_count):
                try:
                    page = doc.load_page(page_idx)
                    # Render at 300 DPI (default is 72, so zoom = 300/72 ≈ 4.17)
                    zoom = 300.0 / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                except Exception as page_exc:
                    logger.warning(
                        "Failed to convert page %d to image: %s",
                        page_idx + 1,
                        page_exc,
                    )
        except Exception as exc:
            logger.error("Failed to open PDF for image conversion: %s", exc)
        finally:
            if doc:
                doc.close()

        return images
