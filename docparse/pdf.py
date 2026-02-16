from __future__ import annotations

from pathlib import Path

from .errors import DocParseError, OCRUnavailableError


def extract_text_pages(pdf_path: Path, max_pages: int | None = None) -> list[tuple[int, str]]:
    fitz_error: Exception | None = None
    try:
        return _extract_text_with_fitz(pdf_path, max_pages=max_pages)
    except Exception as exc:  # pragma: no cover - depends on local pdf stack.
        fitz_error = exc

    try:
        return _extract_text_with_pdfplumber(pdf_path, max_pages=max_pages)
    except Exception as plumber_exc:  # pragma: no cover - depends on local pdf stack.
        raise DocParseError(
            f"Failed to extract text with both PyMuPDF and pdfplumber. "
            f"PyMuPDF error: {fitz_error}; pdfplumber error: {plumber_exc}"
        ) from plumber_exc


def render_pages_to_png(
    pdf_path: Path,
    dpi: int,
    page_numbers: list[int] | None = None,
) -> list[tuple[int, bytes]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise OCRUnavailableError(
            "OCR rendering requires PyMuPDF (fitz). Install with: pip install pymupdf"
        ) from exc

    rendered: list[tuple[int, bytes]] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
        if page_numbers is None:
            page_numbers = [index + 1 for index in range(doc.page_count)]

        for page_number in page_numbers:
            page_index = page_number - 1
            if page_index < 0 or page_index >= doc.page_count:
                raise OCRUnavailableError(
                    f"Cannot render page {page_number}; PDF has {doc.page_count} pages."
                )

            pix = doc.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            rendered.append((page_number, pix.tobytes("png")))

    return rendered


def _extract_text_with_fitz(
    pdf_path: Path,
    max_pages: int | None = None,
) -> list[tuple[int, str]]:
    import fitz

    pages: list[tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
        total_pages = doc.page_count
        read_pages = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_index in range(read_pages):
            page = doc.load_page(page_index)
            pages.append((page_index + 1, page.get_text("text") or ""))

    return pages


def _extract_text_with_pdfplumber(
    pdf_path: Path,
    max_pages: int | None = None,
) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:  # type: ignore[attr-defined]
        total_pages = len(pdf.pages)
        read_pages = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_index in range(read_pages):
            text = pdf.pages[page_index].extract_text() or ""
            pages.append((page_index + 1, text))

    return pages
