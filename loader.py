"""Document loader — PDF and DOCX to raw text."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_document(path: str) -> str:
    """Load a PDF or DOCX file and return its raw text content."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".docx":
        return _load_docx(file_path)
    elif suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in (".txt", ".text"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_docx(path: Path) -> str:
    try:
        import docx  # python-docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        logger.info("Loaded DOCX %s — %d paragraphs", path.name, len(paragraphs))
        return "\n".join(paragraphs)
    except ImportError:
        logger.warning("python-docx not installed; falling back to text read")
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        logger.info("Loaded PDF %s — %d pages", path.name, len(text_parts))
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed; trying PyPDF2")
        try:
            import PyPDF2
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                return "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
        except ImportError:
            raise ImportError("Install pdfplumber or PyPDF2 to process PDFs")
