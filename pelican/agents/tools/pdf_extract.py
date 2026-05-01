"""PDF text extraction helpers for the Researcher agent."""

from __future__ import annotations

import re
from io import BytesIO

import httpx
from pypdf import PdfReader

# Lines where more than 40% of non-space characters are non-ASCII are
# almost certainly garbled math/symbol encoding from PDF glyph fonts.
_GARBLE_THRESHOLD = 0.4


def _clean_pdf_text(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        non_space = [c for c in stripped if not c.isspace()]
        if non_space:
            garble_ratio = sum(1 for c in non_space if ord(c) > 127) / len(non_space)
            if garble_ratio > _GARBLE_THRESHOLD:
                continue
        cleaned.append(stripped)
    # Collapse 3+ consecutive blank lines to one blank line
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def fetch_pdf_text(arxiv_id: str, max_pages: int = 5) -> str:
    try:
        response = httpx.get(
            f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            follow_redirects=True,
            timeout=30,
        )
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        pages = reader.pages[:max_pages]
        raw = "\n".join(page.extract_text() or "" for page in pages)
        return _clean_pdf_text(raw)
    except Exception:
        return ""
