"""PDF text extraction helpers for the Researcher agent."""

from __future__ import annotations

from io import BytesIO

import httpx
from pypdf import PdfReader


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
        text = "\n".join(page.extract_text() or "" for page in pages).strip()
        return text
    except Exception:
        return ""
