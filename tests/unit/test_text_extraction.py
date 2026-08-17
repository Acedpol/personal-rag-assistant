from pathlib import Path

import pytest

from app.rag.text_extraction import extract_text, extract_text_from_pdf

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_extract_text_from_plain_text():
    result = extract_text("text/plain", "Hola mundo".encode("utf-8"))
    assert result == "Hola mundo"


def test_extract_text_rejects_unsupported_type():
    with pytest.raises(ValueError):
        extract_text("application/json", b"{}")


def test_extract_text_from_real_pdf():
    pdf_bytes = (FIXTURES_DIR / "sample.pdf").read_bytes()
    result = extract_text_from_pdf(pdf_bytes)
    assert "Documento de prueba" in result
    assert "pypdf" in result
