import io

from pypdf import PdfReader

SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain"}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_text(content_type: str, content: bytes) -> str:
    if content_type == "application/pdf":
        return extract_text_from_pdf(content)
    if content_type == "text/plain":
        return content.decode("utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported content type: {content_type}")
