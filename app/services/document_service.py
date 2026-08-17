import os
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.rag.chunking import split_text
from app.rag.text_extraction import SUPPORTED_CONTENT_TYPES, extract_text
from app.rag.vector_store import delete_document_chunks, index_document_chunks


def _save_uploaded_file(filename: str, content: bytes) -> str:
    os.makedirs(settings.upload_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(settings.upload_dir, safe_name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def ingest_document(db: Session, filename: str, content_type: str, content: bytes) -> Document:
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")

    text = extract_text(content_type, content)
    if not text.strip():
        raise ValueError("No text could be extracted from this file")

    _save_uploaded_file(filename, content)
    chunks = split_text(text)

    document = Document(
        filename=filename,
        content_type=content_type,
        extracted_text=text,
        char_count=len(text),
        chunk_count=len(chunks),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    index_document_chunks(document.id, chunks)

    return document


def list_documents(db: Session) -> list[Document]:
    return list(db.execute(select(Document).order_by(Document.uploaded_at.desc())).scalars())


def get_document_or_404(db: Session, document_id: int) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


def delete_document(db: Session, document_id: int) -> None:
    document = get_document_or_404(db, document_id)
    delete_document_chunks(document_id)
    db.delete(document)
    db.commit()
