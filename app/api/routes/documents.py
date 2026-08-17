from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.rag.chunking import split_text
from app.schemas.document import ChunkPreview, DocumentDetail, DocumentRead
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Document:
    # Plain `def`, not `async def`: FastAPI runs sync routes in a thread
    # pool automatically. ingest_document() does blocking work (file I/O,
    # pypdf extraction, sentence-transformers encoding) — as `async def`,
    # that would run directly on the event loop and freeze the entire
    # server (including unrelated requests, even /health) for however long
    # embedding takes, which is exactly what happened when this was async.
    content = file.file.read()
    try:
        return document_service.ingest_document(
            db, file.filename or "untitled", file.content_type or "", content
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return document_service.list_documents(db)


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    return document_service.get_document_or_404(db, document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    document_service.delete_document(db, document_id)


@router.get("/{document_id}/chunks", response_model=list[ChunkPreview])
def preview_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[ChunkPreview]:
    document = document_service.get_document_or_404(db, document_id)
    chunks = split_text(document.extracted_text)
    return [
        ChunkPreview(index=i, text=chunk, char_count=len(chunk)) for i, chunk in enumerate(chunks)
    ]
