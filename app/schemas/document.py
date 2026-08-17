from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    char_count: int
    chunk_count: int
    uploaded_at: datetime


class DocumentDetail(DocumentRead):
    extracted_text: str


class ChunkPreview(BaseModel):
    index: int
    text: str
    char_count: int
