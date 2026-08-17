from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: int
    chunk_index: int
    text: str
    similarity: float
