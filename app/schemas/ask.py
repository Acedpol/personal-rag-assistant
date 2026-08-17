from typing import Optional

from pydantic import BaseModel

from app.schemas.search import SearchResult


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    provider: str
