from typing import Literal, Optional

from pydantic import BaseModel

from app.schemas.search import SearchResult


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    # None = use the server's default provider (see GET /providers).
    provider: Optional[Literal["google", "anthropic"]] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    provider: str
