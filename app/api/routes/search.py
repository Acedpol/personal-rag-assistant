from fastapi import APIRouter

from app.core.config import settings
from app.rag.vector_store import search as vector_search
from app.schemas.search import SearchRequest, SearchResult

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=list[SearchResult])
def search_documents(request: SearchRequest) -> list[SearchResult]:
    top_k = request.top_k or settings.retrieval_top_k
    matches = vector_search(request.query, top_k)
    return [
        SearchResult(
            chunk_id=match["chunk_id"],
            document_id=match["document_id"],
            chunk_index=match["chunk_index"],
            text=match["text"],
            similarity=1 - match["distance"],
        )
        for match in matches
    ]
