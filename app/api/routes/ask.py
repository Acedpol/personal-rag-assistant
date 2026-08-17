from fastapi import APIRouter

from app.core.config import settings
from app.rag.llm_provider import get_llm_provider
from app.rag.vector_store import search as vector_search
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.search import SearchResult

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    top_k = request.top_k or settings.retrieval_top_k
    matches = vector_search(request.question, top_k)

    sources = [
        SearchResult(
            chunk_id=match["chunk_id"],
            document_id=match["document_id"],
            chunk_index=match["chunk_index"],
            text=match["text"],
            similarity=1 - match["distance"],
        )
        for match in matches
    ]

    provider = get_llm_provider()
    answer = provider.generate_answer(request.question, [source.text for source in sources])

    return AskResponse(answer=answer, sources=sources, provider=provider.__class__.__name__)
