from fastapi import APIRouter

from app.rag.llm_provider import available_providers, default_provider
from app.rag.vector_store import GoogleEmbeddingProvider, get_embedding_provider
from app.schemas.providers import EmbeddingProviderInfo, GenerationProviders, ProvidersResponse

router = APIRouter(tags=["providers"])


@router.get("/providers", response_model=ProvidersResponse)
def get_providers() -> ProvidersResponse:
    embedding_active = "google" if isinstance(get_embedding_provider(), GoogleEmbeddingProvider) else "local"
    return ProvidersResponse(
        generation=GenerationProviders(default=default_provider(), available=available_providers()),
        embeddings=EmbeddingProviderInfo(active=embedding_active),
    )
