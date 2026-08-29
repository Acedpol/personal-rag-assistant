from pydantic import BaseModel


class GenerationProviders(BaseModel):
    default: str
    available: list[str]


class EmbeddingProviderInfo(BaseModel):
    active: str


class ProvidersResponse(BaseModel):
    generation: GenerationProviders
    embeddings: EmbeddingProviderInfo
