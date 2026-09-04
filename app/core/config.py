from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/rag.db"
    upload_dir: str = "./data/documents"
    chroma_persist_dir: str = "./data/chroma"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-5"

    # Shared between embeddings and (later) generation -- Anthropic has no
    # first-party embeddings API, so this key only ever affects embeddings
    # via Google, never Anthropic.
    google_api_key: Optional[str] = None
    google_generation_model: str = "gemini-2.5-flash"
    google_embedding_model: str = "gemini-embedding-001"
    google_embedding_dimensions: int = 768

    # Plain str, not List[str]: newer pydantic-settings (2.11+) tries to
    # JSON-decode env values for list-typed fields *before* any
    # mode="before" validator runs, and raises instead of falling back to
    # the raw string on failure — so a plain comma-separated env var like
    # "http://a,http://b" crashes at startup instead of reaching a
    # validator that could split it. Splitting via a property sidesteps
    # that parsing path entirely.
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:3000", alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


settings = Settings()
