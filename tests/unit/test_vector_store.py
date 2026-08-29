from app.core.config import settings
from app.rag import vector_store


def test_collection_name_uses_local_suffix_without_google_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)

    assert vector_store.get_collection_name() == "document_chunks_local_384d"


def test_collection_name_uses_google_suffix_with_google_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-key-for-naming-test")
    monkeypatch.setattr(settings, "google_embedding_dimensions", 768)

    assert vector_store.get_collection_name() == "document_chunks_google_768d"


def test_get_embedding_provider_returns_local_by_default(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)

    provider = vector_store.get_embedding_provider()

    assert isinstance(provider, vector_store.LocalEmbeddingProvider)


def test_get_embedding_provider_returns_google_when_key_present(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-key-for-naming-test")

    provider = vector_store.get_embedding_provider()

    assert isinstance(provider, vector_store.GoogleEmbeddingProvider)
