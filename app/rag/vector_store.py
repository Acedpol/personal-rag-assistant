from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.core.config import settings

COLLECTION_PREFIX = "document_chunks"


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def collection_suffix(self) -> str:
        """Identifies this provider+dimension in the Chroma collection name.

        Embeddings from different providers/dimensions can't share a Chroma
        collection (it's dimension-locked) -- baking the suffix into the
        collection name means switching providers routes to a fresh,
        isolated collection automatically instead of corrupting the old one.
        """
        raise NotImplementedError


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


class LocalEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        model = get_embedding_model()
        return model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def collection_suffix(self) -> str:
        return "local_384d"


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dimensions: int):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._types = types

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=self._types.EmbedContentConfig(output_dimensionality=self._dimensions),
        )
        return [embedding.values for embedding in response.embeddings]

    @property
    def collection_suffix(self) -> str:
        return f"google_{self._dimensions}d"


def get_embedding_provider() -> EmbeddingProvider:
    if settings.google_api_key:
        return GoogleEmbeddingProvider(
            settings.google_api_key,
            settings.google_embedding_model,
            settings.google_embedding_dimensions,
        )
    return LocalEmbeddingProvider()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection_name() -> str:
    return f"{COLLECTION_PREFIX}_{get_embedding_provider().collection_suffix}"


def get_collection() -> Collection:
    # Chroma defaults to squared L2 distance, which isn't bounded and isn't
    # what these embeddings are meant to be compared with — cosine distance
    # (bounded [0, 2]) is the standard choice for text embeddings, and what
    # similarity scores below actually assume.
    provider = get_embedding_provider()
    return get_chroma_client().get_or_create_collection(
        name=get_collection_name(),
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": type(provider).__name__,
        },
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_provider().embed(texts)


def index_document_chunks(document_id: int, chunks: list[str]) -> None:
    if not chunks:
        return

    collection = get_collection()
    embeddings = embed_texts(chunks)
    ids = [f"doc{document_id}-chunk{i}" for i in range(len(chunks))]
    metadatas: list[dict[str, Any]] = [
        {"document_id": document_id, "chunk_index": i} for i in range(len(chunks))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def delete_document_chunks(document_id: int) -> None:
    get_collection().delete(where={"document_id": document_id})


def search(query: str, top_k: int) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "chunk_id": ids[i],
            "text": documents[i],
            "document_id": metadatas[i]["document_id"],
            "chunk_index": metadatas[i]["chunk_index"],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]
