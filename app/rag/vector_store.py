from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.core.config import settings

COLLECTION_NAME = "document_chunks"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> Collection:
    # Chroma defaults to squared L2 distance, which isn't bounded and isn't
    # what sentence-transformers embeddings are meant to be compared with —
    # cosine distance (bounded [0, 2]) is the standard choice for text
    # embeddings, and what similarity scores below actually assume.
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


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
