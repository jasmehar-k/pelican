"""ChromaDB-backed paper store for researcher deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from pelican.utils.config import get_settings

_COLLECTION_NAME = "arxiv_papers"
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _collection():
    settings = get_settings()
    path = Path(settings.data_dir) / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata=_COLLECTION_METADATA,
    )


def has_paper(arxiv_id: str) -> bool:
    collection = _collection()
    try:
        result = collection.get(ids=[arxiv_id])
    except Exception:
        return False
    return bool(result and result.get("ids"))


def store_paper(arxiv_id: str, title: str, abstract: str, metadata: dict[str, Any]) -> None:
    collection = _collection()
    payload = {**metadata, "title": title, "abstract": abstract, "arxiv_id": arxiv_id}
    if isinstance(payload.get("authors"), list):
        payload["authors"] = ", ".join(str(author) for author in payload["authors"])
    collection.upsert(
        ids=[arxiv_id],
        documents=[abstract],
        metadatas=[payload],
    )


def find_similar(text: str, n_results: int = 3, threshold: float = 0.92) -> list[dict[str, Any]]:
    collection = _collection()
    try:
        results = collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )
    except Exception:
        return []

    if not results.get("ids"):
        return []

    matches: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    for index, arxiv_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else 1.0
        similarity = 1.0 - float(distance)
        if similarity < threshold:
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        matches.append({
            "arxiv_id": arxiv_id,
            "similarity": similarity,
            "metadata": metadata,
            "abstract": documents[index] if index < len(documents) else "",
        })
    return matches
