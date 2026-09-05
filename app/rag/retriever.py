from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.rag.embeddings import TextEmbedder


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    source_location: str
    score: float


class ChromaRetriever:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedder: TextEmbedder,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_texts([query])[0]
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=str(metadata.get("source_file", "")),
                    source_location=str(metadata.get("source_location", "")),
                    score=similarity,
                )
            )

        return results
