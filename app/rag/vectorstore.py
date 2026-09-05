from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.rag.chunker import ChunkedDocument
from app.rag.embeddings import TextEmbedder


class ChromaIngestStore:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedder: TextEmbedder,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def ingest_chunks(self, chunks: list[ChunkedDocument]) -> int:
        if not chunks:
            return 0

        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "source_location": chunk.source_location,
            }
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = self.embedder.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def peek(self, limit: int = 5) -> dict[str, Any]:
        return self.collection.get(
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )
