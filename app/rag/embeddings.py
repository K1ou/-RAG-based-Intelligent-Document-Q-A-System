from __future__ import annotations

import hashlib
from typing import Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbedder:
    """Deterministic lightweight embedder for local tests and smoke runs."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for index in range(self.dimension):
            digest = hashlib.sha256(f"{text}:{index}".encode("utf-8")).digest()
            int_value = int.from_bytes(digest[:4], byteorder="big", signed=False)
            values[index] = (int_value / 4294967295.0) * 2.0 - 1.0
        return values


class BGEEmbedder:
    """SentenceTransformer embedder for production-like local runs."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
