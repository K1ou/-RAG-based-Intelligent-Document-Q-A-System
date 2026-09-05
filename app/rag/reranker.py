from __future__ import annotations

import re
from typing import Protocol

from app.rag.retriever import RetrievedChunk


def _tokenize(text: str) -> set[str]:
    normalized = text.strip().lower()
    if not normalized:
        return set()
    if " " in normalized:
        return {tok for tok in re.split(r"\s+", normalized) if tok}
    return set(normalized)


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...


class TokenOverlapReranker:
    """Lightweight reranker for local tests and CPU-only fallback."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = _tokenize(query)
        scored: list[tuple[float, RetrievedChunk]] = []
        for item in candidates:
            doc_tokens = _tokenize(item.text)
            overlap = len(query_tokens & doc_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            rerank_score = 0.7 * coverage + 0.3 * max(item.score, 0.0)
            scored.append((rerank_score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[RetrievedChunk] = []
        for score, item in scored[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    source_file=item.source_file,
                    source_location=item.source_location,
                    score=float(score),
                )
            )
        return results


class BGEReranker:
    """Cross-encoder reranker for better relevance in real runs."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not candidates:
            return []

        pairs = [[query, item.text] for item in candidates]
        scores = self.model.predict(pairs)
        merged = list(zip(scores, candidates))
        merged.sort(key=lambda pair: float(pair[0]), reverse=True)

        results: list[RetrievedChunk] = []
        for score, item in merged[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    source_file=item.source_file,
                    source_location=item.source_location,
                    score=float(score),
                )
            )
        return results
