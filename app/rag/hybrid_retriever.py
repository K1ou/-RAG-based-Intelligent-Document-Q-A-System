from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.rag.retriever import ChromaRetriever, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []
    if " " in normalized:
        return [tok for tok in re.split(r"\s+", normalized) if tok]
    return list(normalized)


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


class HybridRetriever:
    def __init__(self, dense_retriever: ChromaRetriever) -> None:
        self.dense_retriever = dense_retriever
        self.collection = dense_retriever.collection

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if dense_k <= 0 or sparse_k <= 0:
            raise ValueError("dense_k and sparse_k must be greater than 0")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.collection.count() == 0:
            return []

        dense_results = self.dense_retriever.search(query=query, top_k=dense_k)
        dense_scores_raw = [item.score for item in dense_results]
        dense_scores_norm = _normalize_scores(dense_scores_raw)
        dense_by_id: dict[str, tuple[RetrievedChunk, float]] = {}
        for item, score in zip(dense_results, dense_scores_norm):
            dense_by_id[item.chunk_id] = (item, score)

        all_records = self.collection.get(include=["documents", "metadatas"])
        ids: list[str] = all_records.get("ids", [])
        docs: list[str] = all_records.get("documents", [])
        metas: list[dict[str, str]] = all_records.get("metadatas", [])

        tokenized_docs = [_tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = _tokenize(query)
        sparse_scores_all = list(bm25.get_scores(query_tokens))

        ranked_sparse_indices = sorted(
            range(len(sparse_scores_all)),
            key=lambda idx: sparse_scores_all[idx],
            reverse=True,
        )[: min(sparse_k, len(sparse_scores_all))]

        sparse_items: list[RetrievedChunk] = []
        sparse_scores_raw: list[float] = []
        for idx in ranked_sparse_indices:
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            sparse_items.append(
                RetrievedChunk(
                    chunk_id=ids[idx],
                    text=docs[idx],
                    source_file=str(meta.get("source_file", "")),
                    source_location=str(meta.get("source_location", "")),
                    score=float(sparse_scores_all[idx]),
                )
            )
            sparse_scores_raw.append(float(sparse_scores_all[idx]))

        sparse_scores_norm = _normalize_scores(sparse_scores_raw)
        sparse_by_id: dict[str, tuple[RetrievedChunk, float]] = {}
        for item, score in zip(sparse_items, sparse_scores_norm):
            sparse_by_id[item.chunk_id] = (item, score)

        merged_ids = set(dense_by_id.keys()) | set(sparse_by_id.keys())
        fused: list[RetrievedChunk] = []
        for chunk_id in merged_ids:
            dense_item, dense_score = dense_by_id.get(chunk_id, (None, 0.0))
            sparse_item, sparse_score = sparse_by_id.get(chunk_id, (None, 0.0))
            base_item = dense_item or sparse_item
            if base_item is None:
                continue
            final_score = alpha * dense_score + (1.0 - alpha) * sparse_score
            fused.append(
                RetrievedChunk(
                    chunk_id=base_item.chunk_id,
                    text=base_item.text,
                    source_file=base_item.source_file,
                    source_location=base_item.source_location,
                    score=final_score,
                )
            )

        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]
