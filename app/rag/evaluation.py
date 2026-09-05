from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.rag.chunker import split_parsed_chunks
from app.rag.embeddings import TextEmbedder
from app.rag.parser import parse_document
from app.rag.retriever import ChromaRetriever
from app.rag.vectorstore import ChromaIngestStore


@dataclass(slots=True)
class EvalItem:
    query: str
    expected_answer_substring: str


@dataclass(slots=True)
class EvalResult:
    chunk_size: int
    chunk_overlap: int
    total_queries: int
    hit_at_k: float
    avg_latency_ms: float


def load_eval_items(file_path: str | Path) -> list[EvalItem]:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[EvalItem] = []
    for row in payload:
        query = str(row.get("query", "")).strip()
        expected = str(row.get("expected_answer_substring", "")).strip()
        if not query or not expected:
            continue
        items.append(EvalItem(query=query, expected_answer_substring=expected))
    return items


def evaluate_chunk_params(
    doc_files: Iterable[str | Path],
    eval_items: list[EvalItem],
    embedder: TextEmbedder,
    store_dir: str | Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int = 5,
) -> EvalResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not eval_items:
        raise ValueError("eval_items must not be empty")

    store = ChromaIngestStore(
        persist_directory=store_dir,
        collection_name=collection_name,
        embedder=embedder,
    )

    for file_path in doc_files:
        parsed = parse_document(file_path)
        chunked = split_parsed_chunks(
            parsed,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        store.ingest_chunks(chunked)

    retriever = ChromaRetriever(
        persist_directory=store_dir,
        collection_name=collection_name,
        embedder=embedder,
    )

    hit_count = 0
    total_latency_ms = 0.0

    for item in eval_items:
        started = time.perf_counter()
        results = retriever.search(item.query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_latency_ms += elapsed_ms

        matched = any(item.expected_answer_substring in r.text for r in results)
        if matched:
            hit_count += 1

    total = len(eval_items)
    return EvalResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_queries=total,
        hit_at_k=hit_count / total,
        avg_latency_ms=total_latency_ms / total,
    )
