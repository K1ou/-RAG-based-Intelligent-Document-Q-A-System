from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.rag.generator import GeneratedAnswer, OllamaClient
from app.rag.pipeline import EchoLLMClient, RAGPipeline


@dataclass(slots=True)
class ModeEvalItem:
    query: str
    expected_answer_substring: str
    should_abstain: bool = False
    topic: str = "general"
    expected_source_file: str = ""
    expected_source_location: str = ""


@dataclass(slots=True)
class ModeEvalResult:
    mode: str
    total_queries: int
    answer_hit_rate: float
    citation_hit_rate: float
    strict_citation_hit_rate: float
    abstain_rate: float
    hallucination_proxy_rate: float
    avg_latency_ms: float


@dataclass(slots=True)
class _ModeEvalSample:
    expected_substring: str
    should_abstain: bool
    topic: str
    expected_source_file: str
    expected_source_location: str
    answer_text: str
    citation_text: str
    citation_sources: set[tuple[str, str]]
    latency_ms: float


def load_mode_eval_items(file_path: str | Path) -> list[ModeEvalItem]:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    items: list[ModeEvalItem] = []
    for row in payload:
        query = str(row.get("query", "")).strip()
        expected = str(row.get("expected_answer_substring", "")).strip()
        if not query or not expected:
            continue
        should_abstain = bool(row.get("should_abstain", False))
        topic = str(row.get("topic", "general")).strip() or "general"
        expected_source_file = str(row.get("expected_source_file", "")).strip()
        expected_source_location = str(row.get("expected_source_location", "")).strip()
        items.append(
            ModeEvalItem(
                query=query,
                expected_answer_substring=expected,
                should_abstain=should_abstain,
                topic=topic,
                expected_source_file=expected_source_file,
                expected_source_location=expected_source_location,
            )
        )
    return items


def _is_abstain(answer_text: str) -> bool:
    normalized = answer_text.strip()
    return "无法确定" in normalized


def _citations_to_text(answer: GeneratedAnswer) -> str:
    return "\n".join(c.quote_excerpt for c in answer.citations)


def _citations_to_sources(answer: GeneratedAnswer) -> set[tuple[str, str]]:
    return {(c.source_file.strip(), c.source_location.strip()) for c in answer.citations}


def _build_mode_result(mode: str, samples: list[_ModeEvalSample]) -> ModeEvalResult:
    if not samples:
        raise ValueError("samples must not be empty")

    answer_hit_count = 0
    citation_hit_count = 0
    strict_citation_hit_count = 0
    abstain_count = 0
    unanswerable_total = 0
    unanswerable_not_abstain = 0
    total_latency_ms = 0.0

    for sample in samples:
        total_latency_ms += sample.latency_ms
        if sample.expected_substring in sample.answer_text:
            answer_hit_count += 1
        if sample.expected_substring in sample.citation_text:
            citation_hit_count += 1

        if sample.expected_source_file and sample.expected_source_location:
            expected_pair = (sample.expected_source_file, sample.expected_source_location)
            if expected_pair in sample.citation_sources:
                strict_citation_hit_count += 1
        elif sample.expected_substring in sample.citation_text:
            strict_citation_hit_count += 1

        is_abstain = _is_abstain(sample.answer_text)
        if is_abstain:
            abstain_count += 1

        if sample.should_abstain:
            unanswerable_total += 1
            if not is_abstain:
                unanswerable_not_abstain += 1

    total = len(samples)
    hallucination_proxy_rate = (
        unanswerable_not_abstain / unanswerable_total if unanswerable_total > 0 else 0.0
    )

    return ModeEvalResult(
        mode=mode,
        total_queries=total,
        answer_hit_rate=answer_hit_count / total,
        citation_hit_rate=citation_hit_count / total,
        strict_citation_hit_rate=strict_citation_hit_count / total,
        abstain_rate=abstain_count / total,
        hallucination_proxy_rate=hallucination_proxy_rate,
        avg_latency_ms=total_latency_ms / total,
    )


def summarize_by_topic(samples: list[_ModeEvalSample]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[_ModeEvalSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.topic, []).append(sample)

    summary: dict[str, dict[str, float]] = {}
    for topic, topic_samples in grouped.items():
        total = len(topic_samples)
        if total == 0:
            continue

        answer_hits = sum(1 for s in topic_samples if s.expected_substring in s.answer_text)
        citation_hits = sum(1 for s in topic_samples if s.expected_substring in s.citation_text)
        strict_hits = 0
        for s in topic_samples:
            if s.expected_source_file and s.expected_source_location:
                if (s.expected_source_file, s.expected_source_location) in s.citation_sources:
                    strict_hits += 1
            elif s.expected_substring in s.citation_text:
                strict_hits += 1

        abstains = sum(1 for s in topic_samples if _is_abstain(s.answer_text))
        avg_latency = sum(s.latency_ms for s in topic_samples) / total

        summary[topic] = {
            "total_queries": float(total),
            "answer_hit_rate": answer_hits / total,
            "citation_hit_rate": citation_hits / total,
            "strict_citation_hit_rate": strict_hits / total,
            "abstain_rate": abstains / total,
            "avg_latency_ms": avg_latency,
        }
    return summary


def evaluate_retrieval_modes(
    doc_files: Iterable[str | Path],
    eval_items: list[ModeEvalItem],
    persist_directory: str | Path,
    collection_name: str,
    embedder_name: str = "hash",
    llm_name: str = "echo",
    llm_model: str = "qwen2:7b",
    modes: Iterable[str] = ("dense", "hybrid", "hybrid_rerank"),
    top_k: int = 5,
    alpha: float = 0.5,
    enable_query_rewrite: bool = True,
    rewrite_mode: str = "rule",
    ollama_timeout_seconds: int = 120,
) -> list[ModeEvalResult]:
    if not eval_items:
        raise ValueError("eval_items must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    llm_client = (
        EchoLLMClient()
        if llm_name == "echo"
        else OllamaClient(timeout_seconds=ollama_timeout_seconds)
    )
    pipeline = RAGPipeline(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedder_name=embedder_name,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    pipeline.ingest_files(file_paths=doc_files)

    results: list[ModeEvalResult] = []
    for mode in modes:
        normalized_mode = mode.strip().lower()
        mode_samples: list[_ModeEvalSample] = []

        for item in eval_items:
            started = time.perf_counter()
            answer = pipeline.ask(
                question=item.query,
                top_k=top_k,
                retrieval_mode=normalized_mode,
                alpha=alpha,
                enable_query_rewrite=enable_query_rewrite,
                rewrite_mode=rewrite_mode,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            mode_samples.append(
                _ModeEvalSample(
                    expected_substring=item.expected_answer_substring,
                    should_abstain=item.should_abstain,
                    topic=item.topic,
                    expected_source_file=item.expected_source_file,
                    expected_source_location=item.expected_source_location,
                    answer_text=answer.answer,
                    citation_text=_citations_to_text(answer),
                    citation_sources=_citations_to_sources(answer),
                    latency_ms=elapsed_ms,
                )
            )

        results.append(_build_mode_result(normalized_mode, mode_samples))

    return results


def evaluate_retrieval_modes_with_topics(
    doc_files: Iterable[str | Path],
    eval_items: list[ModeEvalItem],
    persist_directory: str | Path,
    collection_name: str,
    embedder_name: str = "hash",
    llm_name: str = "echo",
    llm_model: str = "qwen2:7b",
    modes: Iterable[str] = ("dense", "hybrid", "hybrid_rerank"),
    top_k: int = 5,
    alpha: float = 0.5,
    enable_query_rewrite: bool = True,
    rewrite_mode: str = "rule",
    ollama_timeout_seconds: int = 120,
) -> tuple[list[ModeEvalResult], dict[str, dict[str, dict[str, float]]]]:
    if not eval_items:
        raise ValueError("eval_items must not be empty")

    llm_client = (
        EchoLLMClient()
        if llm_name == "echo"
        else OllamaClient(timeout_seconds=ollama_timeout_seconds)
    )
    pipeline = RAGPipeline(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedder_name=embedder_name,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    pipeline.ingest_files(file_paths=doc_files)

    mode_results: list[ModeEvalResult] = []
    topic_results: dict[str, dict[str, dict[str, float]]] = {}

    for mode in modes:
        normalized_mode = mode.strip().lower()
        mode_samples: list[_ModeEvalSample] = []

        for item in eval_items:
            started = time.perf_counter()
            answer = pipeline.ask(
                question=item.query,
                top_k=top_k,
                retrieval_mode=normalized_mode,
                alpha=alpha,
                enable_query_rewrite=enable_query_rewrite,
                rewrite_mode=rewrite_mode,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            mode_samples.append(
                _ModeEvalSample(
                    expected_substring=item.expected_answer_substring,
                    should_abstain=item.should_abstain,
                    topic=item.topic,
                    expected_source_file=item.expected_source_file,
                    expected_source_location=item.expected_source_location,
                    answer_text=answer.answer,
                    citation_text=_citations_to_text(answer),
                    citation_sources=_citations_to_sources(answer),
                    latency_ms=elapsed_ms,
                )
            )

        mode_results.append(_build_mode_result(normalized_mode, mode_samples))
        topic_results[normalized_mode] = summarize_by_topic(mode_samples)

    return mode_results, topic_results
