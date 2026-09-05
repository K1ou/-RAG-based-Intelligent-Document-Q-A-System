from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.rag.chunker import split_parsed_chunks
from app.rag.conversation_memory import ConversationTurn, build_contextualized_query
from app.rag.embeddings import BGEEmbedder, HashEmbedder, TextEmbedder
from app.rag.generator import GeneratedAnswer, LLMClient, OllamaClient, RAGGenerator
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.parser import parse_document
from app.rag.query_rewriter import LLMQueryRewriter, RuleBasedQueryRewriter
from app.rag.reranker import TokenOverlapReranker
from app.rag.retriever import ChromaRetriever
from app.rag.vectorstore import ChromaIngestStore


class EchoLLMClient:
    """Simple local LLM stub for smoke tests without Ollama dependency."""

    def generate(self, prompt: str, model: str) -> str:
        del model
        marker = "检索上下文:\n"
        if marker in prompt:
            context = prompt.split(marker, maxsplit=1)[1].strip()
            if context:
                preview = context.splitlines()[-1][:80]
                return f"基于检索内容，相关信息为: {preview}"
        return "根据已检索内容无法确定。"


def build_embedder(embedder_name: str) -> TextEmbedder:
    name = embedder_name.strip().lower()
    if name == "hash":
        return HashEmbedder(dimension=64)
    if name == "bge":
        return BGEEmbedder()
    raise ValueError("embedder_name must be one of: hash, bge")


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedder_name: str = "hash",
        llm_client: LLMClient | None = None,
        llm_model: str = "qwen2:7b",
    ) -> None:
        self.embedder = build_embedder(embedder_name)
        self.store = ChromaIngestStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=self.embedder,
        )
        self.retriever = ChromaRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=self.embedder,
        )
        self.hybrid_retriever = HybridRetriever(self.retriever)
        self.reranker = TokenOverlapReranker()
        self.rule_rewriter = RuleBasedQueryRewriter()
        self.llm_rewriter = LLMQueryRewriter(llm_client=llm_client or OllamaClient(), model=llm_model)
        self.generator = RAGGenerator(
            llm_client=llm_client or OllamaClient(),
            model=llm_model,
        )

    def ingest_files(
        self,
        file_paths: Iterable[str | Path],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        total = 0
        for file_path in file_paths:
            parsed = parse_document(file_path)
            chunked = split_parsed_chunks(
                parsed,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            total += self.store.ingest_chunks(chunked)
        return total

    def ask(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "dense",
        alpha: float = 0.5,
        enable_query_rewrite: bool = False,
        rewrite_mode: str = "rule",
        conversation_turns: list[ConversationTurn] | None = None,
    ) -> GeneratedAnswer:
        retrieval_query = question
        if conversation_turns:
            retrieval_query = build_contextualized_query(question, conversation_turns)

        if enable_query_rewrite:
            normalized_rewrite_mode = rewrite_mode.strip().lower()
            if normalized_rewrite_mode == "llm":
                retrieval_query = self.llm_rewriter.rewrite(retrieval_query)
            else:
                retrieval_query = self.rule_rewriter.rewrite(retrieval_query)

        mode = retrieval_mode.strip().lower()
        if mode == "hybrid":
            contexts = self.hybrid_retriever.search(
                query=retrieval_query,
                top_k=top_k,
                dense_k=max(top_k, 10),
                sparse_k=max(top_k, 10),
                alpha=alpha,
            )
        elif mode == "hybrid_rerank":
            candidates = self.hybrid_retriever.search(
                query=retrieval_query,
                top_k=max(top_k, 20),
                dense_k=max(top_k, 20),
                sparse_k=max(top_k, 20),
                alpha=alpha,
            )
            contexts = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            contexts = self.retriever.search(retrieval_query, top_k=top_k)
        return self.generator.answer(question=question, contexts=contexts)
