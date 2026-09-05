from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request

from app.rag.retriever import RetrievedChunk


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    source_file: str
    source_location: str
    score: float
    quote_excerpt: str


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    citations: list[Citation]


class LLMClient(Protocol):
    def generate(self, prompt: str, model: str) -> str:
        ...


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.append(
            f"[C{idx}] source={item.source_file} location={item.source_location} chunk_id={item.chunk_id}\n{item.text}"
        )

    context_block = "\n\n".join(context_lines)
    return (
        "你是企业文档问答助手。请严格根据给定上下文作答。\n"
        "规则:\n"
        "1) 如果上下文不足以回答，请明确回复：'根据已检索内容无法确定'。\n"
        "2) 不要编造事实，不要使用上下文之外的信息。\n"
        "3) 每个关键结论后都标注引用编号，例如 [C1]、[C2]。\n"
        "4) 先给出简洁答案，再给出要点。\n\n"
        f"用户问题:\n{question}\n\n"
        "可用引用编号说明:\n"
        "- [C1] 表示第1条上下文，以此类推\n\n"
        f"检索上下文:\n{context_block}\n"
    )


def _ensure_citation_tags(answer_text: str, citation_count: int) -> str:
    if citation_count <= 0:
        return answer_text
    if "[C" in answer_text:
        return answer_text
    tags = " ".join(f"[C{i}]" for i in range(1, citation_count + 1))
    return f"{answer_text}\n\n引用: {tags}"


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, model: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Failed to call Ollama API: {exc}") from exc

        data = json.loads(body)
        return str(data.get("response", "")).strip()


class RAGGenerator:
    def __init__(self, llm_client: LLMClient, model: str = "qwen2:7b") -> None:
        self.llm_client = llm_client
        self.model = model

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")

        if not contexts:
            return GeneratedAnswer(
                answer="根据已检索内容无法确定。",
                citations=[],
            )

        prompt = build_prompt(question=question, contexts=contexts)
        response_text = self.llm_client.generate(prompt=prompt, model=self.model).strip()
        if not response_text:
            response_text = "根据已检索内容无法确定。"
        response_text = _ensure_citation_tags(response_text, citation_count=len(contexts))

        citations = [
            Citation(
                index=idx,
                chunk_id=item.chunk_id,
                source_file=item.source_file,
                source_location=item.source_location,
                score=float(item.score),
                quote_excerpt=item.text[:180],
            )
            for idx, item in enumerate(contexts, start=1)
        ]

        return GeneratedAnswer(answer=response_text, citations=citations)
