from __future__ import annotations

import re
from typing import Protocol


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        ...


class RewriterLLMClient(Protocol):
    def generate(self, prompt: str, model: str) -> str:
        ...


def build_rewrite_prompt(query: str) -> str:
    return (
        "你是检索查询改写助手。请将用户问题改写为更适合检索的短句。\n"
        "要求:\n"
        "1) 保留用户意图，不要新增事实。\n"
        "2) 补足关键词（实体、主题、动作）。\n"
        "3) 输出单行改写结果，不要解释。\n\n"
        f"原问题: {query}\n"
    )


class RuleBasedQueryRewriter:
    def rewrite(self, query: str) -> str:
        q = query.strip()
        if not q:
            raise ValueError("query must not be empty")

        q = re.sub(r"\s+", " ", q)
        replacements = {
            "怎么": "如何",
            "咋": "如何",
            "是啥": "是什么",
            "有啥": "有什么",
            "哪儿": "哪里",
            "哪裡": "哪里",
        }
        for src, dst in replacements.items():
            q = q.replace(src, dst)

        if "这个系统" in q:
            q = q.replace("这个系统", "RAG文档问答系统")
        if "它" in q and "RAG" not in q:
            q = q.replace("它", "RAG文档问答系统")

        q = q.rstrip("？?。！!")
        return f"{q}，请聚焦实现流程、约束条件与关键模块"


class LLMQueryRewriter:
    def __init__(self, llm_client: RewriterLLMClient, model: str = "qwen2:7b") -> None:
        self.llm_client = llm_client
        self.model = model

    def rewrite(self, query: str) -> str:
        q = query.strip()
        if not q:
            raise ValueError("query must not be empty")

        prompt = build_rewrite_prompt(q)
        rewritten = self.llm_client.generate(prompt=prompt, model=self.model).strip()
        if not rewritten:
            return q
        return rewritten
