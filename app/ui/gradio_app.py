from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import requests


def _format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "- none"

    lines: list[str] = []
    for item in citations:
        index = item.get("index", "?")
        source_file = item.get("source_file", "")
        source_location = item.get("source_location", "")
        score = item.get("score", 0.0)
        quote = item.get("quote_excerpt", "")
        lines.append(
            f"- [C{index}] {source_file} | {source_location} | score={float(score):.3f}\n"
            f"  excerpt: {quote}"
        )
    return "\n".join(lines)


def _ingest_files(
    files: list[str] | None,
    api_base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embedder: str,
) -> str:
    if not files:
        return "No files selected."

    endpoint = f"{api_base_url.rstrip('/')}/ingest"
    multipart_files = []
    opened_files = []
    try:
        for fp in files:
            p = Path(fp)
            f = p.open("rb")
            opened_files.append(f)
            content_type = (
                "application/pdf"
                if p.suffix.lower() == ".pdf"
                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            multipart_files.append(("files", (p.name, f, content_type)))

        resp = requests.post(
            endpoint,
            files=multipart_files,
            data={
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap),
                "embedder": embedder,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return f"Ingest failed: HTTP {resp.status_code} - {resp.text}"

        data = resp.json()
        return (
            "Ingest success. "
            f"inserted_chunks={data.get('inserted_chunks', 0)}, "
            f"collection_count={data.get('collection_count', 0)}"
        )
    except Exception as exc:
        return f"Ingest failed: {exc}"
    finally:
        for f in opened_files:
            f.close()


def _chat(
    question: str,
    api_base_url: str,
    top_k: int,
    embedder: str,
    llm: str,
    model: str,
    retrieval_mode: str,
    alpha: float,
    query_rewrite: bool,
    rewrite_mode: str,
    use_session_memory: bool,
    session_id: str,
    history_turn_limit: int,
) -> tuple[str, str]:
    if not question.strip():
        return "", "- none"

    endpoint = f"{api_base_url.rstrip('/')}/chat"
    payload = {
        "question": question,
        "top_k": top_k,
        "embedder": embedder,
        "llm": llm,
        "model": model,
        "retrieval_mode": retrieval_mode,
        "alpha": alpha,
        "query_rewrite": query_rewrite,
        "rewrite_mode": rewrite_mode,
        "use_session_memory": use_session_memory,
        "session_id": session_id,
        "history_turn_limit": history_turn_limit,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"Chat failed: HTTP {resp.status_code}", f"```\n{resp.text}\n```"

        data = resp.json()
        answer = str(data.get("answer", ""))
        citations = _format_citations(data.get("citations", []))
        return answer, citations
    except Exception as exc:
        return f"Chat failed: {exc}", "- none"


def create_demo(api_base_url: str = "http://127.0.0.1:8000") -> gr.Blocks:
    with gr.Blocks(title="RAG Document QA") as demo:
        gr.Markdown("## RAG Document QA Demo")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF/DOCX",
                    file_count="multiple",
                    type="filepath",
                )
                chunk_size = gr.Slider(100, 1200, value=500, step=50, label="chunk_size")
                chunk_overlap = gr.Slider(0, 300, value=50, step=10, label="chunk_overlap")
                embedder = gr.Dropdown(["hash", "bge"], value="hash", label="embedder")
                ingest_btn = gr.Button("Ingest")
                ingest_status = gr.Textbox(label="Ingest status", interactive=False)

            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", lines=3)
                top_k = gr.Slider(1, 20, value=5, step=1, label="top_k")
                llm = gr.Dropdown(["echo", "ollama"], value="echo", label="llm")
                model = gr.Textbox(value="qwen2:7b", label="model")
                retrieval_mode = gr.Dropdown(
                    ["dense", "hybrid", "hybrid_rerank"],
                    value="hybrid",
                    label="retrieval_mode",
                )
                alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="hybrid alpha")
                query_rewrite = gr.Checkbox(value=True, label="Enable query rewrite")
                rewrite_mode = gr.Dropdown(["rule", "llm"], value="rule", label="rewrite mode")
                use_session_memory = gr.Checkbox(value=True, label="Use session memory")
                session_id = gr.Textbox(value="demo-session", label="session_id")
                history_turn_limit = gr.Slider(1, 20, value=4, step=1, label="history_turn_limit")
                chat_btn = gr.Button("Ask")
                answer = gr.Textbox(label="Answer", lines=6, interactive=False)
                citations = gr.Markdown("- none")

        api_base = gr.State(api_base_url)

        ingest_btn.click(
            fn=_ingest_files,
            inputs=[file_input, api_base, chunk_size, chunk_overlap, embedder],
            outputs=[ingest_status],
        )

        chat_btn.click(
            fn=_chat,
            inputs=[
                question,
                api_base,
                top_k,
                embedder,
                llm,
                model,
                retrieval_mode,
                alpha,
                query_rewrite,
                rewrite_mode,
                use_session_memory,
                session_id,
                history_turn_limit,
            ],
            outputs=[answer, citations],
        )

    return demo
