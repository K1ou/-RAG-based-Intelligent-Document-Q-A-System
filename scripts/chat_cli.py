from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.conversation_memory import ConversationTurn, InMemoryConversationStore
from app.rag.pipeline import EchoLLMClient, RAGPipeline


def _load_store(path: Path) -> InMemoryConversationStore:
    store = InMemoryConversationStore()
    if not path.exists():
        return store

    payload = json.loads(path.read_text(encoding="utf-8"))
    store.load_dict(payload)
    return store


def _save_store(path: Path, store: InMemoryConversationStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = store.to_dict()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions from local RAG store")
    parser.add_argument("question", help="Question text")
    parser.add_argument("--store-dir", default="vectorstore/chroma", help="Persist directory")
    parser.add_argument("--collection", default="rag_docs", help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", choices=["hash", "bge"], default="hash")
    parser.add_argument("--llm", choices=["ollama", "echo"], default="echo")
    parser.add_argument("--model", default="qwen2:7b")
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid", "hybrid_rerank"], default="dense")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight for dense score")
    parser.add_argument("--query-rewrite", action="store_true", help="Enable query rewriting before retrieval")
    parser.add_argument("--rewrite-mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--use-session-memory", action="store_true")
    parser.add_argument("--session-id", default="default-session")
    parser.add_argument("--history-turn-limit", type=int, default=4)
    parser.add_argument("--history-file", default="data/session_history.json")
    args = parser.parse_args()

    llm_client = EchoLLMClient() if args.llm == "echo" else None
    pipeline = RAGPipeline(
        persist_directory=args.store_dir,
        collection_name=args.collection,
        embedder_name=args.embedder,
        llm_client=llm_client,
        llm_model=args.model,
    )

    history_store = _load_store(Path(args.history_file)) if args.use_session_memory else InMemoryConversationStore()
    history_turns: list[ConversationTurn] = []
    if args.use_session_memory:
        history_turns = history_store.get_recent_turns(args.session_id, limit=args.history_turn_limit)

    result = pipeline.ask(
        question=args.question,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        enable_query_rewrite=args.query_rewrite,
        rewrite_mode=args.rewrite_mode,
        conversation_turns=history_turns,
    )

    if args.use_session_memory:
        history_store.append_turn(args.session_id, args.question, result.answer)
        _save_store(Path(args.history_file), history_store)

    print("Answer:")
    print(result.answer)
    print("\nCitations:")
    if not result.citations:
        print("- none")
    for citation in result.citations:
        print(f"- {citation.source_file} | {citation.source_location} | {citation.chunk_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
