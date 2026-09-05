from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation_modes import (
    ModeEvalResult,
    evaluate_retrieval_modes_with_topics,
    load_mode_eval_items,
)


def _parse_modes(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("modes must not be empty")
    return values


def _print_results(rows: list[ModeEvalResult]) -> None:
    print(
        "mode,total_queries,answer_hit_rate,citation_hit_rate,strict_citation_hit_rate,abstain_rate,"
        "hallucination_proxy_rate,avg_latency_ms"
    )
    for row in rows:
        print(
            f"{row.mode},{row.total_queries},{row.answer_hit_rate:.4f},"
            f"{row.citation_hit_rate:.4f},{row.strict_citation_hit_rate:.4f},{row.abstain_rate:.4f},"
            f"{row.hallucination_proxy_rate:.4f},{row.avg_latency_ms:.2f}"
        )


def _write_csv(rows: list[ModeEvalResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "total_queries",
                "answer_hit_rate",
                "citation_hit_rate",
                "strict_citation_hit_rate",
                "abstain_rate",
                "hallucination_proxy_rate",
                "avg_latency_ms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.mode,
                    row.total_queries,
                    f"{row.answer_hit_rate:.4f}",
                    f"{row.citation_hit_rate:.4f}",
                    f"{row.strict_citation_hit_rate:.4f}",
                    f"{row.abstain_rate:.4f}",
                    f"{row.hallucination_proxy_rate:.4f}",
                    f"{row.avg_latency_ms:.2f}",
                ]
            )


def _write_topic_json(topic_summary: dict[str, dict[str, dict[str, float]]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(topic_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate dense/hybrid/hybrid_rerank baseline")
    parser.add_argument("--docs", nargs="+", required=True, help="Docx/PDF files")
    parser.add_argument("--eval-json", required=True, help="Evaluation dataset JSON")
    parser.add_argument("--store-dir", default="vectorstore/mode_eval")
    parser.add_argument("--collection", default="mode_eval_docs")
    parser.add_argument("--embedder", choices=["hash", "bge"], default="hash")
    parser.add_argument("--llm", choices=["echo", "ollama"], default="echo")
    parser.add_argument("--model", default="qwen2:7b")
    parser.add_argument("--modes", default="dense,hybrid,hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--query-rewrite", action="store_true")
    parser.add_argument("--rewrite-mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--max-items", type=int, default=0, help="Use first N eval items (0 means all)")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds")
    parser.add_argument("--out-csv", default="vectorstore/mode_eval/results_modes.csv")
    parser.add_argument("--out-topic-json", default="vectorstore/mode_eval/results_modes_topics.json")
    args = parser.parse_args()

    doc_files = [Path(x) for x in args.docs]
    eval_items = load_mode_eval_items(args.eval_json)
    if args.max_items > 0:
        eval_items = eval_items[: args.max_items]
    modes = _parse_modes(args.modes)

    store_dir = Path(args.store_dir)
    if store_dir.exists():
        shutil.rmtree(store_dir)

    rows, topic_summary = evaluate_retrieval_modes_with_topics(
        doc_files=doc_files,
        eval_items=eval_items,
        persist_directory=store_dir,
        collection_name=args.collection,
        embedder_name=args.embedder,
        llm_name=args.llm,
        llm_model=args.model,
        modes=modes,
        top_k=args.top_k,
        alpha=args.alpha,
        enable_query_rewrite=args.query_rewrite,
        rewrite_mode=args.rewrite_mode,
        ollama_timeout_seconds=args.ollama_timeout,
    )

    _print_results(rows)
    _write_csv(rows, Path(args.out_csv))
    _write_topic_json(topic_summary, Path(args.out_topic_json))
    print(f"Saved CSV to: {args.out_csv}")
    print(f"Saved topic summary JSON to: {args.out_topic_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
