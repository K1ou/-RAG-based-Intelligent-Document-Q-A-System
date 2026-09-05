from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation import EvalResult, evaluate_chunk_params, load_eval_items
from app.rag.pipeline import build_embedder


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _print_results(rows: list[EvalResult]) -> None:
    print("chunk_size,chunk_overlap,total_queries,hit_at_k,avg_latency_ms")
    for row in rows:
        print(
            f"{row.chunk_size},{row.chunk_overlap},{row.total_queries},"
            f"{row.hit_at_k:.4f},{row.avg_latency_ms:.2f}"
        )


def _write_csv(rows: list[EvalResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_size", "chunk_overlap", "total_queries", "hit_at_k", "avg_latency_ms"])
        for row in rows:
            writer.writerow(
                [
                    row.chunk_size,
                    row.chunk_overlap,
                    row.total_queries,
                    f"{row.hit_at_k:.4f}",
                    f"{row.avg_latency_ms:.2f}",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate chunk_size/overlap combinations")
    parser.add_argument("--docs", nargs="+", required=True, help="Docx/PDF files")
    parser.add_argument("--eval-json", required=True, help="Evaluation dataset JSON")
    parser.add_argument("--chunk-sizes", default="300,500,800")
    parser.add_argument("--overlaps", default="0,50,100")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", choices=["hash", "bge"], default="hash")
    parser.add_argument("--workspace-store-dir", default="vectorstore/chunk_param_eval")
    parser.add_argument("--out-csv", default="vectorstore/chunk_param_eval/results.csv")
    args = parser.parse_args()

    doc_files = [Path(x) for x in args.docs]
    eval_items = load_eval_items(args.eval_json)
    chunk_sizes = _parse_int_list(args.chunk_sizes)
    overlaps = _parse_int_list(args.overlaps)

    embedder = build_embedder(args.embedder)
    workspace_store_dir = Path(args.workspace_store_dir)
    if workspace_store_dir.exists():
        shutil.rmtree(workspace_store_dir)
    workspace_store_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    run_index = 0
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            run_index += 1
            run_store = workspace_store_dir / f"run_{run_index}_{chunk_size}_{overlap}"
            collection = f"eval_{run_index}_{chunk_size}_{overlap}"
            result = evaluate_chunk_params(
                doc_files=doc_files,
                eval_items=eval_items,
                embedder=embedder,
                store_dir=run_store,
                collection_name=collection,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                top_k=args.top_k,
            )
            results.append(result)

    _print_results(results)
    _write_csv(results, Path(args.out_csv))
    print(f"Saved CSV to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
