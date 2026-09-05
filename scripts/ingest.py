from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX files into Chroma vector store")
    parser.add_argument("--files", nargs="+", required=True, help="List of PDF/DOCX paths")
    parser.add_argument("--store-dir", default="vectorstore/chroma", help="Persist directory")
    parser.add_argument("--collection", default="rag_docs", help="Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--embedder", choices=["hash", "bge"], default="hash")
    args = parser.parse_args()

    file_paths = [Path(p) for p in args.files]
    pipeline = RAGPipeline(
        persist_directory=args.store_dir,
        collection_name=args.collection,
        embedder_name=args.embedder,
    )

    inserted = pipeline.ingest_files(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Ingest completed. Inserted chunks: {inserted}")
    print(f"Collection count: {pipeline.store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
