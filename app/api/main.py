from __future__ import annotations

from pathlib import Path
from typing import Literal
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.conversation_memory import InMemoryConversationStore
from app.rag.pipeline import EchoLLMClient, RAGPipeline


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    embedder: Literal["hash", "bge"] = "hash"
    llm: Literal["echo", "ollama"] = "echo"
    model: str = "qwen2:7b"
    retrieval_mode: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    query_rewrite: bool = False
    rewrite_mode: Literal["rule", "llm"] = "rule"
    use_session_memory: bool = False
    session_id: str | None = None
    history_turn_limit: int = Field(default=4, ge=1, le=20)


class CitationResponse(BaseModel):
    index: int
    chunk_id: str
    source_file: str
    source_location: str
    score: float
    quote_excerpt: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IngestResponse(BaseModel):
    inserted_chunks: int
    collection_count: int
    files: list[str]


def create_app(store_dir: str = "vectorstore/chroma_api", collection: str = "rag_docs") -> FastAPI:
    app = FastAPI(title="RAG Document QA API", version="0.1.0")
    conversation_store = InMemoryConversationStore()

    def _build_pipeline(embedder: str, llm_mode: str, model: str) -> RAGPipeline:
        llm_client = EchoLLMClient() if llm_mode == "echo" else None
        return RAGPipeline(
            persist_directory=store_dir,
            collection_name=collection,
            embedder_name=embedder,
            llm_client=llm_client,
            llm_model=model,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        files: list[UploadFile] = File(...),
        chunk_size: int = Form(500),
        chunk_overlap: int = Form(50),
        embedder: Literal["hash", "bge"] = Form("hash"),
    ) -> IngestResponse:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        upload_dir = Path("data") / "uploads_tmp" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        accepted_names: list[str] = []
        for f in files:
            name = f.filename or ""
            suffix = Path(name).suffix.lower()
            if suffix not in {".pdf", ".docx"}:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {name}")

            content = await f.read()
            target = upload_dir / name
            target.write_bytes(content)
            saved_paths.append(target)
            accepted_names.append(name)

        pipeline = _build_pipeline(embedder=embedder, llm_mode="echo", model="qwen2:7b")
        inserted = pipeline.ingest_files(
            file_paths=saved_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return IngestResponse(
            inserted_chunks=inserted,
            collection_count=pipeline.store.count(),
            files=accepted_names,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        conversation_turns = []
        if payload.use_session_memory and payload.session_id:
            conversation_turns = conversation_store.get_recent_turns(
                payload.session_id,
                limit=payload.history_turn_limit,
            )

        pipeline = _build_pipeline(
            embedder=payload.embedder,
            llm_mode=payload.llm,
            model=payload.model,
        )

        result = pipeline.ask(
            question=payload.question,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            alpha=payload.alpha,
            enable_query_rewrite=payload.query_rewrite,
            rewrite_mode=payload.rewrite_mode,
            conversation_turns=conversation_turns,
        )

        if payload.use_session_memory and payload.session_id:
            conversation_store.append_turn(
                payload.session_id,
                payload.question,
                result.answer,
            )

        return ChatResponse(
            answer=result.answer,
            citations=[
                CitationResponse(
                    index=item.index,
                    chunk_id=item.chunk_id,
                    source_file=item.source_file,
                    source_location=item.source_location,
                    score=item.score,
                    quote_excerpt=item.quote_excerpt,
                )
                for item in result.citations
            ],
        )

    return app


app = create_app()
