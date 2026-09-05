from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.parser import ParsedChunk


@dataclass(slots=True)
class ChunkedDocument:
    chunk_id: str
    text: str
    source_file: str
    source_location: str


def split_parsed_chunks(
    parsed_chunks: list[ParsedChunk],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[ChunkedDocument]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )

    results: list[ChunkedDocument] = []
    for parsed_idx, parsed in enumerate(parsed_chunks):
        split_texts = splitter.split_text(parsed.text)
        for chunk_idx, text in enumerate(split_texts):
            cleaned_text = text.strip()
            if not cleaned_text:
                continue
            results.append(
                ChunkedDocument(
                    chunk_id=(
                        f"{parsed.source_file}:{parsed.location}:"
                        f"{parsed_idx}-{chunk_idx}"
                    ),
                    text=cleaned_text,
                    source_file=parsed.source_file,
                    source_location=parsed.location,
                )
            )
    return results
