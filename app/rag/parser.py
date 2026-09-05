from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf


@dataclass(slots=True)
class ParsedChunk:
    text: str
    source_file: str
    location: str


def parse_pdf(file_path: str | Path) -> list[ParsedChunk]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    chunks: list[ParsedChunk] = []
    with pymupdf.open(path) as pdf_doc:
        for idx, page in enumerate(pdf_doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            chunks.append(
                ParsedChunk(
                    text=text,
                    source_file=path.name,
                    location=f"page:{idx}",
                )
            )
    return chunks


def parse_docx(file_path: str | Path) -> list[ParsedChunk]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Word file not found: {path}")

    document = docx.Document(path)
    chunks: list[ParsedChunk] = []
    para_index = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        para_index += 1
        chunks.append(
            ParsedChunk(
                text=text,
                source_file=path.name,
                location=f"paragraph:{para_index}",
            )
        )
    return chunks


def parse_document(file_path: str | Path) -> list[ParsedChunk]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")
