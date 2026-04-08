"""Text chunker — split sections into overlapping chunks for embedding."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from parser import Section


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_name: str
    section_title: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def chunk_sections(
    sections: List[Section],
    source_name: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """Split sections into overlapping text chunks."""
    chunks: List[Chunk] = []
    global_idx = 0
    for section in sections:
        section_chunks = _chunk_text(
            text=section.text,
            source_name=source_name,
            section_title=section.title,
            chunk_size=chunk_size,
            overlap=overlap,
            start_index=global_idx,
        )
        chunks.extend(section_chunks)
        global_idx += len(section_chunks)
    return chunks


def _chunk_text(
    text: str,
    source_name: str,
    section_title: str,
    chunk_size: int,
    overlap: int,
    start_index: int,
) -> List[Chunk]:
    """Sentence-aware chunking with character-level overlap."""
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    chunk_idx = start_index

    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    chunk_id=f"{source_name}_{chunk_idx}",
                    text=chunk_text,
                    source_name=source_name,
                    section_title=section_title,
                    chunk_index=chunk_idx,
                    metadata={"chunk_id": f"{source_name}_{chunk_idx}", "source": source_name},
                )
            )
            chunk_idx += 1
            # Overlap: keep last overlap characters worth of sentences
            overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current = [overlap_text]
            current_len = len(overlap_text)
        current.append(sentence)
        current_len += sent_len + 1

    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{source_name}_{chunk_idx}",
                text=chunk_text,
                source_name=source_name,
                section_title=section_title,
                chunk_index=chunk_idx,
                metadata={"chunk_id": f"{source_name}_{chunk_idx}", "source": source_name},
            )
        )

    return chunks


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation."""
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]
