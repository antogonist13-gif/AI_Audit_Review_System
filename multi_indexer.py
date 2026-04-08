"""Multi-collection indexer — norms, typical violations, historical checklists."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import config
from bm25_cache import build_bm25_index
from chunker import chunk_sections
from loader import load_document
from parser import parse_sections

logger = logging.getLogger(__name__)

# ChromaDB ограничивает размер одного upsert-вызова ~5461 записями.
# Используем консервативное значение с запасом.
_CHROMA_BATCH_SIZE = 500

COLLECTION_NAMES = {
    "norms": "norms",
    "typical": "typical_violations",
    "historical": "historical_checklists",
}

REF_DB_PATHS = {
    "norms": config.NORMS_DB_PATH,
    "typical": config.TYPICAL_DB_PATH,
    "historical": config.HISTORICAL_DB_PATH,
}

_clients: dict = {}
_collections: dict = {}


def _ref_client(kind: str):
    """Get or create a ChromaDB PersistentClient for the given collection kind."""
    if kind not in _clients:
        try:
            import chromadb
            db_path = REF_DB_PATHS[kind]
            db_path.mkdir(parents=True, exist_ok=True)
            _clients[kind] = chromadb.PersistentClient(path=str(db_path))
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
    return _clients[kind]


def get_ref_collection(kind: str):
    """Get or create the ChromaDB collection for the given kind."""
    if kind not in _collections:
        client = _ref_client(kind)
        _collections[kind] = client.get_or_create_collection(
            name=COLLECTION_NAMES[kind],
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[kind]


def clear_ref_collection(kind: str) -> None:
    """Delete and recreate the collection, resetting it to zero documents."""
    client = _ref_client(kind)
    col_name = COLLECTION_NAMES[kind]
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    _collections.pop(kind, None)
    _collections[kind] = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Cleared collection %s", col_name)


def _embed_and_upsert(collection, chunks) -> None:
    """Embed a list of Chunk objects and upsert into the collection in batches."""
    if not chunks:
        return
    from embeddings import embed_texts
    total = len(chunks)
    for start in range(0, total, _CHROMA_BATCH_SIZE):
        batch = chunks[start : start + _CHROMA_BATCH_SIZE]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in batch],
        )
        logger.debug(
            "Upserted batch %d–%d / %d",
            start + 1, min(start + _CHROMA_BATCH_SIZE, total), total,
        )


def index_norms(file_paths: List[str]) -> None:
    """Index normative documents (PDF/DOCX) into the norms collection."""
    col = get_ref_collection("norms")
    all_texts: List[str] = []
    for path in file_paths:
        raw = load_document(path)
        sections = parse_sections(raw)
        chunks = chunk_sections(sections, source_name=Path(path).name)
        _embed_and_upsert(col, chunks)
        all_texts.extend(c.text for c in chunks)
        logger.info("Indexed norms from %s — %d chunks", Path(path).name, len(chunks))
    if all_texts:
        build_bm25_index("norms", all_texts)


def index_typical_violations(items: List[dict]) -> None:
    """Index typical violations from a list of dicts into the typical collection.

    Each dict should have at minimum: 'text', 'law_ref', optionally 'id'.
    """
    col = get_ref_collection("typical")
    from embeddings import embed_texts
    import uuid as _uuid

    texts = [item.get("text", "") for item in items]
    if not texts:
        return
    ids = [item.get("id", str(_uuid.uuid4())) for item in items]
    metadatas = [
        {
            "chunk_id": ids[i],
            "law": items[i].get("law_ref", ""),
            "source": "typical",
            **{k: v for k, v in items[i].items() if k not in ("text", "id", "law_ref")},
        }
        for i in range(len(items))
    ]
    for start in range(0, len(texts), _CHROMA_BATCH_SIZE):
        batch_texts = texts[start : start + _CHROMA_BATCH_SIZE]
        batch_ids = ids[start : start + _CHROMA_BATCH_SIZE]
        batch_meta = metadatas[start : start + _CHROMA_BATCH_SIZE]
        batch_embeddings = embed_texts(batch_texts)
        col.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
    build_bm25_index("typical", texts)
    logger.info("Indexed %d typical violations", len(items))


def index_historical_checklists(file_paths: List[str]) -> None:
    """Index historical checklists (PDF/DOCX) into the historical collection."""
    col = get_ref_collection("historical")
    all_texts: List[str] = []
    for path in file_paths:
        raw = load_document(path)
        sections = parse_sections(raw)
        chunks = chunk_sections(sections, source_name=Path(path).name)
        _embed_and_upsert(col, chunks)
        all_texts.extend(c.text for c in chunks)
        logger.info("Indexed historical checklist from %s — %d chunks", Path(path).name, len(chunks))
    if all_texts:
        build_bm25_index("historical", all_texts)
