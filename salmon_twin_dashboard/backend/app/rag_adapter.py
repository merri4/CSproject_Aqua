from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .config import dashboard_root, resolve_path, settings


FALLBACK_GUIDELINES = [
    "RAS salmon operation: if unionized ammonia/NH3 rises toward 0.05 mg/L, reduce or stop feeding and increase water exchange/biofilter checks.",
    "RAS salmon operation: if dissolved oxygen falls below 6.0 mg/L or drops quickly, increase oxygen injection and circulation/aeration immediately.",
    "RAS salmon operation: if nitrate exceeds 50 mg/L or keeps rising, start partial water exchange and inspect denitrification capacity.",
    "RAS salmon operation: keep pH near 6.8-7.2 where possible; high pH increases ammonia toxicity and requires pH correction plus ammonia mitigation.",
    "RAS salmon operation: rising CO2 indicates insufficient degassing or circulation; increase degassing/aeration and inspect flow paths.",
]


def _default_rag_root() -> Path:
    return (dashboard_root().parent / "rag").resolve()


def rag_root() -> Path | None:
    return resolve_path(settings.RAG_ROOT, "../../rag", str(_default_rag_root()), "/rag")


def chroma_dir() -> Path | None:
    root = rag_root()
    fallback = str(root / "chroma_db") if root else None
    return resolve_path(settings.RAG_DB_DIR, fallback)


def manuals_path() -> Path | None:
    root = rag_root()
    fallback = str(root / "manuals" / "documents.txt") if root else None
    return resolve_path(settings.RAG_MANUALS_PATH, fallback)


def _manual_context() -> str:
    parts = []
    path = manuals_path()
    if path and path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            parts.append(f"[manual references]\n{text}")
    parts.append("[built-in RAS control guidelines]\n" + "\n".join(f"- {line}" for line in FALLBACK_GUIDELINES))
    return "\n\n".join(parts)


async def _ollama_embedding(query: str) -> list[float]:
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(url, json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": query})
        res.raise_for_status()
        data = res.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama embedding response did not include an embedding list")
    return embedding


def _query_chroma(embedding: list[float]) -> list[dict[str, Any]]:
    try:
        import chromadb
    except Exception as exc:
        raise RuntimeError(f"chromadb is not installed: {exc}") from exc

    db_dir = chroma_dir()
    if db_dir is None:
        raise RuntimeError("RAG Chroma directory was not found")
    if not (db_dir / "chroma.sqlite3").exists():
        raise RuntimeError("RAG Chroma database is not built yet")

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(name=settings.RAG_COLLECTION)
    result = collection.query(query_embeddings=[embedding], n_results=settings.RAG_TOP_K)

    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents[0]):
        rows.append(
            {
                "document": doc,
                "metadata": metadatas[0][idx] if metadatas and metadatas[0] and idx < len(metadatas[0]) else {},
                "distance": distances[0][idx] if distances and distances[0] and idx < len(distances[0]) else None,
            }
        )
    return rows


async def local_rag_context(query: str) -> str:
    try:
        embedding = await _ollama_embedding(query)
        rows = _query_chroma(embedding)
        if rows:
            chunks = []
            for row in rows:
                source = row.get("metadata", {}).get("source") if isinstance(row.get("metadata"), dict) else None
                heading = f"[RAG source: {source or 'chroma'}]"
                chunks.append(f"{heading}\n{row['document']}")
            return "\n\n".join(chunks)
    except Exception as exc:
        return f"[local RAG fallback: {exc}]\n{_manual_context()}"

    return _manual_context()


def rag_status() -> dict[str, Any]:
    root = rag_root()
    db_dir = chroma_dir()
    manual = manuals_path()
    status: dict[str, Any] = {
        "endpoint": settings.RAG_ENDPOINT or None,
        "root": str(root) if root else None,
        "chroma_dir": str(db_dir) if db_dir else None,
        "manuals_path": str(manual) if manual else None,
        "collection": settings.RAG_COLLECTION,
    }
    if settings.RAG_ENDPOINT:
        status["mode"] = "http"
        status["ok"] = True
        return status

    status["mode"] = "local"
    try:
        if db_dir is None:
            raise RuntimeError("Chroma directory not found")
        if not (db_dir / "chroma.sqlite3").exists():
            raise RuntimeError("RAG Chroma database is not built yet")
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(name=settings.RAG_COLLECTION)
        status["count"] = collection.count()
        status["ok"] = True
    except Exception as exc:
        status["ok"] = bool(manual or FALLBACK_GUIDELINES)
        status["fallback"] = True
        status["warning"] = str(exc)
    return status
