"""FAISS-backed retriever over the retention knowledge base.

Embedding model: a local SentenceTransformer (no API call, no rate limits,
free at inference). For production at scale, swap `_Embedder` for an
embedding API — the interface is the same `.encode(list[str]) -> np.ndarray`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.config import settings
from src.rag.knowledge_base import Chunk, load_chunks
from src.utils.logger import get_logger

log = get_logger(__name__)

_INDEX_FILENAME = "kb.index"
_META_FILENAME = "kb_meta.json"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedDoc:
    text: str
    source: str
    chunk_id: int
    score: float


class _Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        log.info("Loading sentence-transformer %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.astype(np.float32)


class Retriever:
    def __init__(self, index_dir: Path | None = None, embedder: _Embedder | None = None):
        self.index_dir = index_dir or settings.faiss_index_dir
        self.embedder = embedder or _Embedder()
        self.index = None
        self.meta: List[dict] = []

    def build(self, chunks: List[Chunk] | None = None) -> None:
        import faiss

        chunks = chunks or load_chunks()
        if not chunks:
            raise ValueError("No knowledge-base chunks found to index.")
        log.info("Embedding %d chunks…", len(chunks))
        embs = self.embedder.encode([c.text for c in chunks])

        index = faiss.IndexFlatIP(self.embedder.dim)  # cosine via normalized vectors.
        index.add(embs)
        self.index = index
        self.meta = [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in chunks]
        self._persist()

    def _persist(self) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / _INDEX_FILENAME))
        (self.index_dir / _META_FILENAME).write_text(json.dumps(self.meta), encoding="utf-8")
        log.info("Persisted FAISS index → %s", self.index_dir)

    def load(self) -> "Retriever":
        import faiss

        idx_path = self.index_dir / _INDEX_FILENAME
        meta_path = self.index_dir / _META_FILENAME
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index missing under {self.index_dir}. Run scripts/ingest_knowledge.py first."
            )
        self.index = faiss.read_index(str(idx_path))
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info("Loaded FAISS index (%d vectors) from %s", self.index.ntotal, self.index_dir)
        return self

    def search(self, query: str, k: int = 4) -> List[RetrievedDoc]:
        if self.index is None:
            self.load()
        q = self.embedder.encode([query])
        scores, ids = self.index.search(q, k)
        results: List[RetrievedDoc] = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            m = self.meta[i]
            results.append(
                RetrievedDoc(
                    text=m["text"],
                    source=m["source"],
                    chunk_id=m["chunk_id"],
                    score=float(score),
                )
            )
        return results


_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazily load a singleton so we don't reload the FAISS index per request."""
    global _retriever_singleton
    if _retriever_singleton is None:
        r = Retriever()
        r.load()
        _retriever_singleton = r
    return _retriever_singleton
