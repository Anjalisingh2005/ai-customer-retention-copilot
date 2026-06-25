"""Build / rebuild the FAISS index from the knowledge_base/ markdown."""
from __future__ import annotations

from src.rag.retriever import Retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("Building FAISS knowledge-base index…")
    Retriever().build()
    log.info("Done.")


if __name__ == "__main__":
    main()
