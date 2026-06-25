"""Build the FAISS index over knowledge_base/*.md."""
import _bootstrap  # noqa: F401

from src.rag.ingest import main

if __name__ == "__main__":
    main()
