"""Rebuild RAG Index Script: Generates and indexes RAG documents for schema, columns, entities, and context."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database.connection import db_manager
from app.rag.document_builder import rag_document_builder
from app.rag.vector_store import database_vector_store
from app.rag.hybrid_retriever import hybrid_retriever
from app.utils.logger import logger


def rebuild_rag():
    print("==================================================")
    print("Universal Dataset AI - Rebuild RAG Index")
    print("==================================================")

    db_manager.init_database()
    print("Building structured RAG documents from catalog and tables...")

    docs = rag_document_builder.build_all_documents()
    print(f"Generated {len(docs)} structured documents.")

    print("Embedding and persisting documents to Vector Store...")
    count = database_vector_store.add_documents(docs)
    print(f"Successfully indexed {count} documents into RAG vector repository.")

    # Verification query
    test_query = "which state has the highest population"
    print(f"\nRunning test hybrid retrieval for: '{test_query}'...")
    results = hybrid_retriever.retrieve(test_query, k=3)

    print(f"Retrieved {len(results)} relevant contexts:")
    for i, res in enumerate(results, 1):
        content_preview = res['content'][:100].replace('\n', ' ')
        print(f"  {i}. [Score: {res['score']}] Type: {res['document_type']} | {content_preview}...")

    if count > 0 and len(results) > 0:
        print("\nSUCCESS: RAG knowledge index rebuilt and verified.")
        return 0
    else:
        print("\nERROR: RAG indexing failed.")
        return 1


if __name__ == "__main__":
    sys.exit(rebuild_rag())
