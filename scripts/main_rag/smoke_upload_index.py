"""Build and query an isolated Advanced index from a newly created PDF."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fitz

from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.settings import MainRAGSettings
from src.main_rag.upload_indexer import index_uploaded_pdf


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="main-advanced-upload-") as directory:
        root = Path(directory)
        pdf_path = root / "uploaded-demo.pdf"
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "Uploaded RFP Demo\nSubmission deadline is 2026-08-31 17:00.\n"
            "The bidder must submit one signed original proposal.",
            fontsize=12,
        )
        pdf.save(pdf_path)
        pdf.close()
        values = {
            "paths": {
                "advanced_manifest": "manifest/documents.jsonl",
                "preprocessing_dir": "preprocessed",
                "chunks": "chunks/chunks.jsonl.gz",
                "chroma": "chroma",
            },
            "index": {
                "collection_name": "upload_smoke",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
            },
            "retrieval": {"top_k": 5},
            "generation": {"max_context_chars": 7000, "max_docs": 6},
        }
        settings = MainRAGSettings(root / "config.yaml", root, values)
        retriever = AdvancedRetriever(settings)
        report = index_uploaded_pdf(
            pdf_path,
            document_id="upload_smoke_pdf",
            title="Uploaded RFP Demo",
            organization="Smoke Test",
            settings=settings,
            retriever=retriever,
        )
        results = retriever.search_documents(
            "submission deadline signed original proposal",
            document_id="upload_smoke_pdf",
            top_k=3,
        )
        if not results or results[0]["metadata"].get("document_id") != "upload_smoke_pdf":
            raise RuntimeError("업로드 문서가 격리된 Advanced Chroma에서 검색되지 않았습니다")
        print(json.dumps({
            **report,
            "retrieved_chunk_ids": [row["chunk_id"] for row in results],
            "top_page": results[0]["page"],
            "temporary_workspace_removed_on_exit": True,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
