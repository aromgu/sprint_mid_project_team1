"""Incrementally add one uploaded PDF to the Main Advanced corpus and Chroma."""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.main_rag.chunking.advanced_chunking import chunk_advanced_corpus
from src.main_rag.embeddings.build_advanced_index import normalize_advanced_metadata
from src.main_rag.loader.load_documents import SourceDocument
from src.main_rag.preprocessing.prepare_advanced import prepare_advanced_document
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.settings import MainRAGSettings, load_settings


_INDEX_LOCK = threading.Lock()


def _replace_jsonl(path: Path, rows: Iterable[dict[str, Any]], source_id: str) -> None:
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    merged = [row for row in existing if str(row.get("source_id")) != source_id]
    merged.extend(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in merged), encoding="utf-8")
    temporary.replace(path)


def _replace_gzip_chunks(path: Path, rows: list[dict[str, Any]], document_id: str) -> int:
    existing = []
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            existing = [json.loads(line) for line in stream if line.strip()]
    merged = [row for row in existing if str(row.get("document_id")) != document_id]
    merged.extend(rows)
    merged.sort(key=lambda row: (str(row.get("source_id")), int(row.get("chunk_order") or 0)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return len(merged)


def _write_live_status(settings: MainRAGSettings, payload: dict[str, Any]) -> None:
    reports_value = settings.values.get("paths", {}).get("reports")
    if not reports_value:
        return
    path = settings.path("reports") / "live_index_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def index_uploaded_pdf(
    path: Path,
    *,
    document_id: str,
    title: str,
    organization: str,
    settings: MainRAGSettings | None = None,
    retriever: AdvancedRetriever | None = None,
) -> dict[str, Any]:
    """Run the ported Advanced parser/chunker and upsert only this document."""
    selected = settings or load_settings()
    content = path.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    source_id = source_sha256[:16]
    manifest = {
        "schema_version": "main_advanced_manifest_v1",
        "source_id": source_id,
        "document_id": document_id,
        "source_sha256": source_sha256,
        "source_relative_path": path.name,
        "source_filename": path.name,
        "all_source_filenames": [path.name],
        "filename_aliases": [],
        "duplicate_alias_count": 0,
        "source_file_size_bytes": len(content),
        "file_type": "pdf",
        "project_name": title,
        "issuer": organization,
        "metadata_validation_status": "user_upload",
        "canonical_selection_reason": "user_upload",
    }
    source = SourceDocument(
        source_id=source_id,
        # The preprocessing integrity contract is content-addressed. The
        # external upload_* ID is restored from manifest in build_advanced_result.
        document_id=source_id,
        source_path=path.resolve(),
        source_relative_path=path.name,
        source_filename=path.name,
        source_sha256=source_sha256,
        file_type="pdf",
        source_file_size_bytes=len(content),
        duplicate_group_size=1,
        is_default_canonical=True,
        default_canonical_filename=path.name,
        all_source_filenames=(path.name,),
        canonical_selection_source="user_upload",
        canonical_selection_reason="user_upload",
    )
    advanced = prepare_advanced_document(source, manifest)
    chunked = list(chunk_advanced_corpus([advanced.document], list(advanced.blocks)).chunks)
    if not chunked:
        raise RuntimeError("업로드 PDF에서 Advanced chunk가 생성되지 않았습니다")

    model = str(selected.get("index", "embedding_model", "text-embedding-3-small"))
    create_date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadatas = [normalize_advanced_metadata(row, embedding_model=model, create_date=create_date) for row in chunked]
    with _INDEX_LOCK:
        active_retriever = retriever or AdvancedRetriever(selected)
        try:
            store = active_retriever._initialize()
        except Exception as exc:
            # The query retriever intentionally rejects a missing/empty index.
            # Upload indexing, however, is also the collection creation path.
            if "인덱스가 없습니다" not in str(exc) and "collection이 비어" not in str(exc):
                raise
            load_dotenv()
            active_retriever.persist_directory.mkdir(parents=True, exist_ok=True)
            embeddings = OpenAIEmbeddings(
                model=active_retriever.embedding_model,
                dimensions=active_retriever.embedding_dimension,
            )
            store = Chroma(
                collection_name=active_retriever.collection_name,
                embedding_function=embeddings,
                persist_directory=str(active_retriever.persist_directory),
            )
            active_retriever._vectorstore = store
        store._collection.delete(where={"document_id": document_id})
        store.add_texts(
            texts=[str(row["embedding_text"]) for row in chunked],
            metadatas=metadatas,
            ids=[str(row["chunk_id"]) for row in chunked],
        )
        preprocessing = selected.path("preprocessing_dir")
        _replace_jsonl(preprocessing / "documents_advanced_v1.jsonl", [advanced.document], source_id)
        _replace_jsonl(preprocessing / "blocks_advanced_v1.jsonl", advanced.blocks, source_id)
        _replace_jsonl(preprocessing / "tables_advanced_v1.jsonl", advanced.tables, source_id)
        _replace_jsonl(preprocessing / "images_advanced_v1.jsonl", advanced.images, source_id)
        total_chunks = _replace_gzip_chunks(selected.path("chunks"), chunked, document_id)
        manifest_path = selected.path("advanced_manifest").with_name("uploads.jsonl")
        _replace_jsonl(manifest_path, [manifest], source_id)
        collection_count = store._collection.count()
        upload_count = len([
            line for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
        _write_live_status(selected, {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_document_id": document_id,
            "collection_name": active_retriever.collection_name,
            "embedding_model": active_retriever.embedding_model,
            "collection_count": collection_count,
            "corpus_chunk_count": total_chunks,
            "upload_document_count": upload_count,
            "status": "ready" if collection_count == total_chunks else "stale_index",
        })
    return {
        "document_id": document_id,
        "source_id": source_id,
        "advanced_chunks": len(chunked),
        "advanced_collection_count": collection_count,
        "advanced_corpus_chunk_count": total_chunks,
        "status": "advanced_indexed",
    }
