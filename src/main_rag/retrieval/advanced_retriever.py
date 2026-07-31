"""Lazy dense retriever for the isolated Main Advanced Chroma collection."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.main_rag.settings import MainRAGSettings, load_settings
from src.main_rag.chunking.advanced_chunking import KiwiBm25Tokenizer, _tokenize_bm25
from src.main_rag.embeddings.build_advanced_index import load_bm25_artifact


class AdvancedRetrieverError(RuntimeError):
    """A recoverable Advanced index or embedding configuration failure."""


@dataclass(frozen=True)
class RetrievalDiagnostics:
    mode: str
    dense_candidates: int
    bm25_candidates: int
    fused_candidates: int


class AdvancedRetriever:
    def __init__(self, settings: MainRAGSettings | None = None, *, mode: str | None = None) -> None:
        self.settings = settings or load_settings()
        self.persist_directory: Path = self.settings.path("chroma")
        self.collection_name = str(
            self.settings.get("index", "collection_name", "ai11_policy_advanced_v2")
        )
        self.embedding_model = str(
            self.settings.get("index", "embedding_model", "text-embedding-3-small")
        )
        self.embedding_dimension = int(
            self.settings.get("index", "embedding_dimension", 1536)
        )
        self.default_top_k = int(self.settings.get("retrieval", "top_k", 5))
        self.mode = mode or str(self.settings.get("retrieval", "mode", "dense"))
        self.bm25_path = self.settings.path("bm25")
        self.dense_candidate_k = int(self.settings.get("retrieval", "dense_candidate_k", 12))
        self.bm25_candidate_k = int(self.settings.get("retrieval", "bm25_candidate_k", 12))
        self.rrf_k = int(self.settings.get("retrieval", "rrf_k", 60))
        self.dense_weight = float(self.settings.get("retrieval", "dense_weight", 0.3))
        self.bm25_weight = float(self.settings.get("retrieval", "bm25_weight", 0.7))
        self.require_document_id = bool(self.settings.get("retrieval", "require_document_id", True))
        self._vectorstore: Chroma | None = None
        self._bm25_payload: dict[str, Any] | None = None
        self._bm25_rows: dict[str, dict[str, Any]] | None = None
        self._kiwi: KiwiBm25Tokenizer | None = None
        self._initialize_lock = threading.RLock()

    def _initialize(self) -> Chroma:
        if self._vectorstore is not None:
            return self._vectorstore
        with self._initialize_lock:
            if self._vectorstore is not None:
                return self._vectorstore
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise AdvancedRetrieverError("OPENAI_API_KEY가 설정되지 않았습니다")
            if not self.persist_directory.is_dir():
                raise AdvancedRetrieverError(
                    f"Advanced Chroma 인덱스가 없습니다: {self.persist_directory}"
                )
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                dimensions=self.embedding_dimension,
            )
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )
            if store._collection.count() == 0:
                raise AdvancedRetrieverError(
                    f"Advanced Chroma collection이 비어 있습니다: {self.collection_name}"
                )
            self._vectorstore = store
            return store

    def _initialize_bm25(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if self._bm25_payload is not None and self._bm25_rows is not None:
            return self._bm25_payload, self._bm25_rows
        with self._initialize_lock:
            if self._bm25_payload is None:
                try:
                    self._bm25_payload = load_bm25_artifact(self.bm25_path)
                except Exception as exc:
                    raise AdvancedRetrieverError(f"Advanced BM25 인덱스 로딩 실패: {exc}") from exc
                self._kiwi = KiwiBm25Tokenizer()
            if self._bm25_rows is None:
                store = self._initialize()
                ids = [str(value) for value in self._bm25_payload["chunk_ids"]]
                fetched = store.get(ids=ids, include=["documents", "metadatas"])
                self._bm25_rows = {
                    str(chunk_id): {"text": text, "metadata": metadata or {}}
                    for chunk_id, text, metadata in zip(
                        fetched.get("ids", []), fetched.get("documents", []), fetched.get("metadatas", [])
                    )
                }
        return self._bm25_payload, self._bm25_rows

    @staticmethod
    def _result(document: Any, score: float, **diagnostics: Any) -> dict[str, Any]:
        metadata = dict(document.metadata or {})
        chunk_id = str(metadata.get("chunk_id") or document.id or "")
        page = metadata.get("page") if metadata.get("page") is not None else metadata.get("page_start")
        return {
            "id": chunk_id, "chunk_id": chunk_id, "text": document.page_content,
            "file_nm": metadata.get("source_filename") or metadata.get("file_nm"),
            "page": page, "score": float(score), "metadata": metadata, **diagnostics,
        }

    def _dense_search(self, query: str, *, k: int, document_id: str) -> list[dict[str, Any]]:
        docs = self._initialize().max_marginal_relevance_search(
            query, k=k, fetch_k=max(k * 3, 20), filter={"document_id": document_id},
        )
        return [self._result(doc, 1.0 / rank, dense_rank=rank) for rank, doc in enumerate(docs, 1)]

    def _bm25_search(self, query: str, *, k: int, document_id: str) -> list[dict[str, Any]]:
        payload, rows = self._initialize_bm25()
        tokens = _tokenize_bm25(self._kiwi, query)
        scores = payload["index"].get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
        results = []
        for index, score in ranked:
            chunk_id = str(payload["chunk_ids"][index])
            row = rows.get(chunk_id)
            if not row or str(row["metadata"].get("document_id")) != document_id:
                continue
            metadata = row["metadata"]
            page = metadata.get("page") if metadata.get("page") is not None else metadata.get("page_start")
            results.append({
                "id": chunk_id, "chunk_id": chunk_id, "text": row["text"],
                "file_nm": metadata.get("source_filename") or metadata.get("file_nm"),
                "page": page, "score": float(score), "metadata": metadata,
                "bm25_rank": len(results) + 1, "bm25_score": float(score),
            })
            if len(results) >= k:
                break
        return results

    def search_documents(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if self.require_document_id and not document_id:
            raise ValueError("현재 문서 검색에는 document_id가 필요합니다")
        limit = self.default_top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k는 1 이상이어야 합니다")
        if self.mode == "hybrid_rrf":
            dense = self._dense_search(query, k=max(limit, self.dense_candidate_k), document_id=str(document_id))
            bm25 = self._bm25_search(query, k=max(limit, self.bm25_candidate_k), document_id=str(document_id))
            fused: dict[str, dict[str, Any]] = {}
            for source, weight, rank_name in ((dense, self.dense_weight, "dense_rank"), (bm25, self.bm25_weight, "bm25_rank")):
                for rank, row in enumerate(source, 1):
                    entry = fused.setdefault(row["chunk_id"], {**row, "rrf_score": 0.0})
                    entry.update({key: value for key, value in row.items() if key not in {"score"}})
                    entry[rank_name] = rank
                    entry["rrf_score"] += weight / (self.rrf_k + rank)
            ranked = sorted(fused.values(), key=lambda row: (-row["rrf_score"], row["chunk_id"]))[:limit]
            for rank, row in enumerate(ranked, 1):
                row.update(score=float(row["rrf_score"]), score_type="rrf", retriever="hybrid_rrf", rank=rank)
            return ranked

        metadata_filter = {"document_id": document_id} if document_id else None
        try:
            matches = self._initialize().similarity_search_with_relevance_scores(
                query,
                k=limit,
                filter=metadata_filter,
            )
        except AdvancedRetrieverError:
            raise
        except Exception as exc:
            raise AdvancedRetrieverError(f"Advanced dense 검색 실패: {exc}") from exc

        results: list[dict[str, Any]] = []
        for document, score in matches:
            metadata = dict(document.metadata or {})
            chunk_id = str(metadata.get("chunk_id") or document.id or "")
            page = metadata.get("page")
            if page is None:
                page = metadata.get("page_start")
            results.append({"id": chunk_id, "chunk_id": chunk_id, "text": document.page_content,
                "file_nm": metadata.get("source_filename") or metadata.get("file_nm"), "page": page,
                "score": float(score), "score_type": "dense_relevance", "retriever": "main_advanced_dense",
                "dense_rank": len(results) + 1, "metadata": metadata})
        return results
