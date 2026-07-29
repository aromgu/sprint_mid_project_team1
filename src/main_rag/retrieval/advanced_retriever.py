"""Lazy dense retriever for the isolated Main Advanced Chroma collection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.main_rag.settings import MainRAGSettings, load_settings


class AdvancedRetrieverError(RuntimeError):
    """A recoverable Advanced index or embedding configuration failure."""


class AdvancedRetriever:
    def __init__(self, settings: MainRAGSettings | None = None) -> None:
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
        self._vectorstore: Chroma | None = None

    def _initialize(self) -> Chroma:
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

    def search_documents(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        limit = self.default_top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k는 1 이상이어야 합니다")
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
            results.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "text": document.page_content,
                    "file_nm": metadata.get("source_filename")
                    or metadata.get("file_nm"),
                    "page": page,
                    "score": float(score),
                    "metadata": metadata,
                }
            )
        return results
