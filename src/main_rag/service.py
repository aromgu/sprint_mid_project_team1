"""Application service joining Advanced retrieval and BidMate generation."""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

from src.main_rag.generation.generate_answer import BidMateRAGSession
from src.main_rag.generation.gemini_session import GeminiBidMateRAGSession
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.settings import MainRAGSettings, load_settings


class MainAdvancedRAGService:
    def __init__(
        self,
        *,
        settings: MainRAGSettings | None = None,
        retriever: AdvancedRetriever | None = None,
        session: BidMateRAGSession | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.retriever = retriever or AdvancedRetriever(self.settings)
        if session is None:
            load_dotenv()
            provider = str(self.settings.get("generation", "provider", "openai"))
            key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            api_key = os.getenv(key_name)
            if not api_key:
                raise RuntimeError(f"{key_name}가 설정되지 않았습니다")
            session_class = GeminiBidMateRAGSession if provider == "gemini" else BidMateRAGSession
            session = session_class(
                api_key=api_key,
                model=str(self.settings.get("generation", "model", "gpt-5-nano")),
                max_context_chars=int(
                    self.settings.get("generation", "max_context_chars", 7000)
                ),
                max_docs=int(self.settings.get("generation", "max_docs", 6)),
            )
        self.session = session

    async def answer(
        self,
        question: str,
        *,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        rewritten = await self.session.rewrite_query(question)
        retrieval_started = time.perf_counter()
        documents = self.retriever.search_documents(
            rewritten,
            top_k=top_k,
            document_id=document_id,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        generation_started = time.perf_counter()
        result = await self.session.ask(question, documents, rewritten_query=rewritten)
        generation_seconds = time.perf_counter() - generation_started
        by_chunk_id = {str(doc["chunk_id"]): doc for doc in documents}
        validated_evidence = []
        rejected_evidence = []
        for evidence in result.get("evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            source = by_chunk_id.get(chunk_id)
            if source is None:
                rejected_evidence.append(evidence)
                continue
            validated_evidence.append(
                {
                    **evidence,
                    "source": source.get("file_nm") or evidence.get("source"),
                    "page": source.get("page"),
                    "chunk_id": chunk_id,
                    "score": source.get("score"),
                }
            )
        return {
            **result,
            "evidence": validated_evidence,
            "rejected_evidence": rejected_evidence,
            "rewritten_query": rewritten,
            "retrieved_chunk_ids": [doc["chunk_id"] for doc in documents],
            "latency": {
                "retrieval_seconds": round(retrieval_seconds, 6),
                "generation_seconds": round(generation_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }

    def reset(self) -> None:
        self.session.reset()
