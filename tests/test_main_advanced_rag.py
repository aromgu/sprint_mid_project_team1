from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.main_rag.embeddings.build_advanced_index import normalize_advanced_metadata
from src.main_rag.answerability import classify_answer_status, is_answerable_status
from src.main_rag.evaluation import adapt_retrieval_results
from src.main_rag.service import MainAdvancedRAGService
from src.main_rag.settings import load_settings
from src.main_rag.runtime import MainAdvancedSessionManager


def test_three_state_answerability_classification() -> None:
    assert classify_answer_status("사업비는 1억원입니다.", [{}]) == "answered"
    assert classify_answer_status("사업비는 1억원이며 일정은 확인 불가합니다.", [{}]) == "partially_answered"
    assert classify_answer_status("공고문에서 해당 조건은 확인할 수 없습니다.", [{}]) == "partially_answered"
    assert classify_answer_status("문서에서 확인 불가합니다.", []) == "unanswerable"
    assert classify_answer_status("추가 확인이 필요합니다.", [{}], needs_clarification=True) == "partially_answered"
    assert is_answerable_status("partially_answered") is True
    assert is_answerable_status("unanswerable") is False


class FakeRetriever:
    def search_documents(self, query, *, top_k=None, document_id=None):
        assert query == "재작성 질문"
        assert document_id == "eval_01"
        return [
            {
                "chunk_id": "chunk-1",
                "text": "사업 수행 기간은 계약일로부터 6개월이다.",
                "file_nm": "sample.pdf",
                "page": 3,
                "score": 0.9,
                "metadata": {"document_id": "eval_01"},
            }
        ]


class FakeSession:
    async def rewrite_query(self, question):
        return "재작성 질문"

    async def ask(self, question, documents, rewritten_query=None):
        assert documents[0]["chunk_id"] == "chunk-1"
        assert rewritten_query == "재작성 질문"
        return {
            "answer": "6개월입니다.",
            "evidence": [
                {
                    "source": "sample.pdf",
                    "page": 3,
                    "chunk_id": "chunk-1",
                    "quote": "계약일로부터 6개월",
                    "score": 0.9,
                }
            ],
            "confidence": 0.9,
        }

    def reset(self):
        pass


def test_settings_paths_stay_inside_workspace():
    settings = load_settings()
    assert settings.path("chunks").is_relative_to(settings.project_root)
    assert settings.path("chroma").is_relative_to(settings.project_root)


def test_metadata_preserves_eval_document_and_page():
    metadata = normalize_advanced_metadata(
        {
            "chunk_id": "chunk-1",
            "document_id": "eval_01",
            "source_id": "abcdef0123456789",
            "source_filename": "sample.pdf",
            "page_start": 3,
        },
        embedding_model="text-embedding-3-small",
        create_date="2026-01-01T00:00:00+00:00",
    )
    assert metadata["document_id"] == "eval_01"
    assert metadata["page_start"] == 3
    assert metadata["file_nm"] == "sample.pdf"


def test_service_connects_rewrite_retrieval_and_generation():
    service = MainAdvancedRAGService(
        retriever=FakeRetriever(),
        session=FakeSession(),
    )
    result = asyncio.run(
        service.answer("사업 수행 기간은?", document_id="eval_01", top_k=5)
    )
    assert result["answer"] == "6개월입니다."
    assert result["retrieved_chunk_ids"] == ["chunk-1"]
    assert result["latency"]["total_seconds"] >= 0


def test_retrieval_adapter_preserves_identity_page_score_and_rank():
    adapted = adapt_retrieval_results(
        [{
            "chunk_id": "chunk-1",
            "text": "PER-004 오류 메시지는 5초 이내 제시한다.",
            "page": 7,
            "score": 0.88,
            "metadata": {"document_id": "eval_01", "page_end": 8},
        }],
        latency_ms=12.5,
    )
    result = adapted[0]
    assert result.rank == 1
    assert result.score == 0.88
    assert result.chunk.document_id == "eval_01"
    assert result.chunk.page_start == 7
    assert result.chunk.page_end == 8
    assert result.chunk.requirement_ids == ("PER-004",)


def test_runtime_sessions_are_isolated_and_reset(monkeypatch):
    manager = MainAdvancedSessionManager(retriever=FakeRetriever(), max_sessions=10)
    created = []

    def fake_service(provider):
        service = SimpleNamespace(provider=provider, reset=lambda: None)
        created.append(service)
        return service

    monkeypatch.setattr(manager, "_new_service", fake_service)
    first = manager.get("browser-1", "eval_01", "openai")
    assert manager.get("browser-1", "eval_01", "openai") is first
    assert manager.get("browser-1", "eval_02", "openai") is not first
    assert manager.get("browser-1", "eval_01", "gemini-lite") is not first
    assert len(created) == 3
    assert manager.reset("browser-1", "eval_01") == 2
    assert manager.session_count == 1
