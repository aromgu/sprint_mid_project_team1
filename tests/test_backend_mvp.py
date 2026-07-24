from pathlib import Path

from backend.services.state_service import UserStateService
from src.search.models import SearchChunk, SearchResult


def test_state_service_round_trip(tmp_path: Path) -> None:
    service = UserStateService(tmp_path / "state.json")
    saved = service.update("상_1", "eligibility", "e1", {"user_status": "met"})
    assert saved == {"user_status": "met"}
    assert service.get("상_1")["eligibility"]["e1"]["user_status"] == "met"


def test_toc_filters_placeholder_and_page_only_headings() -> None:
    from backend.routers.documents import meaningful_heading

    assert not meaningful_heading("")
    assert not meaningful_heading("본문")
    assert not meaningful_heading("목 차")
    assert not meaningful_heading("p. 12")
    assert not meaningful_heading("- - -")
    assert meaningful_heading("3. 사업 수행 범위")
    assert meaningful_heading("SFR-001 기능 요구사항")


def test_backend_models_accept_empty_evidence() -> None:
    from backend.models import AskRequest, OverviewResponse, RequirementsResponse

    assert OverviewResponse(document_id="상_1").eligibility_summary == "review_required"
    assert RequirementsResponse(document_id="상_1").items == []
    assert AskRequest(question="질문").provider == "gemini-lite"
    assert AskRequest(question="질문", provider="gemini").provider == "gemini"
    assert AskRequest(question="질문", provider="gemini-lite").provider == "gemini-lite"


def test_evidence_conversion_is_bounded() -> None:
    from backend.services.rag_client import RAGClient

    chunk = SearchChunk("c1", "상_1", "제목", 2, 2, [], ["REQ-1"], "requirement", "x" * 500, 20)
    result = SearchResult(chunk, 1, 0.8, "hybrid", {}, {}, 4.0)
    evidence = RAGClient._evidence(result)
    assert evidence.document_name == "제목"
    assert evidence.page_number == 2
    assert len(evidence.quote) <= 301
    assert evidence.requirement_ids == ["REQ-1"]


def test_workspace_card_evidence_uses_selected_source_id() -> None:
    from backend.models import EligibilityItem, EligibilityResponse
    from backend.services.rag_client import RAGClient

    first = SearchChunk("c1", "상_1", "제목", 2, 2, [], [], "section", "첫 번째 근거", 10)
    second = SearchChunk("c2", "상_1", "제목", 9, 9, [], ["REQ-9"], "section", "선택된 근거", 10)
    source_map = {
        "S1": SearchResult(first, 1, 0.9, "bm25", {}, {}, 1.0),
        "S2": SearchResult(second, 2, 0.8, "bm25", {}, {}, 1.0),
    }
    value = EligibilityResponse(items=[EligibilityItem(
        id="e1", title="자격", description="조건", source_ids=["S2"],
    )])

    RAGClient._attach_source_evidence(value, source_map)

    assert value.items[0].evidence is not None
    assert value.items[0].evidence.chunk_id == "c2"
    assert value.items[0].evidence.page_number == 9


def test_workspace_fallback_cards_keep_source_page() -> None:
    from backend.services.rag_client import RAGClient

    chunk = SearchChunk(
        "c9", "상_1", "제목", 12, 12, ["필수 제출서류"], ["SER-009"],
        "requirement_table_row", "PDF 제안서 원본 1부를 직인 날인하여 제출", 20,
    )
    result = SearchResult(chunk, 1, 0.8, "bm25", {}, {}, 1.0)

    deliverable = RAGClient._fallback_deliverables([result])[0]
    requirement = RAGClient._fallback_requirements([result])[0]

    assert deliverable.format == "PDF/전자파일"
    assert deliverable.requires_original and deliverable.requires_seal
    assert deliverable.evidence is not None and deliverable.evidence.page_number == 12
    assert requirement.category == "security"
    assert requirement.evidence is not None and requirement.evidence.chunk_id == "c9"


def test_golden_v3_evaluation_is_joined_for_ui() -> None:
    from backend.routers.evaluation import golden_v3_evaluation

    payload = golden_v3_evaluation()
    assert payload is not None
    assert payload["run"]["question_count"] == 104
    assert payload["run"]["status"] == "complete"
    assert len(payload["details"]) == 104
    assert payload["low_score_counts"]["false_rejections"] == 10
    assert payload["summary"]["faithfulness"] > 0
    assert payload["e2e"]["answerability_accuracy"] > 0
