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
    from backend.models import AskRequest, EligibilityStatusUpdate, OverviewResponse, RequirementsResponse

    assert OverviewResponse(document_id="상_1").eligibility_summary == "review_required"
    assert RequirementsResponse(document_id="상_1").items == []
    assert AskRequest(question="질문").provider == "gemini-lite"
    assert AskRequest(question="질문", provider="gemini").provider == "gemini"
    assert AskRequest(question="질문", provider="gemini-lite").provider == "gemini-lite"
    assert EligibilityStatusUpdate(user_status="unchecked").user_status == "unchecked"


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


def test_workspace_card_evidence_accepts_bracketed_source_id() -> None:
    from backend.models import EligibilityItem, EligibilityResponse
    from backend.services.rag_client import RAGClient

    chunk = SearchChunk("c2", "eval_01", "문서", 9, 9, [], [], "section", "선택 근거", 10)
    value = EligibilityResponse(items=[EligibilityItem(
        id="e1", title="자격", description="조건", source_ids=["[S2]"],
    )])

    RAGClient._attach_source_evidence(
        value,
        {"S2": SearchResult(chunk, 1, 0.8, "main_advanced_dense", {}, {}, 1.0)},
    )

    assert value.items[0].evidence is not None
    assert value.items[0].evidence.chunk_id == "c2"


def test_workspace_card_without_source_id_recovers_closest_evidence() -> None:
    from backend.models import EligibilityItem, EligibilityResponse
    from backend.services.rag_client import RAGClient

    first = SearchChunk("c1", "eval_01", "문서", 2, 2, [], [], "section", "일반 사업 개요", 10)
    second = SearchChunk("c2", "eval_01", "문서", 11, 11, [], [], "section", "입찰 참가 자격 업종코드 1468", 10)
    source_map = {
        "S1": SearchResult(first, 1, 0.9, "main_advanced_dense", {}, {}, 1.0),
        "S2": SearchResult(second, 2, 0.8, "main_advanced_dense", {}, {}, 1.0),
    }
    value = EligibilityResponse(items=[EligibilityItem(
        id="e1", title="입찰 참가 자격", description="업종코드 1468 등록",
    )])

    RAGClient._attach_source_evidence(value, source_map)

    assert value.items[0].evidence is not None
    assert value.items[0].evidence.chunk_id == "c2"
    assert value.items[0].evidence.page_number == 11


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


def test_requirement_table_fallback_extracts_distinct_cards_and_removes_headers() -> None:
    from backend.services.rag_client import RAGClient

    def result(chunk_id: str, text: str, page: int) -> SearchResult:
        ids = __import__("re").findall(r"\b[A-Z]{2,5}-\d{2,5}\b", text)
        chunk = SearchChunk(
            chunk_id, "eval_03", "한영대학", page, page, ["Ⅱ. 기술 및 기능"], ids,
            "table", text, 50,
        )
        return SearchResult(chunk, 1, 0.9, "main_advanced_dense", {}, {}, 1.0)

    quality = """| 요구사항 분류 | 품질 |
| 요구사항 고유번호 | QUR-010 |
| 요구사항 명칭 | 보안정책 및 지침준수 |
| 요구사항 상세설명 | 정의 | 보안정책 및 지침준수 개념 정의 |
| | 세부 / 내용 | 시스템은 발주기관의 보안정책에 따라 개발되어야 한다. / ○ 개발 보안가이드를 준수해야 한다. |"""
    contract = """| 요구사항 분류 | 제약사항 |
| 요구사항 고유번호 | COR-006 |
| 요구사항 명칭 | 유상 유지관리 요건 |
| 요구사항 상세설명 | 정의 | 유상 유지관리 요건 정의 |
| | 세부 내용 | 하자담보 책임기간 이후 유상 유지관리 계약을 체결해야 한다. |"""

    items = RAGClient._fallback_requirements([
        result("qur", quality, 28), result("cor", contract, 29), result("qur", quality, 28),
    ])

    assert [item.id for item in items] == ["QUR-010", "COR-006"]
    assert [item.title for item in items] == ["보안정책 및 지침준수", "유상 유지관리 요건"]
    assert [item.category for item in items] == ["quality", "contract"]
    assert "요구사항 분류" not in items[0].description
    assert items[0].description == "시스템은 발주기관의 보안정책에 따라 개발되어야 한다. ○ 개발 보안가이드를 준수해야 한다."
    assert items[1].evidence is not None and items[1].evidence.page_number == 29


def test_requirement_fallback_deduplicates_identical_cards_without_requirement_id() -> None:
    from backend.services.rag_client import RAGClient

    def result(chunk_id: str, page: int, text: str) -> SearchResult:
        chunk = SearchChunk(
            chunk_id, "eval_01", "농수산", page, page, ["Ⅳ. 제안요청 내용"], [],
            "requirement", text, 30,
        )
        return SearchResult(chunk, 1, 0.9, "main_advanced_dense", {}, {}, 1.0)

    items = RAGClient._fallback_requirements([
        result("c1", 10, "수행 계약 시 제안요청 내용을 준수하여야 한다."),
        result("c2", 11, "수행   계약 시 제안요청 내용을 준수하여야 한다."),
        result("c3", 12, "수행 계약 시 제안요청 내용을 준수하여야 한다."),
    ])

    assert len(items) == 1
    assert items[0].title == "Ⅳ. 제안요청 내용"
    assert items[0].description == "수행 계약 시 제안요청 내용을 준수하여야 한다."


def test_deliverable_retrieval_merges_queries_and_deduplicates_chunks() -> None:
    from backend.services.rag_client import RAGClient

    first = SearchChunk("c1", "상_1", "제목", 7, 7, [], [], "section", "착수신고서를 제출", 10)
    second = SearchChunk("c2", "상_1", "제목", 9, 9, [], [], "section", "최종보고서를 제출", 10)
    result1 = SearchResult(first, 1, 0.9, "bm25", {}, {}, 1.0)
    result2 = SearchResult(second, 1, 0.8, "bm25", {}, {}, 1.0)
    client = object.__new__(RAGClient)
    client._retrieve = lambda document_id, query, top_k: [result1] if "착수신고서" not in query else [result1, result2]

    results = client._retrieve_deliverables("상_1")

    assert [result.chunk.chunk_id for result in results] == ["c1", "c2"]


def test_deliverable_fallback_separates_bid_and_project_outputs() -> None:
    from backend.services.rag_client import RAGClient

    bid = SearchChunk("c1", "상_1", "제목", 7, 7, [], [], "section", "제안서 원본 1부를 제출한다", 10)
    output = SearchChunk("c2", "상_1", "제목", 9, 9, [], [], "section", "사업 완료 시 최종보고서와 소스코드를 납품한다", 10)
    results = [SearchResult(chunk, 1, 0.9, "bm25", {}, {}, 1.0) for chunk in (bid, output)]

    items = RAGClient._fallback_deliverables(results)

    assert [item.kind for item in items] == ["bid_submission", "project_deliverable"]


def test_deliverable_fallback_adds_titles_summaries_and_deduplicates() -> None:
    from backend.services.rag_client import RAGClient

    plan_text = """| 요구사항 고유번호 | PMR-006 |
| 요구사항 명칭 | 정기보고 |
| 요구사항 상세설명 | 정의 | 정기보고 제출 방안 |
| | 세부 / 내용 | 계약일로부터 10일 이내 사업수행계획서를 제출하여야 함 |"""
    plan = SearchChunk("c1", "eval_03", "한영대학", 30, 30, ["프로젝트관리"], ["PMR-006"], "table", plan_text, 20)
    duplicate = SearchChunk("c2", "eval_03", "한영대학", 31, 31, ["프로젝트관리"], [], "text", "사업수행계획서를 제출하여야 함", 10)
    manual = SearchChunk("c3", "eval_03", "한영대학", 32, 32, ["프로젝트지원"], [], "text", "관리자 매뉴얼을 전자파일로 제출한다", 10)
    results = [SearchResult(chunk, index, .9, "main_advanced_dense", {}, {}, 1.0) for index, chunk in enumerate((plan, duplicate, manual), 1)]

    items = RAGClient._fallback_deliverables(results)

    assert [item.name for item in items] == ["사업수행계획서", "관리자 매뉴얼"]
    assert "요구사항 고유번호" not in items[0].description
    assert items[0].evidence is not None and items[0].evidence.page_number == 30
    assert items[1].format == "PDF/전자파일"


def test_workspace_retrieval_uses_main_advanced_adapter() -> None:
    from backend.services.rag_client import RAGClient

    class FakeAdvanced:
        def search_documents(self, query, *, top_k=None, document_id=None):
            assert (query, top_k, document_id) == ("보안 요구사항", 5, "eval_01")
            return [{
                "chunk_id": "advanced-1", "text": "SER-001 접근통제를 적용한다.",
                "file_nm": "sample.pdf", "page": 7, "score": 0.88,
                "metadata": {
                    "document_id": "eval_01", "page_end": 8,
                    "content_type": "requirement", "section_path": "보안 요구사항",
                    "token_count": 12,
                },
            }]

    client = object.__new__(RAGClient)
    client.advanced_retriever = FakeAdvanced()
    client.search = None
    result = client._retrieve("eval_01", "보안 요구사항", 5, {"requirement"})[0]
    assert result.retriever == "main_advanced_dense"
    assert result.chunk.chunk_id == "advanced-1"
    assert result.chunk.page_start == 7 and result.chunk.page_end == 8
    assert result.chunk.requirement_ids == ["SER-001"]


def test_golden_v3_evaluation_is_joined_for_ui() -> None:
    from backend.routers.evaluation import golden_v3_evaluation

    payload = golden_v3_evaluation()
    assert payload is not None
    assert payload["run"]["question_count"] == 104
    assert payload["run"]["answered_question_count"] == 104
    assert payload["run"]["scored_question_count"] == 104
    assert payload["run"]["status"] == "complete"
    assert payload["run"]["generation_model"] == "gpt-5-nano"
    assert payload["run"]["evaluation_model"] == "gpt-4o-mini"
    assert payload["summary"]["answer_relevancy_by_status"]["answered"]["count"] > 0
    assert payload["summary"]["answer_relevancy_by_status"]["answered"]["average"] is not None
    assert len(payload["details"]) == 104
    assert payload["low_score_counts"]["false_rejections"] > 0
    assert payload["summary"]["faithfulness"] > 0
    assert payload["e2e"]["answerability_accuracy"] > 0
