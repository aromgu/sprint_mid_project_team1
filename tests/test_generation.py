from types import SimpleNamespace

from src.generation.models import ModelAnswer
from src.generation.openai_generator import OpenAIRAGService
from src.search.models import SearchChunk, SearchResult
from scripts.run_golden_v3_answers import rate_limit_retry_delay


def fixture_result() -> SearchResult:
    chunk = SearchChunk(
        "c1", "중_3", "체육시설 RFP", 9, 9, ["요구사항"], ["SFR-007"],
        "requirement", "예약기간과 시간대를 관리자가 설정한다.", 15,
    )
    return SearchResult(chunk, 1, 1.0, "hybrid", {"bm25": 1}, {"bm25": 10.0}, 8.0)


class FakeSearch:
    default_retriever = "hybrid"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed

    def parse(self, **kwargs):
        return SimpleNamespace(
            output_parsed=self.parsed,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def test_answer_maps_only_valid_citations() -> None:
    service = OpenAIRAGService(
        client=FakeClient(ModelAnswer(answer="관리자가 설정합니다.", is_answerable=True, source_ids=["S1", "S99"])),
        search_service=FakeSearch([fixture_result()]),
    )
    answer = service.answer("예약기간은 누가 설정합니까?")
    assert answer.is_answerable
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert answer.input_tokens == 100


def test_answer_refuses_without_valid_citation() -> None:
    service = OpenAIRAGService(
        client=FakeClient(ModelAnswer(answer="추측", is_answerable=True, source_ids=["S99"])),
        search_service=FakeSearch([fixture_result()]),
    )
    answer = service.answer("근거 없는 질문")
    assert not answer.is_answerable
    assert answer.citations == []


def test_no_search_result_skips_openai() -> None:
    service = OpenAIRAGService(client=None, search_service=FakeSearch([]))
    answer = service.answer("없는 질문")
    assert not answer.is_answerable
    assert answer.generation_latency_ms == 0


def test_compound_question_uses_default_hybrid_per_field() -> None:
    search = FakeSearch([fixture_result()])
    service = OpenAIRAGService(
        client=FakeClient(ModelAnswer(
            answer="총예산: 1억원\n무상유지보수기간: 미확인",
            is_answerable=True,
            source_ids=["S1"],
        )),
        search_service=search,
    )

    answer = service.answer("총예산과 무상유지보수기간은 얼마입니까?")

    assert answer.is_answerable
    assert len(search.calls) == 2
    assert all(call[1]["retriever"] == "hybrid" for call in search.calls)
    assert all(call[1]["top_k"] == 10 for call in search.calls)
    assert "사업비" in search.calls[0][0][0]
    assert "하자보수" in search.calls[1][0][0]
    assert answer.answer.splitlines() == ["총예산: 1억원", "무상유지보수기간: 미확인"]
    assert answer.retriever == "hybrid-multi-query"


def test_explicit_retriever_is_used_for_every_field() -> None:
    search = FakeSearch([fixture_result()])
    service = OpenAIRAGService(
        client=FakeClient(ModelAnswer(answer="예산: 1억원", is_answerable=True, source_ids=["S1"])),
        search_service=search,
    )

    answer = service.answer("총예산과 기간은?", retriever="bm25")

    assert all(call[1]["retriever"] == "bm25" for call in search.calls)
    assert answer.retriever == "bm25-multi-query"


def test_provider_can_be_selected_per_request() -> None:
    service = OpenAIRAGService(
        client=FakeClient(ModelAnswer(answer="예산: 확인", is_answerable=True, source_ids=["S1"])),
        search_service=FakeSearch([fixture_result()]),
    )
    assert service.resolve_provider("gemini") == "gemini"
    assert service.model_name_for("openai") == "gpt-5-nano"
    assert service.model_name_for("gemini") == "gemini-3.5-flash"
    assert service.resolve_provider("gemini-lite") == "gemini-lite"
    assert service.model_name_for("gemini-lite") == "gemini-3.5-flash-lite"


def test_rate_limit_retry_delay_uses_provider_hint() -> None:
    error = RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: '34s'")
    assert rate_limit_retry_delay(error, fallback=60) == 35.0


def test_non_rate_limit_error_is_not_retried() -> None:
    assert rate_limit_retry_delay(RuntimeError("invalid response"), fallback=60) is None
