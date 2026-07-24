from pathlib import Path

import numpy as np

from src.search.bm25 import BM25SearchEngine
from src.search.evaluation import GoldenQuery, evaluate_queries, ndcg_at_k, recall_at_k, reciprocal_rank
from src.search.hybrid import RRFHybridSearchEngine, WeightedScoreHybridSearchEngine, build_fusion_engine
from src.search.models import SearchChunk, SearchResult
from src.search.reranker import CrossEncoderReranker
from src.search.service import SearchService
from src.search.query_planning import merge_and_boost_results, plan_search_questions
from src.search.tokenization import RegexTokenizer, WhitespaceTokenizer, build_tokenizer


def chunks() -> list[SearchChunk]:
    return [
        SearchChunk("c1", "상_1", "문서", 1, 1, ["보안"], ["SER-001"], "requirement", "사용자 접근통제와 감사기록을 저장한다", 20),
        SearchChunk("c2", "상_1", "문서", 2, 2, ["기능"], ["SFR-001"], "requirement", "전자기록 검색 기능을 제공한다", 20),
        SearchChunk("c3", "하_1", "문서2", 3, 3, ["예약"], [], "section", "체육관 예약과 결제를 처리한다", 20),
    ]


def test_bm25_search_and_cache(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("fixture", encoding="utf-8")
    engine = BM25SearchEngine(chunks(), chunks_path, tmp_path / "indexes")
    engine.build()
    results = engine.search("SER-001 접근통제", top_k=2)
    assert results[0].chunk.chunk_id == "c1"
    assert engine.cache_path.exists()


class FakeEngine:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k, filters=None):
        return self.results[:top_k]


def result(chunk: SearchChunk, rank: int, name: str, score: float) -> SearchResult:
    return SearchResult(chunk, rank, score, name, {name: rank}, {name: score})


def test_rrf_fuses_rankings() -> None:
    items = chunks()
    bm25 = FakeEngine([result(items[0], 1, "bm25", 5.0), result(items[1], 2, "bm25", 3.0)])
    dense = FakeEngine([result(items[1], 1, "dense", 0.9), result(items[0], 2, "dense", 0.8)])
    hybrid = RRFHybridSearchEngine(bm25, dense, rrf_k=60)
    results = hybrid.search("질문", top_k=2)
    assert {item.chunk.chunk_id for item in results} == {"c1", "c2"}
    assert all(set(item.component_ranks) == {"bm25", "dense"} for item in results)


def test_weighted_score_fuses_normalized_scores() -> None:
    items = chunks()
    bm25 = FakeEngine([result(items[0], 1, "bm25", 100.0), result(items[1], 2, "bm25", 10.0)])
    dense = FakeEngine([result(items[1], 1, "dense", 0.9), result(items[0], 2, "dense", 0.8)])
    engine = WeightedScoreHybridSearchEngine(bm25, dense, bm25_weight=2.0, dense_weight=1.0)
    assert engine.search("질문", top_k=2)[0].chunk.chunk_id == "c1"
    assert isinstance(build_fusion_engine({"type": "weighted_score"}, bm25, dense), WeightedScoreHybridSearchEngine)


def test_tokenizer_factory() -> None:
    assert isinstance(build_tokenizer({"type": "regex"}), RegexTokenizer)
    assert isinstance(build_tokenizer({"type": "whitespace"}), WhitespaceTokenizer)
    assert "접근" in build_tokenizer({"type": "korean_ngram", "ngram_sizes": [2]})("접근통제")


def test_retrieval_metrics() -> None:
    items = chunks()
    results = [result(items[0], 1, "bm25", 1.0), result(items[1], 2, "bm25", 0.5)]
    assert recall_at_k(results, {"c1", "c2"}, 1) == 0.5
    assert reciprocal_rank(results, {"c2"}, 10) == 0.5
    assert np.isclose(ndcg_at_k(results, {"c2"}, 10), 1 / np.log2(3))


def test_evaluate_queries_uses_reference_document_scope_by_default() -> None:
    class RecordingService:
        def __init__(self):
            self.document_ids = None

        def search(self, query, retriever, top_k, document_ids=None):
            self.document_ids = document_ids
            return []

    service = RecordingService()
    query = GoldenQuery("q1", "질문", ["c1"], ["상_1"], [1])

    _, summary = evaluate_queries(service, [query], "hybrid")

    assert service.document_ids == {"상_1"}
    assert summary["scope"] == "document"


def test_reranker_order_is_stable() -> None:
    assert CrossEncoderReranker.order([0.2, 0.9, 0.9, -0.1]) == [1, 2, 0, 3]


def test_neighbor_expansion_preserves_ranked_chunk() -> None:
    items = chunks()
    service = SearchService.__new__(SearchService)
    service._document_sequences = {"상_1": items[:2], "하_1": items[2:]}
    service._positions = {"c1": ("상_1", 0), "c2": ("상_1", 1), "c3": ("하_1", 0)}
    results = [result(items[0], 1, "bm25", 1.0)]
    service.expand_context(results, window=1)
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].context_chunk_ids == ["c1", "c2"]
    assert items[1].text in results[0].context_text


def test_query_planner_splits_rfp_fields_and_expands_synonyms() -> None:
    plans = plan_search_questions("총예산과 무상유지보수기간을 알려주세요")
    assert [plan.label for plan in plans] == ["예산", "유지보수"]
    assert "추정가격" in plans[0].expanded_query
    assert "안정화 지원" in plans[1].expanded_query


def test_query_planner_preserves_unmatched_compound_requirements() -> None:
    plans = plan_search_questions(
        "이 사업의 총 사업비 구성 항목과 공동수급 시 구성원별 최소 지분율 제한은 무엇인가?"
    )
    assert len(plans) == 2
    assert plans[0].label == "예산"
    assert "사업비" in plans[0].expanded_query
    assert "공동수급" in plans[1].question
    assert "최소 지분율" in plans[1].expanded_query


def test_query_planner_preserves_maintenance_followup_action() -> None:
    plans = plan_search_questions("하자보수 기간과 하자 발생 시 요청 후 처리 방식은?")
    assert len(plans) == 2
    assert plans[0].label == "유지보수"
    assert "처리 방식" in plans[1].question


def test_query_planner_does_not_split_words_ending_in_gwa() -> None:
    plans = plan_search_questions("보안 위약금 부과 대상 중 C급 판정 기준과 부과 금액은 각각 얼마인가?")
    assert len(plans) == 2
    assert plans[0].question.startswith("보안 위약금 부과 대상")
    assert plans[1].question.startswith("부과 금액")


def test_merge_boosts_requirement_amount_and_period_evidence() -> None:
    plain = chunks()[2]
    evidence = SearchChunk(
        "c4", "하_1", "문서2", 4, 4, ["계약"], ["SLA-001"], "requirement",
        "SLA-001 계약금액은 100,000,000원이며 하자보수 기간은 12개월이다", 20,
    )
    merged = merge_and_boost_results(
        [[result(plain, 1, "bm25", 1.0), result(evidence, 2, "bm25", 0.9)]], top_k=2,
    )
    assert merged[0].chunk.chunk_id == "c4"
    assert merged[0].component_scores["evidence_boost"] > 1.0
