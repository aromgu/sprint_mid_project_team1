from pathlib import Path

from scripts.prepare_eval_corpus_v3 import match_documents
from src.evaluation.golden_v3 import GoldenV3Item, evaluate_result_set, fact_match_score, section_match
from src.search.models import SearchChunk, SearchResult


def make_chunk(text: str, section_path=None, requirements=None) -> SearchChunk:
    return SearchChunk(
        "eval_01_p004_c0001", "eval_01", "문서", 4, 4,
        section_path or [], requirements or [], "section", text, 50,
    )


def test_requirement_section_matches_without_stable_chunk_id() -> None:
    chunk = make_chunk("하자보수 일반 요구사항", ["상세 요구사항"], ["PSR-001"])
    matched, score = section_match("Ⅳ_제안요청내용_2_상세요구사항_PSR-001", chunk)
    assert matched
    assert score == 1.0


def test_semantic_section_heading_matches_normalized_content() -> None:
    chunk = make_chunk("입찰 및 계약 방식에 대한 평가 기준", ["2. 입찰 및 계약 방식"])
    assert section_match("Ⅴ_제안안내_2_입찰및계약방식", chunk)[0]


def test_fact_match_requires_expected_number() -> None:
    fact = "사업비 391,542,840원, 공동수급 지분율 10% 이상"
    assert fact_match_score(fact, "사업비는 391,542,840원이며 지분율은 10% 이상이다") >= 0.7
    assert fact_match_score(fact, "사업비와 공동수급 지분율을 규정한다") == 0.0


def test_document_matching_ignores_filename_punctuation(tmp_path: Path) -> None:
    pdf = tmp_path / "한국수자원공사_건설통합시스템(CMS) 고도화.pdf"
    pdf.touch()
    matches = match_documents(["한국수자원공사_건설통합시스템CMS_고도화.pdf"], [pdf])
    assert matches[0][1] == pdf


def test_fact_coverage_changes_by_k() -> None:
    item = GoldenV3Item(
        "q1", "사업비와 기간은?", "문서.pdf", ["Ⅰ_사업개요"], [], "사업비 100원, 기간 12개월",
        ["사업비 100원", "기간 12개월"], True, "multihop", "hard", 2, "draft",
    )
    chunks = [
        make_chunk("사업비는 100원이다", ["사업개요"]),
        make_chunk("기간은 12개월이다", ["사업개요"]),
    ]
    results = [SearchResult(chunk, index, 1.0, "bm25", {}, {}, 1.0) for index, chunk in enumerate(chunks, 1)]
    metrics = evaluate_result_set(item, results, ks=(1, 3))["metrics"]
    assert metrics["fact_coverage@1"] < metrics["fact_coverage@3"]
