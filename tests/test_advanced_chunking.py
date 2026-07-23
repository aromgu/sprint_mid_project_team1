"""Advanced KSS/Kiwi 청킹의 구현 전 회귀 계약을 검증한다.

이 테스트가 고정하는 공개 계약은 다음과 같다.

* ``align_kss_sentences``는 KSS가 공백을 정규화해도 원문의 연속 문자 범위를
  ``SentenceSpan``으로 복원한다.
* ``build_advanced_text_streams``는 PDF 페이지 또는 HWP 문단 경계를 넘지 않는다.
* ``pack_sentence_spans``는 512 토큰을 상한으로 사용하고, 51 토큰은 문장 단위
  overlap의 목표값으로 사용한다. 단일 문장이 상한을 넘을 때만 토큰 fallback을
  허용한다.
* corpus 결과는 기존 청킹 결과와 같이 ``chunks``와 ``summary``를 가진다.
* 임베딩 입력은 접두사 없는 ``embedding_text``이며, Kiwi 결과는 별도
  ``bm25_tokens`` 필드에 저장한다.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from scripts.run_advanced_chunking import validate_no_embedding_prefix
from src.chunking.advanced_chunking import (
    SCHEMA_VERSION,
    STRATEGY_ID,
    AdvancedChunkConfig,
    KssSentenceSplitter,
    PackedTextChunk,
    SentenceSpan,
    _repair_short_final_chunk,
    align_kss_sentences,
    build_advanced_text_streams,
    chunk_advanced_corpus,
    chunk_advanced_table_block,
    extract_page_marker_numbers,
    normalize_text_for_embedding,
    pack_sentence_spans,
    validate_advanced_chunks,
)


class CharacterCodec:
    """문자 하나를 토큰 하나로 보는 결정적 테스트 코덱이다."""

    model_name = "character-test-model"
    encoding_name = "unicode-codepoint"
    version = "test-v1"

    def encode(self, text: str) -> list[int]:
        """각 문자를 유니코드 코드 포인트 토큰으로 바꾼다."""
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        """oversized sentence fallback의 원문 복원을 지원한다."""
        return "".join(chr(token_id) for token_id in token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        """UTF-8 안전 경계 검증에 사용할 토큰 바이트를 반환한다."""
        return chr(token_id).encode("utf-8")


class SpaceExpansionCodec:
    """줄바꿈보다 공백을 더 많은 토큰으로 세는 오프라인 테스트 코덱이다.

    실제 tiktoken도 문맥에 따라 ``newline → space`` 치환 전후 토큰 수가
    달라질 수 있다. 이 대역은 공백을 빈 토큰+공백 토큰 두 개로 표현해
    네트워크나 tiktoken 캐시에 의존하지 않고 그 차이를 결정적으로 만든다.
    """

    model_name = "space-expansion-test-model"
    encoding_name = "space-two-tokens"
    version = "test-v1"
    _EMPTY_TOKEN = 0x110000
    _SPACE_TOKEN = 0x110001

    def encode(self, text: str) -> list[int]:
        """공백은 2토큰, 나머지 문자는 1토큰으로 변환한다."""
        tokens: list[int] = []
        for char in text:
            if char == " ":
                tokens.extend([self._EMPTY_TOKEN, self._SPACE_TOKEN])
            else:
                tokens.append(ord(char))
        return tokens

    def token_bytes(self, token_id: int) -> bytes:
        """TokenTextMap이 원문 UTF-8을 정확히 재구성하도록 바이트를 준다."""
        if token_id == self._EMPTY_TOKEN:
            return b""
        if token_id == self._SPACE_TOKEN:
            return b" "
        return chr(token_id).encode("utf-8")


class KssSpy:
    """공백을 정규화하고 호출 원문을 기록하는 KSS 대역이다."""

    def __init__(self, responses: Sequence[Sequence[str]] | None = None) -> None:
        self.responses = [list(response) for response in responses or []]
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[str]:
        """실제 ``Kss('split_sentences')``처럼 callable로 동작한다."""
        self.calls.append(text)
        if self.responses:
            return self.responses.pop(0)
        return [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()
        ]

    def split_sentences(self, text: str) -> list[str]:
        """함수형 wrapper를 쓰는 구현도 같은 대역을 사용할 수 있게 한다."""
        return self(text)


class KiwiSpy:
    """형태소 분석 대상과 반환 표면형을 기록하는 Kiwi 대역이다."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[SimpleNamespace]:
        """Kiwi Token의 ``form``과 ``tag``만 제공한다."""
        self.calls.append(text)
        return [
            SimpleNamespace(form=token.strip(".,!?"), tag="NNG")
            for token in text.split()
            if token.strip(".,!?")
        ]


class TaggedKiwiSpy:
    """J/E 제외 정책을 품사별로 고정하는 Kiwi 대역이다."""

    def __init__(self, values: Sequence[SimpleNamespace]) -> None:
        self.values = list(values)
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[SimpleNamespace]:
        """입력 호출을 기록하고 준비한 Kiwi 형태소를 반환한다."""
        self.calls.append(text)
        return list(self.values)


CODEC = CharacterCodec()
CONFIG = AdvancedChunkConfig(
    max_tokens=512,
    overlap_tokens=51,
    model_name=CODEC.model_name,
    encoding_name=CODEC.encoding_name,
    strategy_id="advanced_kss_kiwi_character_512_51_test",
)


def make_document(
    source_id: str = "source-001",
    *,
    file_type: str = "hwp",
) -> dict[str, Any]:
    """출처와 사업 메타데이터를 포함한 최소 Advanced 문서를 만든다."""
    return {
        "schema_version": "rfp_advanced_preprocessing_v1",
        "source_id": source_id,
        "document_id": source_id,
        "source_sha256": (source_id.encode().hex() + "0" * 64)[:64],
        "source_filename": f"{source_id}.{file_type}",
        "source_relative_path": f"원본/{source_id}.{file_type}",
        "filename_aliases": [f"{source_id}-별칭.{file_type}"],
        "file_type": file_type,
        "project_name": f"{source_id} 정보시스템 구축",
        "issuer": "테스트 발주기관",
        "notice_number": "2026-0001",
        "embedding_prefix_policy": "metadata_only_not_in_vector_text",
    }


def make_text_block(
    order: int,
    text: str,
    *,
    source_id: str = "source-001",
    file_type: str = "hwp",
    section_idx: int | None = 0,
    para_idx: int | None = None,
    page: int | None = None,
) -> dict[str, Any]:
    """KSS와 Kiwi 처리 대상인 일반 텍스트 블록을 만든다."""
    resolved_para = order if para_idx is None and file_type != "pdf" else para_idx
    resolved_page = page if file_type == "pdf" else None
    if file_type == "pdf":
        boundary_type = "pdf_page"
        boundary_id = f"{source_id}:page:{int(resolved_page or 1):04d}"
    else:
        boundary_type = "hwp_paragraph"
        boundary_id = f"{source_id}:section:{section_idx}:paragraph:{resolved_para}"
    return {
        "schema_version": "rfp_advanced_preprocessing_v1",
        "source_id": source_id,
        "document_id": source_id,
        "block_id": f"{source_id}:B{order:06d}",
        "block_order": order,
        "block_type": "text" if file_type == "pdf" else "paragraph",
        "content_type": "text",
        "index_policy": "index",
        "section_path": "Ⅰ. 사업 개요",
        "section_idx": None if file_type == "pdf" else section_idx,
        "para_idx": None if file_type == "pdf" else resolved_para,
        "page": resolved_page,
        "table_id": None,
        "picture_id": None,
        "text": text,
        "table_html": None,
        "table_markdown": None,
        "image_uri": None,
        "vectorize_field": "text",
        "dense_eligible": True,
        "kss_eligible": True,
        "bm25_eligible": True,
        "kss_boundary_type": boundary_type,
        "kss_boundary_id": boundary_id,
        "embedding_prefix_included": False,
        "quality_flags": [],
    }


def make_table_block(
    order: int,
    markdown: str,
    *,
    source_id: str = "source-001",
    section_idx: int | None = 0,
    para_idx: int | None = 9,
    page: int | None = None,
) -> dict[str, Any]:
    """KSS와 Kiwi에서 제외되는 Markdown 표 블록을 만든다."""
    table_id = f"{source_id}:T{order:06d}"
    return {
        "schema_version": "rfp_advanced_preprocessing_v1",
        "source_id": source_id,
        "document_id": source_id,
        "block_id": f"{source_id}:B{order:06d}",
        "block_order": order,
        "block_type": "table",
        "content_type": "table",
        "index_policy": "index",
        "section_path": "Ⅱ. 요구사항",
        "section_idx": section_idx,
        "para_idx": para_idx,
        "page": page,
        "table_id": table_id,
        "picture_id": None,
        "text": None,
        "table_html": "<table><tr><th>구분</th><th>내용</th></tr></table>",
        "table_markdown": markdown,
        "image_uri": None,
        "vectorize_field": "table_markdown",
        "dense_eligible": True,
        "kss_eligible": False,
        "bm25_eligible": False,
        "kss_boundary_type": None,
        "kss_boundary_id": None,
        "embedding_prefix_included": False,
        "render_mode": "dual_html_gfm",
        "format_version": "html_gfm_dual_v1",
        "quality_flags": [],
    }


def make_image_block(
    order: int,
    *,
    source_id: str = "source-001",
) -> dict[str, Any]:
    """검색·문장 분리에서 제외되는 이미지 참조 블록을 만든다."""
    return {
        "schema_version": "rfp_advanced_preprocessing_v1",
        "source_id": source_id,
        "document_id": source_id,
        "block_id": f"{source_id}:B{order:06d}",
        "block_order": order,
        "block_type": "picture",
        "content_type": "image",
        "index_policy": "exclude",
        "section_path": "Ⅱ. 요구사항",
        "section_idx": 0,
        "para_idx": order,
        "page": None,
        "table_id": None,
        "picture_id": f"{source_id}:I{order:06d}",
        "text": None,
        "table_html": None,
        "table_markdown": None,
        "image_uri": f"image://{source_id}:I{order:06d}",
        "vectorize_field": None,
        "dense_eligible": False,
        "kss_eligible": False,
        "bm25_eligible": False,
        "kss_boundary_type": None,
        "kss_boundary_id": None,
        "embedding_prefix_included": False,
        "quality_flags": [],
    }


def run_corpus(
    documents: Sequence[dict[str, Any]],
    blocks: Sequence[dict[str, Any]],
    *,
    kss: KssSpy | None = None,
    kiwi: KiwiSpy | None = None,
):
    """공통 fake 의존성으로 Advanced corpus를 생성한다."""
    return chunk_advanced_corpus(
        documents,
        blocks,
        codec=CODEC,
        config=CONFIG,
        sentence_splitter=kss or KssSpy(),
        kiwi_tokenizer=kiwi or KiwiSpy(),
    )


def test_default_config_uses_512_limit_and_51_target_overlap() -> None:
    """Advanced 전략 기본값은 512 상한과 약 10%인 51 overlap이다."""
    config = AdvancedChunkConfig()

    assert config.max_tokens == 512
    assert config.overlap_tokens == 51
    assert config.min_tail_tokens == 51


def test_advanced_chunk_output_contract_is_versioned_v2() -> None:
    """의미 기반 tail 변경은 v1과 충돌하지 않는 별도 스키마·전략을 쓴다."""
    assert SCHEMA_VERSION == "rfp_advanced_chunk_v2"
    assert STRATEGY_ID.endswith("_v2")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("- 2 -", (2,)),
        ("-5-", (5,)),
        ("- 130 - - 131 -", (130, 131)),
        ("- 130 -\u200b- 131 -", (130, 131)),
        ("\uf085—\u2009130\u2009—\uf085", (130,)),
        ("− 9 −\n− 10 −", (9, 10)),
        ("제1조", None),
        ("- 2026 -", None),
        ("사업비 - 123 - 원", None),
        ("- 130 - - 132 -", None),
        ("- 001 -", None),
        ("- 0 -", None),
        ("－ １ －", None),
    ],
)
def test_page_marker_detector_is_narrow_and_preserves_lookalikes(
    value: str,
    expected: tuple[int, ...] | None,
) -> None:
    """PDF footer 문법만 찾고 연도·금액·전각 숫자는 본문으로 보존한다."""
    original = value

    assert extract_page_marker_numbers(value) == expected
    assert value == original


def test_whitespace_normalizing_kss_preserves_exact_original_spans() -> None:
    """KSS가 공백을 제거해도 span들은 원문을 한 글자도 잃지 않는다."""
    source = "첫째 문장입니다.   둘째 문장입니다.\n\t셋째 문장입니다."
    normalized = ["첫째 문장입니다.", "둘째 문장입니다.", "셋째 문장입니다."]

    spans = align_kss_sentences(
        source,
        normalized,
        boundary_id="source-001:section:0:paragraph:1",
    )

    assert all(isinstance(span, SentenceSpan) for span in spans)
    assert [span.normalized_text for span in spans] == normalized
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(source)
    assert "".join(span.raw_text for span in spans) == source
    assert all(
        span.raw_text == source[span.char_start : span.char_end] for span in spans
    )
    assert "   " in "".join(span.raw_text for span in spans)
    assert "\n\t" in "".join(span.raw_text for span in spans)


def test_alignment_mismatch_reports_boundary_context() -> None:
    """공백 차이가 아닌 KSS 불일치는 조용히 원문을 훼손하지 않는다."""
    boundary_id = "source-001:section:0:paragraph:99"

    with pytest.raises(ValueError, match=boundary_id):
        align_kss_sentences(
            "실제 원문입니다.",
            ["다른 문장입니다."],
            boundary_id=boundary_id,
        )


def test_private_use_hwp_marker_is_ignored_only_for_alignment() -> None:
    """pecab이 버리는 한컴 PUA 글리프도 최종 원문 span에는 그대로 남는다."""
    source = "벤처확인 재확인\uf08c, 또는 취소\uf08d(청문절차)입니다."
    normalized = ["벤처확인 재확인,또는취소 (청문절차)입니다."]

    spans = align_kss_sentences(
        source,
        normalized,
        boundary_id="source-001:section:0:paragraph:232",
    )

    assert "".join(span.raw_text for span in spans) == source
    assert "\uf08c" in spans[0].raw_text
    assert "\uf08d" in spans[0].raw_text


def test_private_use_marker_is_sanitized_before_kss_and_preserved_in_chunk() -> None:
    """한컴 PUA는 KSS 입력에서만 공백이 되고 최종 청크에는 남는다."""
    source = (
        "\uf085 Windows Server, MS-SQL기반 ASP.NET 개발환경에 구축 실적이 있는 업체"
    )
    normalized = ["Windows Server, MS-SQL기반 ASP.NET 개발환경에 구축 실적이 있는 업체"]
    document = make_document()
    block = make_text_block(1, source, para_idx=127)
    kss = KssSpy(responses=[normalized])

    result = run_corpus([document], [block], kss=kss)
    chunk = result.chunks[0]

    assert kss.calls == [
        "  Windows Server, MS-SQL기반 ASP.NET 개발환경에 구축 실적이 있는 업체"
    ]
    assert chunk["embedding_text"] == source
    assert chunk["kss_input_sanitized"] is True
    assert chunk["kss_stream_sanitized_character_count"] == 1
    assert chunk["kss_alignment_fallback"] is False
    assert chunk["kss_alignment_status"] == "sanitized_aligned"
    assert "kss_input_sanitized_private_format_or_decorative" in chunk["quality_flags"]


def test_kss_alignment_corruption_falls_back_to_exact_whole_boundary() -> None:
    """KSS가 의미 문자를 바꿔도 해당 위치 원문을 잃거나 섞지 않는다."""
    source = "원문을 반드시 그대로 보존합니다."
    document = make_document()
    block = make_text_block(1, source, para_idx=128)
    kss = KssSpy(responses=[["KSS가 완전히 바꾼 문자열입니다."]])

    result = run_corpus([document], [block], kss=kss)
    chunk = result.chunks[0]

    assert chunk["embedding_text"] == source
    assert chunk["kss_alignment_fallback"] is True
    assert "kss_alignment_fallback_whole_boundary" in chunk["quality_flags"]
    assert result.summary["kss_alignment_fallback_chunk_count"] == 1
    assert result.summary["kss_alignment_fallback_stream_count"] == 1


def test_real_pecab_private_use_regression_preserves_exact_source() -> None:
    """실제 KSS 6.0.6/pecab에서도 U+F085 숫자 변환 회귀를 차단한다."""
    source = (
        "\uf085 Windows Server, MS-SQL기반 ASP.NET 개발환경에 구축 실적이 있는 업체"
    )
    document = make_document()
    block = make_text_block(1, source, para_idx=127)

    result = chunk_advanced_corpus(
        [document],
        [block],
        codec=CODEC,
        config=CONFIG,
        sentence_splitter=KssSentenceSplitter(),
        kiwi_tokenizer=KiwiSpy(),
    )
    chunk = result.chunks[0]

    assert chunk["embedding_text"] == source
    assert chunk["kss_alignment_status"] == "sanitized_aligned"
    assert chunk["kss_alignment_fallback"] is False


def test_sentence_splitter_runtime_error_is_not_hidden_by_alignment_fallback() -> None:
    """KSS 실행 장애는 원문 fallback으로 숨기지 않고 호출자에게 전달한다."""
    document = make_document()
    block = make_text_block(1, "KSS 실행 오류 전파를 검증합니다.")

    def failing_splitter(_: str) -> list[str]:
        raise RuntimeError("KSS backend unavailable")

    with pytest.raises(RuntimeError, match="KSS backend unavailable"):
        chunk_advanced_corpus(
            [document],
            [block],
            codec=CODEC,
            config=CONFIG,
            sentence_splitter=failing_splitter,
            kiwi_tokenizer=KiwiSpy(),
        )


def test_decorative_bullet_omitted_by_kss_stays_in_raw_text() -> None:
    """KSS가 생략하는 장식 bullet도 경계에서만 무시하고 원문은 보존한다."""
    source = "⦁\U000f0854지방자치단체 계약법\U000f0855 제43조입니다."
    normalized = ["지방자치단체계약법제 43 조입니다."]

    spans = align_kss_sentences(
        source,
        normalized,
        boundary_id="source-002:section:0:paragraph:173",
    )

    assert "".join(span.raw_text for span in spans) == source
    assert spans[0].raw_text.startswith("⦁\U000f0854")


def test_hwp_text_streams_never_cross_section_and_paragraph_boundary() -> None:
    """HWP에서는 같은 section이어도 서로 다른 para를 합치지 않는다."""
    document = make_document(file_type="hwp")
    blocks = [
        make_text_block(1, "첫 문단입니다.", section_idx=0, para_idx=7),
        make_text_block(2, "둘째 문단입니다.", section_idx=0, para_idx=8),
        make_text_block(3, "다른 절 문단입니다.", section_idx=1, para_idx=1),
    ]
    kss = KssSpy()

    streams = build_advanced_text_streams(document, blocks)

    assert len(streams) == 3
    assert [stream.boundary_id for stream in streams] == [
        block["kss_boundary_id"] for block in blocks
    ]
    run_corpus([document], blocks, kss=kss)
    assert kss.calls == [block["text"] for block in blocks]


def test_pdf_text_streams_join_only_blocks_on_the_same_page() -> None:
    """PDF의 같은 페이지 텍스트는 합치되 다음 페이지로 넘어가지 않는다."""
    document = make_document(file_type="pdf")
    blocks = [
        make_text_block(1, "첫 페이지 앞 문장입니다.", file_type="pdf", page=1),
        make_text_block(2, "첫 페이지 뒤 문장입니다.", file_type="pdf", page=1),
        make_text_block(3, "둘째 페이지 문장입니다.", file_type="pdf", page=2),
    ]
    kss = KssSpy()

    streams = build_advanced_text_streams(document, blocks)

    assert len(streams) == 2
    assert [stream.boundary_id for stream in streams] == [
        "source-001:page:0001",
        "source-001:page:0002",
    ]
    assert "첫 페이지 앞" in streams[0].text
    assert "첫 페이지 뒤" in streams[0].text
    assert "둘째 페이지" not in streams[0].text
    assert "둘째 페이지" in streams[1].text
    run_corpus([document], blocks, kss=kss)
    assert len(kss.calls) == 2


def test_pdf_page_marker_after_table_is_metadata_only() -> None:
    """표 뒤의 페이지 번호는 별도 벡터가 되지 않고 입력 block/page에 남는다."""
    document = make_document(file_type="pdf")
    table = make_table_block(
        1,
        "| 구분 | 내용 |\n| --- | --- |\n| 기간 | 90일 |",
        section_idx=None,
        para_idx=None,
        page=1,
    )
    marker = make_text_block(2, "- 1 -", file_type="pdf", page=1)
    kss = KssSpy()
    kiwi = KiwiSpy()

    result = run_corpus([document], [table, marker], kss=kss, kiwi=kiwi)

    assert [chunk["content_type"] for chunk in result.chunks] == ["table"]
    assert kss.calls == []
    assert kiwi.calls == []
    assert result.summary["page_marker_block_count"] == 1
    assert result.summary["page_marker_vectorized_block_count"] == 0
    assert result.summary["page_marker_metadata_only_block_count"] == 1
    assert result.summary["page_marker_refs"] == [
        {
            "source_id": document["source_id"],
            "block_id": marker["block_id"],
            "block_order": 2,
            "physical_page": 1,
            "printed_page_numbers": [1],
            "raw_text": "- 1 -",
            "previous_content_type": "table",
        }
    ]
    assert result.summary["validation"]["overall_pass"] is True


def test_pdf_page_marker_is_removed_even_when_same_page_text_has_room() -> None:
    """합칠 여유가 있어도 footer 문자열은 임베딩·raw 청크에서 제외한다."""
    document = make_document(file_type="pdf")
    text = make_text_block(
        1,
        "같은 페이지의 의미 있는 본문입니다.",
        file_type="pdf",
        page=1,
    )
    marker = make_text_block(2, "- 1 -", file_type="pdf", page=1)

    result = run_corpus([document], [text, marker])

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk["source_block_ids"] == [text["block_id"]]
    assert chunk["page_start"] == chunk["page_end"] == 1
    assert "- 1 -" not in chunk["embedding_text"]
    assert "- 1 -" not in chunk["raw_text"]
    assert result.summary["page_marker_vectorized_block_count"] == 0
    assert result.summary["page_marker_metadata_only_block_count"] == 1


def test_pdf_page_marker_is_suppressed_instead_of_overlap_when_merge_overflows() -> (
    None
):
    """510토큰 본문 뒤 footer는 51토큰 중복 벡터로 부풀리지 않는다."""
    document = make_document(file_type="pdf")
    substantive = make_text_block(
        1,
        ("가" * 509) + ".",
        file_type="pdf",
        page=1,
    )
    marker = make_text_block(2, "- 1 -", file_type="pdf", page=1)

    result = run_corpus([document], [substantive, marker])

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk["embedding_text"] == substantive["text"]
    assert chunk["token_count"] == 510
    assert marker["block_id"] not in chunk["source_block_ids"]
    assert result.summary["page_marker_metadata_only_block_count"] == 1
    assert result.summary["short_tail_token_overlap_fallback_count"] == 0


def test_marker_only_pdf_page_can_produce_no_vector_but_still_validate() -> None:
    """의미 본문이 없는 PDF footer page도 coverage 예외를 명시해 통과한다."""
    document = make_document(file_type="pdf")
    marker = make_text_block(1, "- 9 - - 10 -", file_type="pdf", page=5)

    result = run_corpus([document], [marker])

    assert result.chunks == ()
    assert result.summary["page_marker_block_count"] == 1
    assert result.summary["page_marker_metadata_only_block_count"] == 1
    assert result.summary["validation"]["overall_pass"] is True


def test_hwp_dash_number_text_is_not_treated_as_pdf_page_metadata() -> None:
    """물리 page 근거가 없는 HWP의 같은 문자열은 일반 문단으로 보존한다."""
    document = make_document(file_type="hwp")
    block = make_text_block(1, "- 123 -", file_type="hwp")

    result = run_corpus([document], [block])

    assert len(result.chunks) == 1
    assert result.chunks[0]["embedding_text"] == "- 123 -"
    assert result.summary["page_marker_block_count"] == 0


def test_text_embedding_replaces_newlines_with_spaces_but_preserves_raw_text() -> None:
    """일반 텍스트는 단어를 붙이지 않고 줄바꿈만 임베딩에서 제외한다."""
    document = make_document(file_type="pdf")
    raw = "첫째 줄입니다.\n\n둘째 줄입니다.\r\n셋째 줄입니다."
    block = make_text_block(1, raw, file_type="pdf", page=1)

    result = run_corpus([document], [block])

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk["raw_text"] == raw
    assert chunk["embedding_text"] == ("첫째 줄입니다.  둘째 줄입니다.  셋째 줄입니다.")
    assert "\n" not in chunk["embedding_text"]
    assert "\r" not in chunk["embedding_text"]
    assert (
        chunk["embedding_text_normalization"]
        == "line_separators_to_spaces_preserve_offsets_v1"
    )
    assert result.summary["validation"]["gates"][
        "text_newlines_are_excluded_from_embedding_only"
    ]
    validate_no_embedding_prefix(result.chunks)


def test_all_line_separators_become_offset_preserving_spaces() -> None:
    """CR/LF 외 HWP soft break와 Unicode line separator도 벡터에서 제거한다."""
    raw = "가\n나\r다\v라\f마\x85바\u2028사\u2029아"
    normalized = normalize_text_for_embedding(raw)

    assert len(normalized) == len(raw)
    assert normalized == "가 나 다 라 마 바 사 아"
    assert not any(
        separator in normalized
        for separator in ("\n", "\r", "\v", "\f", "\x85", "\u2028", "\u2029")
    )


def test_packing_budget_uses_normalized_embedding_tokens() -> None:
    """newline→space로 토큰이 늘어도 실제 embedding 기준 상한을 지킨다."""
    codec = SpaceExpansionCodec()
    raw = ".\n2026"
    normalized = normalize_text_for_embedding(raw)
    assert len(codec.encode(raw)) < len(codec.encode(normalized))
    config = AdvancedChunkConfig(
        max_tokens=len(codec.encode(raw)),
        overlap_tokens=1,
        min_tail_tokens=1,
        model_name=codec.model_name,
        encoding_name=codec.encoding_name,
        strategy_id="normalized-budget-test",
    )
    spans = align_kss_sentences(raw, [raw], boundary_id="source-001:page:0001")

    packed = pack_sentence_spans(raw, spans, codec=codec, config=config)

    assert len(packed) > 1
    assert all(
        len(codec.encode(normalize_text_for_embedding(part.text))) <= config.max_tokens
        for part in packed
    )


def test_cli_final_validation_rejects_wrong_text_newline_normalization() -> None:
    """저장 직전 검사도 text raw와 embedding이 달라질 수 있음을 이해한다."""
    document = make_document(file_type="pdf")
    block = make_text_block(
        1,
        "첫째 줄입니다.\n둘째 줄입니다.",
        file_type="pdf",
        page=1,
    )
    result = run_corpus([document], [block])
    tampered = [copy.deepcopy(result.chunks[0])]
    tampered[0]["embedding_text"] = tampered[0]["raw_text"]

    with pytest.raises(ValueError, match="정규화 계약"):
        validate_no_embedding_prefix(tampered)


def test_only_general_text_calls_kss_and_kiwi() -> None:
    """표·이미지는 KSS/Kiwi 경로에 들어가지 않고 텍스트만 분석한다."""
    document = make_document()
    text = make_text_block(1, "일반 본문을 형태소로 분석합니다.")
    table = make_table_block(
        2,
        "| 구분 | 내용 |\n| --- | --- |\n| 표행 | KSS 제외 |",
    )
    image = make_image_block(3)
    kss = KssSpy()
    kiwi = KiwiSpy()

    result = run_corpus([document], [text, table, image], kss=kss, kiwi=kiwi)

    assert kss.calls == [text["text"]]
    assert kiwi.calls
    assert all("표행" not in call for call in kiwi.calls)
    assert [chunk["content_type"] for chunk in result.chunks] == ["text", "table"]
    text_chunk, table_chunk = result.chunks
    assert text_chunk["bm25_tokens"]
    assert table_chunk["bm25_tokens"] == []
    assert table_chunk["bm25_pos_policy"] is None
    assert table_chunk["bm25_excluded_pos_prefixes"] == []


def test_kiwi_excludes_only_josa_and_eomi_prefixes() -> None:
    """팀 합의대로 J*·E*만 제외하고 다른 품사는 임의로 버리지 않는다."""
    document = make_document()
    block = make_text_block(1, "Kiwi 품사 필터를 검증합니다.")
    kiwi = TaggedKiwiSpy(
        [
            SimpleNamespace(form="사업", tag="NNG"),
            SimpleNamespace(form="은", tag="JX"),
            SimpleNamespace(form="수행", tag="NNG"),
            SimpleNamespace(form="하", tag="XSV"),
            SimpleNamespace(form="ㅂ니다", tag="EF"),
            SimpleNamespace(form="아", tag="IC"),
            SimpleNamespace(form=".", tag="SF"),
            SimpleNamespace(form="§", tag="SW"),
            SimpleNamespace(form="미등록", tag="ZZZ"),
            SimpleNamespace(form="API", tag="SL"),
            SimpleNamespace(form="", tag="NNG"),
        ]
    )

    result = run_corpus([document], [block], kiwi=kiwi)
    chunk = result.chunks[0]

    assert chunk["bm25_tokens"] == [
        "사업",
        "수행",
        "하",
        "아",
        ".",
        "§",
        "미등록",
        "api",
    ]
    assert chunk["bm25_pos_policy"] == "exclude_josa_eomi_prefix_v1"
    assert chunk["bm25_excluded_pos_prefixes"] == ["J", "E"]
    assert chunk["bm25_token_normalization"] == "strip_casefold"
    assert result.summary["bm25_excluded_pos_prefixes"] == ["J", "E"]


def test_sentence_packing_respects_512_and_exact_51_when_possible() -> None:
    """51 토큰짜리 온전한 문장은 다음 청크의 정확한 overlap이 된다."""
    first = "가" * 461
    overlap = "나" * 51
    last = "다" * 461
    spans = align_kss_sentences(
        first + overlap + last,
        [first, overlap, last],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert [part["text"] for part in packed] == [first + overlap, overlap + last]
    assert [len(CODEC.encode(part["text"])) for part in packed] == [512, 512]
    assert packed[0]["overlap_actual_tokens"] == 0
    assert packed[1]["overlap_target_tokens"] == 51
    assert packed[1]["overlap_actual_tokens"] == 51


def test_sentence_overlap_is_a_target_not_a_sentence_split_request() -> None:
    """정확히 51이 불가능하면 51 이하의 가장 긴 온전한 문장을 겹친다."""
    first = "가" * 460
    overlap = "나" * 40
    last = "다" * 460
    spans = align_kss_sentences(
        first + overlap + last,
        [first, overlap, last],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert [part["text"] for part in packed] == [first + overlap, overlap + last]
    assert packed[1]["overlap_target_tokens"] == 51
    assert packed[1]["overlap_actual_tokens"] == 40


def test_short_new_tail_reuses_existing_whole_sentence_overlap_without_copying_more() -> (
    None
):
    """기존 40토큰 overlap+신규 20이면 추가 복사 없이 60토큰을 유지한다."""
    first = "가" * 460
    overlap = "나" * 40
    tail = "다" * 20
    spans = align_kss_sentences(
        first + overlap + tail,
        [first, overlap, tail],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert [part["text"] for part in packed] == [
        first + overlap,
        overlap + tail,
    ]
    assert packed[-1]["overlap_actual_tokens"] == 40
    assert packed[-1]["short_tail_adjustment_mode"] == "existing_overlap_context"
    assert packed[-1]["short_tail_original_new_token_count"] == 20
    assert packed[-1]["short_tail_context_added_tokens"] == 40


def test_short_final_chunk_merges_without_copying_overlap_when_union_fits() -> None:
    """326+5처럼 합쳐 512 이하이면 중복 없는 연속 원문 하나로 만든다."""
    previous_text = "가" * 326
    tail = "나" * 5
    stream_text = previous_text + tail
    spans = align_kss_sentences(
        stream_text,
        [previous_text, tail],
        boundary_id="source-001:page:0001",
    )
    packed = [
        PackedTextChunk(
            text=previous_text,
            char_start=0,
            char_end=326,
            sentence_start=1,
            sentence_end=1,
            overlap_target_tokens=51,
            overlap_actual_tokens=0,
            overlap_sentence_count=0,
            oversized_sentence_split=False,
        ),
        PackedTextChunk(
            text=tail,
            char_start=326,
            char_end=331,
            sentence_start=2,
            sentence_end=2,
            overlap_target_tokens=51,
            overlap_actual_tokens=0,
            overlap_sentence_count=0,
            oversized_sentence_split=False,
        ),
    ]

    repaired = _repair_short_final_chunk(
        packed,
        stream_text,
        normalize_text_for_embedding(stream_text),
        spans,
        CODEC,
        CONFIG,
    )

    assert len(repaired) == 1
    assert repaired[0]["text"] == stream_text
    assert repaired[0]["short_tail_adjustment_mode"] == "merged_with_previous"
    assert repaired[0]["short_tail_original_new_token_count"] == 5


def test_tail_rebalance_never_leaves_an_overlap_only_middle_chunk() -> None:
    """문장 이동 후 중간 청크에 이전 overlap만 남는 후보는 거부한다."""
    first = "가" * 300
    middle_new = "나" * 450
    tail = "다" * 20
    stream_text = first + middle_new + tail
    spans = align_kss_sentences(
        stream_text,
        [first, middle_new, tail],
        boundary_id="source-001:page:0001",
    )
    packed = [
        PackedTextChunk(
            text=first,
            char_start=0,
            char_end=300,
            sentence_start=1,
            sentence_end=1,
            overlap_target_tokens=51,
            overlap_actual_tokens=0,
            overlap_sentence_count=0,
            oversized_sentence_split=False,
        ),
        PackedTextChunk(
            text=stream_text[250:750],
            char_start=250,
            char_end=750,
            sentence_start=1,
            sentence_end=2,
            overlap_target_tokens=51,
            overlap_actual_tokens=50,
            overlap_sentence_count=1,
            oversized_sentence_split=False,
        ),
        PackedTextChunk(
            text=tail,
            char_start=750,
            char_end=770,
            sentence_start=3,
            sentence_end=3,
            overlap_target_tokens=51,
            overlap_actual_tokens=0,
            overlap_sentence_count=0,
            oversized_sentence_split=False,
        ),
    ]

    repaired = _repair_short_final_chunk(
        packed,
        stream_text,
        normalize_text_for_embedding(stream_text),
        spans,
        CODEC,
        CONFIG,
    )

    assert repaired[-2]["char_end"] == 750
    assert repaired[-2]["text"] == stream_text[250:750]
    assert repaired[-1]["short_tail_adjustment_mode"] == ("safe_token_overlap_fallback")
    assert repaired[-1]["text"].endswith(tail)


def test_short_final_chunk_rebalances_whole_sentences() -> None:
    """51토큰 미만의 마지막 조각은 이전 온전한 문장을 넘겨 보정한다."""
    first = "가" * 300
    moved = "나" * 210
    tail = "다" * 10
    spans = align_kss_sentences(
        first + moved + tail,
        [first, moved, tail],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert [part["text"] for part in packed] == [first, moved + tail]
    assert [len(CODEC.encode(part["text"])) for part in packed] == [300, 220]
    assert packed[-1]["short_tail_adjusted"] is True
    assert packed[-1]["short_tail_token_overlap_fallback"] is False
    assert packed[-1]["short_tail_adjustment_mode"] == "whole_sentence_rebalance"
    assert packed[-1]["short_tail_original_new_token_count"] == 10


def test_short_final_chunk_uses_safe_overlap_when_sentence_cannot_move() -> None:
    """500+20처럼 문장 이동이 불가능하면 이전 suffix로 51을 채운다."""
    first = "가" * 500
    tail = "나" * 20
    spans = align_kss_sentences(
        first + tail,
        [first, tail],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert [len(CODEC.encode(part["text"])) for part in packed] == [500, 51]
    assert packed[-1]["text"] == first[-31:] + tail
    assert packed[-1]["overlap_actual_tokens"] == 31
    assert packed[-1]["short_tail_adjusted"] is True
    assert packed[-1]["short_tail_token_overlap_fallback"] is True
    assert packed[-1]["short_tail_adjustment_mode"] == "safe_token_overlap_fallback"
    assert packed[-1]["short_tail_context_added_tokens"] == 31


def test_final_chunk_at_tail_minimum_and_single_short_stream_are_unchanged() -> None:
    """정확히 51인 tail과 이전 청크가 없는 짧은 stream은 그대로 둔다."""
    first = "가" * 500
    exact_tail = "나" * 51
    exact_spans = align_kss_sentences(
        first + exact_tail,
        [first, exact_tail],
        boundary_id="source-001:section:0:paragraph:1",
    )
    single_spans = align_kss_sentences(
        "다" * 20,
        ["다" * 20],
        boundary_id="source-001:section:0:paragraph:2",
    )

    exact = pack_sentence_spans(exact_spans, codec=CODEC, config=CONFIG)
    single = pack_sentence_spans(single_spans, codec=CODEC, config=CONFIG)

    assert [len(CODEC.encode(part["text"])) for part in exact] == [500, 51]
    assert all(part["short_tail_adjusted"] is False for part in exact)
    assert len(single) == 1
    assert single[0]["text"] == "다" * 20
    assert single[0]["short_tail_adjusted"] is False


def test_oversized_single_sentence_uses_token_fallback_without_loss() -> None:
    """512를 넘는 단일 문장만 토큰 fallback으로 51 overlap 분할한다."""
    source = "".join(chr(0x4E00 + index) for index in range(1_100))
    spans = align_kss_sentences(
        source,
        [source],
        boundary_id="source-001:section:0:paragraph:1",
    )

    packed = pack_sentence_spans(spans, codec=CODEC, config=CONFIG)

    assert len(packed) == 3
    assert all(len(CODEC.encode(part["text"])) <= 512 for part in packed)
    assert all(part["oversized_sentence_fallback"] is True for part in packed)
    assert packed[1]["text"].startswith(packed[0]["text"][-51:])
    assert packed[2]["text"].startswith(packed[1]["text"][-51:])
    rebuilt = packed[0]["text"] + "".join(
        part["text"][part["overlap_actual_tokens"] :] for part in packed[1:]
    )
    assert rebuilt == source


def test_markdown_table_splits_by_whole_rows_and_repeats_header() -> None:
    """큰 표는 Markdown 행 단위로 나누고 모든 후속 조각에 헤더를 반복한다."""
    header = ["| 구분 | 내용 |", "| --- | --- |"]
    rows = [f"| 행{i:02d} | {'가' * 20} |" for i in range(40)]
    markdown = "\n".join([*header, *rows])
    document = make_document()
    block = make_table_block(1, markdown)

    chunks = chunk_advanced_table_block(
        document,
        block,
        codec=CODEC,
        config=CONFIG,
    )

    assert len(chunks) > 1
    assert all(chunk["content_type"] == "table" for chunk in chunks)
    assert all(chunk["table_id"] == block["table_id"] for chunk in chunks)
    assert all(chunk["bm25_tokens"] == [] for chunk in chunks)
    assert all(
        chunk["embedding_text_normalization"] == "preserve_markdown_newlines"
        for chunk in chunks
    )
    assert all("\n" in chunk["embedding_text"] for chunk in chunks)
    assert all(len(CODEC.encode(chunk["embedding_text"])) <= 512 for chunk in chunks)
    assert chunks[0]["table_header_repeated"] is False
    assert all(chunk["table_header_repeated"] is True for chunk in chunks[1:])
    assert [chunk["table_part_index"] for chunk in chunks] == list(
        range(1, len(chunks) + 1)
    )
    assert all(chunk["table_part_count"] == len(chunks) for chunk in chunks)

    rebuilt_rows: list[str] = []
    for chunk in chunks:
        lines = chunk["embedding_text"].splitlines()
        assert lines[:2] == header
        rebuilt_rows.extend(lines[2:])
    assert rebuilt_rows == rows


def test_oversized_table_row_with_multiline_context_stays_within_budget() -> None:
    """표 제목이 포함된 긴 행 fallback은 헤더를 중복 삽입하지 않는다."""
    markdown = "\n".join(
        [
            "요구사항 상세 표",
            "| 구분 | 내용 |",
            "| --- | --- |",
            f"| 기능 | {'가' * 900} |",
        ]
    )
    document = make_document()
    block = make_table_block(1, markdown)

    chunks = chunk_advanced_table_block(
        document,
        block,
        codec=CODEC,
        config=CONFIG,
    )

    assert len(chunks) > 1
    assert all(len(CODEC.encode(chunk["embedding_text"])) <= 512 for chunk in chunks)
    assert all(
        chunk["embedding_text"].count("요구사항 상세 표") == 1 for chunk in chunks
    )
    assert all(
        "oversized_table_row_split" in chunk["quality_flags"] for chunk in chunks
    )


def test_embedding_text_has_no_document_location_or_type_prefix() -> None:
    """문서·위치·유형 메타데이터는 벡터 입력 문자열에 붙이지 않는다."""
    document = make_document()
    document["project_name"] = "META_ONLY_SENTINEL_9X 사업"
    text = make_text_block(1, "접두사 없이 임베딩할 본문입니다.")
    table_markdown = "| 구분 | 내용 |\n| --- | --- |\n| 기간 | 90일 |"
    table = make_table_block(2, table_markdown)

    result = run_corpus([document], [text, table])
    text_chunk, table_chunk = result.chunks

    assert text_chunk["embedding_text"] == text["text"]
    assert table_chunk["embedding_text"] == table_markdown
    for chunk in result.chunks:
        assert chunk["embedding_prefix_included"] is False
        assert "META_ONLY_SENTINEL_9X" not in chunk["embedding_text"]
        assert all(
            prefix not in chunk["embedding_text"]
            for prefix in ("[문서]", "[위치]", "[유형]")
        )


def test_source_and_location_metadata_are_preserved_outside_embedding_text() -> None:
    """검색 필터와 인용에 필요한 문서·위치·표 메타데이터를 보존한다."""
    document = make_document()
    text = make_text_block(1, "메타데이터를 보존할 본문입니다.", para_idx=7)
    table = make_table_block(
        2,
        "| 항목 | 값 |\n| --- | --- |\n| 기간 | 90일 |",
        para_idx=8,
    )

    result = run_corpus([document], [text, table])
    text_chunk, table_chunk = result.chunks

    for chunk in result.chunks:
        assert chunk["source_id"] == document["source_id"]
        assert chunk["document_id"] == document["document_id"]
        assert chunk["source_sha256"] == document["source_sha256"]
        assert chunk["source_filename"] == document["source_filename"]
        assert chunk["project_name"] == document["project_name"]
        assert chunk["issuer"] == document["issuer"]
        assert chunk["notice_number"] == document["notice_number"]
    assert text_chunk["source_block_ids"] == [text["block_id"]]
    assert text_chunk["kss_boundary_id"] == text["kss_boundary_id"]
    assert text_chunk["stream_char_start"] == 0
    assert text_chunk["stream_char_end"] == len(text["text"])
    assert text_chunk["token_count_basis"] == "embedding_text"
    assert text_chunk["overlap_token_basis"] == "normalized_embedding_text"
    assert text_chunk["bm25_source_field"] == "embedding_text"
    assert text_chunk["section_idx_start"] == 0
    assert text_chunk["para_idx_start"] == 7
    assert table_chunk["source_block_ids"] == [table["block_id"]]
    assert table_chunk["table_id"] == table["table_id"]
    assert table_chunk["para_idx_start"] == 8


def test_corpus_output_is_deterministic_and_input_order_independent() -> None:
    """입력 순서와 실행 시각이 청크 ID·순서·요약을 바꾸지 않는다."""
    document_a = make_document("source-a")
    document_b = make_document("source-b")
    block_a = make_text_block(1, "A 문서의 본문입니다.", source_id="source-a")
    block_b = make_text_block(1, "B 문서의 본문입니다.", source_id="source-b")

    first = run_corpus([document_b, document_a], [block_b, block_a])
    second = run_corpus([document_a, document_b], [block_a, block_b])

    assert first.chunks == second.chunks
    assert first.summary == second.summary
    assert [chunk["source_id"] for chunk in first.chunks] == [
        "source-a",
        "source-b",
    ]
    assert all(":ADV2:A512O51:C" in chunk["chunk_id"] for chunk in first.chunks)
    assert all(chunk["schema_version"] == SCHEMA_VERSION for chunk in first.chunks)
    assert all(chunk["corpus_id"] == "advanced_v2" for chunk in first.chunks)
    assert "started_at" not in first.summary
    assert "finished_at" not in first.summary


def test_validation_accepts_valid_chunks_and_rejects_token_overflow() -> None:
    """검증기는 정상 corpus를 통과시키고 512 초과 변조를 탐지한다."""
    document = make_document()
    block = make_text_block(1, "검증 대상 본문입니다.")
    result = run_corpus([document], [block])

    valid = validate_advanced_chunks(
        [document],
        [block],
        result.chunks,
        codec=CODEC,
        config=CONFIG,
    )
    assert valid["overall_pass"] is True

    tampered = [copy.deepcopy(chunk) for chunk in result.chunks]
    tampered[0]["embedding_text"] = "초" * 513
    tampered[0]["token_count"] = 513
    invalid = validate_advanced_chunks(
        [document],
        [block],
        tampered,
        codec=CODEC,
        config=CONFIG,
    )
    assert invalid["overall_pass"] is False


def test_validation_rejects_cross_document_source_block_links() -> None:
    """다른 문서의 block ID로 바꾼 청크는 위치가 같아도 승인하지 않는다."""
    document_a = make_document("source-a", file_type="pdf")
    document_b = make_document("source-b", file_type="pdf")
    block_a = make_text_block(
        1,
        "A 문서 본문입니다.",
        source_id="source-a",
        file_type="pdf",
        page=1,
    )
    block_b = make_text_block(
        1,
        "B 문서 본문입니다.",
        source_id="source-b",
        file_type="pdf",
        page=1,
    )
    result = run_corpus([document_a, document_b], [block_a, block_b])
    tampered = [copy.deepcopy(chunk) for chunk in result.chunks]
    tampered[0]["source_block_ids"] = [block_b["block_id"]]

    invalid = validate_advanced_chunks(
        [document_a, document_b],
        [block_a, block_b],
        tampered,
        codec=CODEC,
        config=CONFIG,
    )

    assert invalid["overall_pass"] is False
    assert invalid["gates"]["source_block_links_are_valid"] is False


def test_validation_rejects_raw_text_not_found_at_recorded_source_span() -> None:
    """ID만 맞고 본문을 바꾼 청크도 원본 stream 대조에서 탐지한다."""
    document = make_document()
    block = make_text_block(1, "실제 원본 문장입니다.")
    result = run_corpus([document], [block])
    tampered = [copy.deepcopy(result.chunks[0])]
    tampered[0]["raw_text"] = "전혀 다른 문장입니다."
    tampered[0]["embedding_text"] = normalize_text_for_embedding(
        tampered[0]["raw_text"]
    )
    tampered[0]["token_count"] = len(CODEC.encode(tampered[0]["embedding_text"]))

    invalid = validate_advanced_chunks(
        [document],
        [block],
        tampered,
        codec=CODEC,
        config=CONFIG,
    )

    assert invalid["overall_pass"] is False
    assert invalid["gates"]["text_raw_matches_source_stream_span"] is False


def test_validation_rejects_missing_tail_chunks_even_if_block_is_still_covered() -> (
    None
):
    """첫 청크가 block ID를 덮어도 stream 뒷부분 유실은 반드시 실패한다."""
    document = make_document()
    block = make_text_block(1, "가" * 1100)
    result = run_corpus([document], [block])
    assert len(result.chunks) >= 3

    truncated = [copy.deepcopy(result.chunks[0])]
    invalid = validate_advanced_chunks(
        [document],
        [block],
        truncated,
        codec=CODEC,
        config=CONFIG,
    )

    assert invalid["overall_pass"] is False
    assert invalid["gates"]["all_required_dense_text_and_tables_are_covered"] is True
    assert invalid["gates"]["text_streams_are_contiguously_covered"] is False


def test_validation_rejects_table_markdown_not_derived_from_source() -> None:
    """파싱 가능한 Markdown이어도 원본 표와 다른 본문이면 실패한다."""
    document = make_document()
    block = make_table_block(
        1,
        "| 항목 | 값 |\n| --- | --- |\n| 기간 | 90일 |",
    )
    result = run_corpus([document], [block])
    tampered = [copy.deepcopy(result.chunks[0])]
    replacement = "| X |\n| --- |\n| 완전히 다른 값 |"
    tampered[0]["raw_text"] = replacement
    tampered[0]["embedding_text"] = replacement
    tampered[0]["token_count"] = len(CODEC.encode(replacement))

    invalid = validate_advanced_chunks(
        [document],
        [block],
        tampered,
        codec=CODEC,
        config=CONFIG,
    )

    assert invalid["overall_pass"] is False
    assert invalid["gates"]["table_chunks_match_source_markdown"] is False
