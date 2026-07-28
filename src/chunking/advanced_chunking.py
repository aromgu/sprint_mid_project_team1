"""Advanced RAG용 KSS·표·Kiwi 청크를 만든다.

이 모듈은 ``rfp_advanced_preprocessing_v1`` 블록만 입력으로 받는다.
Naive RAG의 Recursive 512/102 코드와 결과를 변경하지 않고, 다음 계약을
별도 스키마와 전략 ID로 구현한다.

* 일반 텍스트만 KSS로 문장을 찾고 최대 512 tiktoken 토큰으로 묶는다.
* PDF page와 HWP/HWPX section+paragraph 경계를 절대 넘지 않는다.
* 온전한 문장 suffix를 최대 51토큰까지 반복한다. 정확히 51을 만들기 위해
  정상 문장을 자르지 않고, 512를 넘는 단일 문장만 토큰 fallback을 쓴다.
* 여러 청크 중 마지막 조각의 신규 내용이 51토큰 미만이면 중복 없는 병합,
  온전한 문장 재배치, 안전한 토큰 overlap 순서로 보정한다.
* PDF 페이지 끝의 ``- 123 -`` 표식은 임베딩 본문에서 항상 제외하고,
  원본 블록과 실제 PDF ``page`` metadata로만 보존한다.
* 표는 KSS·Kiwi에서 제외하고 ``table_markdown``만 Dense 본문으로 쓴다.
* 표가 크면 행 경계로 나누고 뒤 part에 헤더를 반복한다.
* 일반 텍스트 청크만 Kiwi 형태소에서 조사(J*)·어미(E*)를 제외해 별도
  ``bm25_tokens``에 저장한다.
* 파일명·위치·유형은 metadata로 보존하되 임베딩 본문에 prefix로 붙이지 않는다.

KSS는 내부 공백을 정규화할 수 있다. 따라서 KSS 반환 문자열을 저장하지 않고
공백을 제외한 문자 열을 원문과 대조해 문장 끝 위치만 찾은 뒤, 최종 본문은
항상 원문의 문자 span에서 다시 자른다.
"""

from __future__ import annotations

import importlib.metadata
import re
import statistics
import unicodedata
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.chunking.split_text import (
    TiktokenCodec,
    TokenCodec,
    TokenTextMap,
    escape_markdown_fallback_cell,
    parse_markdown_table_segments,
    segment_header_text,
    unique_in_order,
)

SCHEMA_VERSION = "rfp_advanced_chunk_v2"
STRATEGY_ID = (
    "advanced_kss_kiwi_exclude_je_semantic_tail_page_marker_no_text_newline_"
    "cl100k_base_512_51_v2"
)
CORPUS_ID = "advanced_v2"
INPUT_SCHEMA_VERSION = "rfp_advanced_preprocessing_v1"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 51
DEFAULT_MIN_TAIL_TOKENS = DEFAULT_OVERLAP_TOKENS
EXPECTED_KSS_VERSION = "6.0.6"
EXPECTED_KIWI_VERSION = "0.23.2"
INDEXABLE_POLICIES = frozenset({"index", "flatten"})
KSS_ALIGNMENT_DECORATIVE_CHARS = frozenset("⦁•◦▪▫□■○●※◆◇▶▷▸▹►▻‣⁃·∙⋅")
PAGE_MARKER_DETECTOR_ID = "pdf_trailing_dash_number_ascii_v1"
TEXT_EMBEDDING_NORMALIZATION_ID = "line_separators_to_spaces_preserve_offsets_v1"
TEXT_LINE_SEPARATOR_CHARS = frozenset(
    {"\n", "\r", "\v", "\f", "\x85", "\u2028", "\u2029"}
)

FORBIDDEN_IMAGE_PAYLOAD = re.compile(
    r"(?:data\s*:\s*image|base64\s*,|\"(?:payload|image_bytes|binary_payload)\"\s*:)",
    re.IGNORECASE,
)
HTML_TABLE_TAG = re.compile(r"</?(?:table|thead|tbody|tr|th|td)\b", re.IGNORECASE)
METADATA_PREFIX = re.compile(r"^\[문서\].*\n\[위치\].*\n\[유형\]", re.DOTALL)
PAGE_MARKER_ATOM = re.compile(r"- *([1-9][0-9]{0,2}) *-")
PAGE_MARKER_ONLY = re.compile(r"(?:- *[1-9][0-9]{0,2} *-)(?: *- *[1-9][0-9]{0,2} *-)*")
TEXT_LINE_SEPARATOR_TRANSLATION = str.maketrans(
    {char: " " for char in TEXT_LINE_SEPARATOR_CHARS}
)

# 팀 합의 코드를 그대로 재현한다. Kiwi 결과 중 조사(J*)와 어미(E*)만
# 제외하며, 접사·감탄사·기호·사용자 정의 태그 등 나머지는 임의로 버리지 않는다.
BM25_POS_POLICY_ID = "exclude_josa_eomi_prefix_v1"
BM25_EXCLUDED_POS_PREFIXES = ("J", "E")
BM25_TOKEN_NORMALIZATION = "strip_casefold"

BUSINESS_METADATA_FIELDS = (
    "source_row",
    "notice_number",
    "notice_round",
    "project_name",
    "issuer",
    "project_amount_won",
    "project_amount_status",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "bid_period_status",
    "metadata_quality_flags",
    "metadata_match_status",
    "metadata_review_status",
    "metadata_correction_fields",
    "metadata_correction_source",
    "metadata_correction_note",
    "metadata_verified_by",
)


class SentenceSplitter(Protocol):
    """KSS와 테스트 대역이 따라야 하는 최소 호출 계약이다."""

    def __call__(self, text: str) -> Sequence[str]:
        """입력 한 경계의 문장 문자열을 순서대로 반환한다."""


class Bm25Tokenizer(Protocol):
    """Kiwi와 테스트 대역이 따라야 하는 최소 토큰화 계약이다."""

    def tokenize(self, text: str) -> Sequence[Any]:
        """입력 본문의 형태소 또는 최종 토큰 문자열을 반환한다."""


@dataclass(frozen=True, slots=True)
class AdvancedChunkConfig:
    """Advanced 512/51 실험을 재현하기 위한 설정이다."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tail_tokens: int = DEFAULT_MIN_TAIL_TOKENS
    model_name: str = DEFAULT_MODEL
    encoding_name: str = DEFAULT_ENCODING
    strategy_id: str = STRATEGY_ID


@dataclass(frozen=True, slots=True)
class AdvancedTextStream:
    """표·이미지를 넘지 않는 하나의 PDF page/HWP paragraph 텍스트다."""

    stream_id: str
    stream_order: int
    boundary_type: str
    boundary_id: str
    blocks: tuple[dict[str, Any], ...]
    text: str
    block_char_spans: tuple[tuple[int, int, dict[str, Any]], ...]


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """KSS 경계와 정확히 대응하는 원문 문자 구간이다."""

    boundary_id: str
    sentence_index: int
    char_start: int
    char_end: int
    normalized_text: str
    raw_text: str

    @property
    def text(self) -> str:
        """packing 코드가 읽는 정확한 원문 문자열을 반환한다."""
        return self.raw_text


@dataclass(frozen=True, slots=True)
class PackedTextChunk:
    """문장 packing 또는 긴 문장 fallback이 만든 텍스트 조각이다."""

    text: str
    char_start: int
    char_end: int
    sentence_start: int
    sentence_end: int
    overlap_target_tokens: int
    overlap_actual_tokens: int
    overlap_sentence_count: int
    oversized_sentence_split: bool
    sentence_fragment_index: int | None = None
    sentence_fragment_count: int | None = None
    sentence_token_start: int | None = None
    sentence_token_end: int | None = None
    short_tail_adjusted: bool = False
    short_tail_token_overlap_fallback: bool = False
    short_tail_adjustment_mode: str = "none"
    short_tail_original_new_token_count: int | None = None
    short_tail_context_added_tokens: int = 0

    @property
    def oversized_sentence_fallback(self) -> bool:
        """테스트·보고서에서 쓰는 긴 문장 fallback 별칭이다."""
        return self.oversized_sentence_split

    def __getitem__(self, key: str) -> Any:
        """JSON 레코드처럼 필드명을 사용해 읽을 수 있게 한다."""
        if key == "oversized_sentence_fallback":
            return self.oversized_sentence_fallback
        return getattr(self, key)


@dataclass(frozen=True, slots=True)
class AdvancedChunkingResult:
    """검증을 통과한 청크와 결정적 요약을 함께 반환한다."""

    chunks: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class KssAlignmentError(ValueError):
    """KSS 결과를 원문 문자 span과 손실 없이 맞출 수 없을 때 발생한다."""


class KssSentenceSplitter:
    """KSS 6.0.6 pecab 단일 worker 설정을 고정한 운영 wrapper다."""

    backend = "pecab"
    num_workers = 1

    def __init__(self) -> None:
        """설치 버전을 확인하고 문장 분리 함수를 한 번만 준비한다."""
        version = importlib.metadata.version("kss")
        if version != EXPECTED_KSS_VERSION:
            raise RuntimeError(
                f"kss 버전이 다릅니다: {version} (기대값: {EXPECTED_KSS_VERSION})"
            )
        try:
            import kss
        except ImportError as error:  # pragma: no cover - 설치 안내 경로
            raise RuntimeError("kss가 없습니다. uv sync를 실행하세요.") from error
        self.version = version
        self._split = kss.Kss("split_sentences")

    def __call__(self, text: str) -> Sequence[str]:
        """결정적인 pecab·단일 worker 설정으로 문장을 반환한다."""
        # pecab이 일부 긴 HWP 문장에서 numpy 정수 overflow 경고를 내지만
        # 문장 결과는 반환한다. 이 알려진 경고만 실행 로그에서 숨긴다.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in scalar add",
                category=RuntimeWarning,
                module=r"pecab\..*",
            )
            result = self._split(
                text,
                backend=self.backend,
                num_workers=self.num_workers,
                strip=True,
            )
        if not isinstance(result, list) or any(
            not isinstance(sentence, str) for sentence in result
        ):
            raise TypeError("KSS가 문자열 목록이 아닌 결과를 반환했습니다")
        return result


class KiwiBm25Tokenizer:
    """Kiwi 0.23.2 형태소 중 BM25에 사용할 어휘만 고르는 wrapper다."""

    def __init__(self) -> None:
        """설치 버전을 고정하고 Kiwi 분석기를 한 번만 생성한다."""
        version = importlib.metadata.version("kiwipiepy")
        if version != EXPECTED_KIWI_VERSION:
            raise RuntimeError(
                "kiwipiepy 버전이 다릅니다: "
                f"{version} (기대값: {EXPECTED_KIWI_VERSION})"
            )
        try:
            from kiwipiepy import Kiwi
        except ImportError as error:  # pragma: no cover - 설치 안내 경로
            raise RuntimeError("kiwipiepy가 없습니다. uv sync를 실행하세요.") from error
        self.version = version
        self._kiwi = Kiwi()

    def tokenize(self, text: str) -> Sequence[Any]:
        """Kiwi 원시 형태소를 반환하고 POS 필터는 공통 함수가 적용한다."""
        return self._kiwi.tokenize(text)


def validate_advanced_config(config: AdvancedChunkConfig) -> None:
    """토큰 상한과 overlap이 앞으로 진행 가능한 값인지 확인한다."""
    if config.max_tokens <= 0:
        raise ValueError("max_tokens는 양수여야 합니다")
    if not 0 <= config.overlap_tokens < config.max_tokens:
        raise ValueError("overlap_tokens는 0 이상이고 max_tokens보다 작아야 합니다")
    if not 0 <= config.min_tail_tokens <= config.max_tokens:
        raise ValueError("min_tail_tokens는 0 이상이고 max_tokens 이하여야 합니다")
    if not config.model_name or not config.encoding_name or not config.strategy_id:
        raise ValueError("모델·인코딩·전략 ID는 비어 있을 수 없습니다")


def _is_kss_alignment_ignorable(character: str) -> bool:
    """KSS/pecab이 버리는 공백·한컴 표시 글리프·형식 문자를 판별한다."""
    return (
        character.isspace()
        or character in KSS_ALIGNMENT_DECORATIVE_CHARS
        or unicodedata.category(character) in {"Co", "Cf"}
    )


def _alignment_key(value: str) -> str:
    """KSS 경계 대조용으로 의미 문자는 유지하고 표시용 문자만 제거한다."""
    return "".join(
        character for character in value if not _is_kss_alignment_ignorable(character)
    )


def _sanitize_kss_input(value: str) -> tuple[str, int]:
    """pecab 오해를 막도록 표시용 문자만 공백으로 바꿔 KSS에 전달한다.

    한컴 Private Use Area 글리프와 장식 bullet·형식 문자는 문장 의미가 아닌
    표시 정보다. 삭제 대신 공백으로 치환해 양옆 단어가 붙지 않게 하며, 최종
    청크는 이 문자열이 아니라 원문 span에서 만들기 때문에 원문은 보존된다.
    """
    sanitized: list[str] = []
    replaced_count = 0
    for character in value:
        # alignment에서 무시하는 표시 문자 정의를 한곳에서 공유하되,
        # 정상 공백·개행은 KSS 입력에서도 그대로 유지한다.
        should_replace = (
            _is_kss_alignment_ignorable(character) and not character.isspace()
        )
        sanitized.append(" " if should_replace else character)
        replaced_count += int(should_replace)
    return "".join(sanitized), replaced_count


def align_kss_sentences(
    source_text: str,
    kss_sentences: Sequence[str],
    *,
    boundary_id: str = "",
    context: str = "",
) -> list[SentenceSpan]:
    """KSS 문장을 원문 span으로 바꾸고 공백·개행·탭을 정확히 보존한다.

    KSS가 내부 공백을 정리해도 공백을 제외한 문자 열이 같으면 경계로 사용할
    수 있다. 문장 사이 공백은 뒤 문장의 span에 포함하고, 마지막 문장은 원문
    끝의 공백까지 포함해 모든 span을 합치면 원문과 정확히 같게 만든다.
    """
    label = context or boundary_id
    if not source_text:
        return []
    sentences = [str(sentence) for sentence in kss_sentences if str(sentence)]
    if not sentences:
        raise KssAlignmentError(f"KSS가 비어 있는 결과를 반환했습니다: {label}")

    source_compact = _alignment_key(source_text)
    sentence_compacts = [_alignment_key(sentence) for sentence in sentences]
    if any(not value for value in sentence_compacts):
        raise KssAlignmentError(f"KSS에 공백뿐인 문장이 포함됐습니다: {label}")
    if "".join(sentence_compacts) != source_compact:
        raise KssAlignmentError(
            "KSS 결과의 비공백 문자가 원문과 다릅니다: "
            f"{label}; source={source_text[:120]!r}; kss={sentences[:3]!r}"
        )

    # compact 문자 n번째가 원문 어디에 있었는지 기록한다. 전처리 입력은 NFC로
    # 정규화됐으므로 여기서는 공백 외 문자를 변경하지 않는다.
    source_positions = [
        index
        for index, character in enumerate(source_text)
        if not _is_kss_alignment_ignorable(character)
    ]
    spans: list[SentenceSpan] = []
    compact_cursor = 0
    source_cursor = 0
    for sentence_index, compact_sentence in enumerate(sentence_compacts, start=1):
        compact_cursor += len(compact_sentence)
        if compact_cursor > len(source_positions):
            raise KssAlignmentError(f"KSS 문자 위치가 원문을 넘었습니다: {label}")
        end = source_positions[compact_cursor - 1] + 1
        if sentence_index == len(sentence_compacts):
            end = len(source_text)
        exact_text = source_text[source_cursor:end]
        spans.append(
            SentenceSpan(
                boundary_id=boundary_id or context,
                sentence_index=sentence_index,
                char_start=source_cursor,
                char_end=end,
                normalized_text=sentences[sentence_index - 1],
                raw_text=exact_text,
            )
        )
        source_cursor = end

    if "".join(span.text for span in spans) != source_text:
        raise KssAlignmentError(f"KSS 원문 span 복원에 실패했습니다: {label}")
    return spans


def _join_text_blocks(
    blocks: Sequence[dict[str, Any]],
) -> tuple[str, tuple[tuple[int, int, dict[str, Any]], ...]]:
    """한 경계의 일반 텍스트를 잇고 원본 블록 문자 구간을 기록한다."""
    parts: list[str] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for block_index, block in enumerate(blocks):
        if block_index:
            parts.append("\n\n")
            cursor += 2
        text = str(block.get("text") or "")
        if FORBIDDEN_IMAGE_PAYLOAD.search(text):
            raise ValueError(
                f"일반 텍스트에 이미지 payload가 있습니다: {block['block_id']}"
            )
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, block))
    return "".join(parts), tuple(spans)


def normalize_text_for_embedding(value: str) -> str:
    """일반 텍스트의 줄 구분자를 같은 길이의 공백으로 바꾼다.

    줄바꿈 문자를 단순 삭제하면 ``사업기간``처럼 서로 다른 줄의 단어가
    붙으므로 공백으로 치환한다. 문자 수를 유지해 KSS 원문 span과 임베딩
    토큰 budget이 같은 문자 offset을 공유하게 한다. 표는 이 함수를 거치지
    않아 Markdown 행 구분을 유지하고, 원문은 ``raw_text``에 보존한다.
    """
    return value.translate(TEXT_LINE_SEPARATOR_TRANSLATION)


def _is_advanced_text(block: Mapping[str, Any]) -> bool:
    """KSS·Dense·BM25 계약을 모두 만족하는 일반 텍스트인지 확인한다."""
    return (
        block.get("content_type") == "text"
        and block.get("index_policy") in INDEXABLE_POLICIES
        and block.get("dense_eligible") is True
        and block.get("kss_eligible") is True
        and block.get("bm25_eligible") is True
        and block.get("vectorize_field") == "text"
        and bool(str(block.get("text") or "").strip())
    )


def _is_advanced_table(block: Mapping[str, Any]) -> bool:
    """Markdown Dense 대상이고 KSS·BM25가 꺼진 표인지 확인한다."""
    return (
        block.get("content_type") == "table"
        and block.get("index_policy") in INDEXABLE_POLICIES
        and block.get("dense_eligible") is True
        and block.get("kss_eligible") is False
        and block.get("bm25_eligible") is False
        and block.get("vectorize_field") == "table_markdown"
        and bool(str(block.get("table_markdown") or "").strip())
    )


def _normalize_page_marker_key(value: str) -> str:
    """페이지 표식 판정용 복사본만 dash·공백·제어 문자를 정규화한다.

    NFKC는 ``①``·``²``·전각 숫자를 ASCII 숫자로 바꿔 오탐을 만들 수 있어
    사용하지 않는다. 최종 ``raw_text``와 전처리 블록 원문은 바꾸지 않는다.
    """
    normalized: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category == "Pd" or char == "\u2212":
            normalized.append("-")
        elif category == "Cf":
            continue
        elif category == "Co" or char.isspace():
            normalized.append(" ")
        else:
            normalized.append(char)
    return " ".join("".join(normalized).split())


def extract_page_marker_numbers(value: str) -> tuple[int, ...] | None:
    """``- 130 - - 131 -``처럼 페이지 표식만 있는 문자열을 판별한다.

    ASCII 숫자 1~3자리만 허용하고, 여러 번호는 반드시 1씩 증가해야 한다.
    따라서 연도·전화번호·금액·요구사항 번호 같은 일반 텍스트는 제외되지
    않는다. PDF 위치 근거는 ``_find_pdf_page_marker_blocks``에서 추가한다.
    """
    key = _normalize_page_marker_key(value)
    if PAGE_MARKER_ONLY.fullmatch(key) is None:
        return None
    numbers = tuple(int(match.group(1)) for match in PAGE_MARKER_ATOM.finditer(key))
    if not numbers or any(
        right != left + 1 for left, right in zip(numbers, numbers[1:])
    ):
        return None
    return numbers


def _find_pdf_page_marker_blocks(
    document: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """PDF 각 물리 페이지 끝에 연속된 페이지 표식 블록만 찾는다.

    문자열 모양만으로 HWP 본문을 제거하지 않으며, PDF의 유효한 ``page``와
    페이지 끝이라는 구조적 근거가 모두 있을 때만 metadata-only 후보가 된다.
    제외 이미지 같은 비색인 블록은 footer 뒤에 있어도 판정을 방해하지 않는다.
    """
    if str(document.get("file_type") or "").casefold() != "pdf":
        return {}

    by_page: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for block in sorted(blocks, key=lambda item: int(item["block_order"])):
        page = block.get("page")
        if isinstance(page, int) and page >= 1:
            by_page[page].append(block)

    detected: dict[str, tuple[int, ...]] = {}
    for page_blocks in by_page.values():
        for block in reversed(page_blocks):
            numbers = (
                extract_page_marker_numbers(str(block.get("text") or ""))
                if _is_advanced_text(block)
                else None
            )
            if numbers is not None:
                detected[str(block["block_id"])] = numbers
                continue
            if block.get("dense_eligible") is not True:
                continue
            break
    return detected


def build_advanced_streams(
    document: Mapping[str, Any],
    blocks: Sequence[dict[str, Any]],
) -> list[AdvancedTextStream | dict[str, Any]]:
    """문서 순서를 유지하며 텍스트 stream과 독립 표 stream을 만든다.

    PDF는 같은 page라도 표·이미지가 끼면 stream을 끊는다. 이렇게 해야 표
    앞뒤 문장이 KSS 입력에서 인위적으로 붙지 않으면서 page 경계도 지킨다.
    HWP/HWPX도 같은 paragraph ID 안의 연속 텍스트만 합친다.
    """
    source_id = str(document["source_id"])
    streams: list[AdvancedTextStream | dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []
    current_boundary: str | None = None
    stream_order = 0
    page_marker_blocks = _find_pdf_page_marker_blocks(document, blocks)

    def flush_text() -> None:
        """모아 둔 연속 텍스트를 AdvancedTextStream 하나로 확정한다."""
        nonlocal current_blocks, current_boundary, stream_order
        if not current_blocks:
            return
        stream_order += 1
        text, block_spans = _join_text_blocks(current_blocks)
        anchor = current_blocks[0]
        streams.append(
            AdvancedTextStream(
                stream_id=f"{source_id}:AS{stream_order:06d}",
                stream_order=stream_order,
                boundary_type=str(anchor["kss_boundary_type"]),
                boundary_id=str(anchor["kss_boundary_id"]),
                blocks=tuple(current_blocks),
                text=text,
                block_char_spans=block_spans,
            )
        )
        current_blocks = []
        current_boundary = None

    for block in sorted(blocks, key=lambda item: int(item["block_order"])):
        if str(block.get("source_id")) != source_id:
            raise ValueError("다른 문서의 블록을 같은 stream에 넣을 수 없습니다")
        block_id = str(block["block_id"])
        if block_id in page_marker_blocks:
            # 팀 합의에 따라 합칠 여유가 있어도 footer 문자를 임베딩하지
            # 않는다. 직전 일반 텍스트는 먼저 확정하고 marker는 입력 block의
            # 원문과 실제 PDF page metadata에만 남긴다.
            flush_text()
            continue
        if _is_advanced_text(block):
            boundary = str(block.get("kss_boundary_id") or "")
            if not boundary:
                raise ValueError(
                    f"KSS 대상에 boundary ID가 없습니다: {block['block_id']}"
                )
            if current_blocks and boundary != current_boundary:
                flush_text()
            current_boundary = boundary
            current_blocks.append(block)
            continue

        # 표·이미지·제외 블록은 문장 결합을 끊는다. 색인 대상 표만 독립
        # stream으로 남기고, 이미지 및 exclude 블록은 청크를 만들지 않는다.
        flush_text()
        if _is_advanced_table(block):
            stream_order += 1
            table_stream = dict(block)
            table_stream["stream_id"] = f"{source_id}:AS{stream_order:06d}"
            table_stream["stream_order"] = stream_order
            streams.append(table_stream)
    flush_text()
    return streams


def build_advanced_text_streams(
    document: Mapping[str, Any],
    blocks: Sequence[dict[str, Any]],
) -> list[AdvancedTextStream]:
    """테스트·감사용으로 일반 텍스트 stream만 반환한다."""
    return [
        stream
        for stream in build_advanced_streams(document, blocks)
        if isinstance(stream, AdvancedTextStream)
    ]


def _farthest_safe_end(
    token_map: TokenTextMap,
    start_token: int,
    max_tokens: int,
    *,
    required_next_overlap: int = 0,
) -> int:
    """상한 안에서 UTF-8과 다음 overlap이 모두 안전한 가장 먼 끝을 찾는다."""
    upper = min(len(token_map), start_token + max_tokens)
    for end_token in reversed(token_map.safe_token_indices):
        if end_token > upper or end_token <= start_token:
            continue
        if end_token < len(token_map) and required_next_overlap:
            next_start = end_token - required_next_overlap
            if (
                next_start <= start_token
                or next_start not in token_map.safe_token_index_set
            ):
                continue
        return end_token
    raise ValueError("UTF-8 문자 경계를 지키며 토큰 조각을 만들 수 없습니다")


def _split_oversized_sentence(
    sentence: SentenceSpan,
    budget_text: str,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> list[PackedTextChunk]:
    """512를 넘는 한 문장을 임베딩 기준 토큰 overlap으로 안전하게 나눈다."""
    if len(budget_text) != len(sentence.text):
        raise ValueError("문장 원문과 임베딩 budget 문자열 길이가 다릅니다")
    # 줄 구분자를 같은 길이의 공백으로 바꿨으므로 budget의 char offset을
    # 원문에 그대로 적용해도 raw span이 한 글자도 손실되지 않는다.
    token_map = TokenTextMap(budget_text, codec)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    previous_end = 0
    while start < len(token_map):
        end = _farthest_safe_end(
            token_map,
            start,
            config.max_tokens,
            required_next_overlap=(
                config.overlap_tokens
                if start + config.max_tokens < len(token_map)
                else 0
            ),
        )
        overlap = previous_end - start if ranges else 0
        ranges.append((start, end, overlap))
        if end >= len(token_map):
            break
        next_start = end - config.overlap_tokens
        if next_start <= start:
            raise ValueError("긴 문장 overlap 때문에 앞으로 진행하지 못합니다")
        previous_end = end
        start = next_start

    chunks: list[PackedTextChunk] = []
    for fragment_index, (start, end, overlap) in enumerate(ranges, start=1):
        local_start = token_map.token_to_char[start]
        local_end = token_map.token_to_char[end]
        chunks.append(
            PackedTextChunk(
                text=sentence.text[local_start:local_end],
                char_start=sentence.char_start + local_start,
                char_end=sentence.char_start + local_end,
                sentence_start=sentence.sentence_index,
                sentence_end=sentence.sentence_index,
                overlap_target_tokens=config.overlap_tokens,
                overlap_actual_tokens=overlap,
                overlap_sentence_count=0,
                oversized_sentence_split=True,
                sentence_fragment_index=fragment_index,
                sentence_fragment_count=len(ranges),
                sentence_token_start=start,
                sentence_token_end=end,
            )
        )
    return chunks


def _whole_sentence_overlap_start(
    budget_text: str,
    spans: Sequence[SentenceSpan],
    chunk_start: int,
    chunk_end: int,
    next_sentence_index: int,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> tuple[int, int, int]:
    """다음 문장과 함께 들어가는 51토큰 이하 최대 suffix를 선택한다."""
    if config.overlap_tokens <= 0 or chunk_end - chunk_start <= 1:
        return chunk_end, 0, 0
    next_sentence = spans[next_sentence_index]
    selected_start = chunk_end
    selected_tokens = 0
    selected_sentences = 0
    # 뒤에서 한 문장씩 늘리되 target을 넘으면 더 앞 suffix도 넘으므로 끝낸다.
    for candidate_start in range(chunk_end - 1, chunk_start, -1):
        suffix = budget_text[
            spans[candidate_start].char_start : spans[chunk_end - 1].char_end
        ]
        suffix_tokens = len(codec.encode(suffix))
        if suffix_tokens > config.overlap_tokens:
            break
        with_next = budget_text[
            spans[candidate_start].char_start : next_sentence.char_end
        ]
        if len(codec.encode(with_next)) > config.max_tokens:
            continue
        selected_start = candidate_start
        selected_tokens = suffix_tokens
        selected_sentences = chunk_end - candidate_start
    return selected_start, selected_tokens, selected_sentences


def _repair_short_final_chunk(
    packed: list[PackedTextChunk],
    stream_text: str,
    budget_text: str,
    spans: Sequence[SentenceSpan],
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> list[PackedTextChunk]:
    """마지막 조각의 *신규 내용*을 기준으로 의미 있는 tail을 만든다.

    최종 청크에 기존 overlap이 48토큰 있고 새 정보가 3토큰뿐이면 전체
    51토큰이어도 짧은 tail이다. 처리 순서는 중복 없는 병합 → 온전한 문장
    재배치 → 안전 token overlap이다. PDF 페이지 표식은 stream 생성 전에
    제거되므로 이 함수에 들어오지 않는다.
    """
    if len(packed) < 2 or config.min_tail_tokens <= 0:
        return packed

    previous = packed[-2]
    final = packed[-1]
    new_char_start = max(previous.char_end, final.char_start)
    new_budget_text = budget_text[new_char_start : final.char_end]
    new_tokens = len(codec.encode(new_budget_text))
    if new_tokens >= config.min_tail_tokens:
        return packed

    # 1순위: overlap 중복을 제거한 연속 원문이 512 이하면 청크 하나로 합친다.
    merged_text = stream_text[previous.char_start : final.char_end]
    merged_budget_text = budget_text[previous.char_start : final.char_end]
    merged_tokens = len(codec.encode(merged_budget_text))
    if (
        not previous.oversized_sentence_split
        and not final.oversized_sentence_split
        and merged_text.strip()
        and merged_tokens <= config.max_tokens
    ):
        merged = replace(
            previous,
            text=merged_text,
            char_end=final.char_end,
            sentence_end=max(previous.sentence_end, final.sentence_end),
            short_tail_adjusted=True,
            short_tail_token_overlap_fallback=False,
            short_tail_adjustment_mode="merged_with_previous",
            short_tail_original_new_token_count=new_tokens,
            short_tail_context_added_tokens=0,
        )
        return [*packed[:-2], merged]

    # 2순위: 이전 청크에서 가장 적은 수의 온전한 문장을 마지막으로 옮긴다.
    candidate_spans = (
        [
            span
            for span in spans
            if previous.char_start < span.char_start < final.char_start
        ]
        if not previous.oversized_sentence_split and not final.oversized_sentence_split
        else []
    )
    for candidate in reversed(candidate_spans):
        left_text = stream_text[previous.char_start : candidate.char_start]
        right_text = stream_text[candidate.char_start : final.char_end]
        left_budget_text = budget_text[previous.char_start : candidate.char_start]
        right_budget_text = budget_text[candidate.char_start : final.char_end]
        left_tokens = len(codec.encode(left_budget_text))
        right_tokens = len(codec.encode(right_budget_text))
        left_unique_start = (
            max(previous.char_start, packed[-3].char_end)
            if len(packed) >= 3
            else previous.char_start
        )
        left_unique_text = stream_text[left_unique_start : candidate.char_start]
        if not (
            left_text.strip()
            and left_unique_text.strip()
            and config.min_tail_tokens <= right_tokens <= config.max_tokens
            and left_tokens <= config.max_tokens
        ):
            continue
        packed[-2] = replace(
            previous,
            text=left_text,
            char_end=candidate.char_start,
            sentence_end=candidate.sentence_index - 1,
        )
        packed[-1] = replace(
            final,
            text=right_text,
            char_start=candidate.char_start,
            sentence_start=candidate.sentence_index,
            overlap_actual_tokens=0,
            overlap_sentence_count=0,
            sentence_fragment_index=None,
            sentence_fragment_count=None,
            sentence_token_start=None,
            sentence_token_end=None,
            short_tail_adjusted=True,
            short_tail_token_overlap_fallback=False,
            short_tail_adjustment_mode="whole_sentence_rebalance",
            short_tail_original_new_token_count=new_tokens,
            short_tail_context_added_tokens=0,
        )
        return packed

    # 이미 온전한 문장 overlap이 충분한 문맥을 제공한다면 원문을 더 복사하지
    # 않는다. 신규 정보가 의미 있는 경우 이 상태 자체가 정상적인 3순위다.
    final_tokens = len(codec.encode(budget_text[final.char_start : final.char_end]))
    if final_tokens >= config.min_tail_tokens and final.overlap_actual_tokens > 0:
        packed[-1] = replace(
            final,
            short_tail_adjusted=True,
            short_tail_token_overlap_fallback=False,
            short_tail_adjustment_mode="existing_overlap_context",
            short_tail_original_new_token_count=new_tokens,
            short_tail_context_added_tokens=final.overlap_actual_tokens,
        )
        return packed

    # 3순위: 500토큰짜리 한 문장 뒤에 20토큰 문장이 남으면 온전한 문장을
    # 이동할 수 없다. 이때만 이전 문장의 최소 suffix를 문맥 overlap으로 쓴다.
    context_text = stream_text[previous.char_start : final.char_start]
    context_budget_text = budget_text[previous.char_start : final.char_start]
    if not context_text:
        raise ValueError("짧은 마지막 청크에 붙일 이전 문맥이 없습니다")
    context_map = TokenTextMap(context_budget_text, codec)
    for token_start in sorted(context_map.safe_token_indices, reverse=True):
        if token_start >= len(context_map):
            continue
        new_char_start = previous.char_start + context_map.token_to_char[token_start]
        expanded_text = stream_text[new_char_start : final.char_end]
        expanded_budget_text = budget_text[new_char_start : final.char_end]
        expanded_tokens = len(codec.encode(expanded_budget_text))
        overlap_tokens = len(
            codec.encode(budget_text[new_char_start : previous.char_end])
        )
        if not (
            config.min_tail_tokens <= expanded_tokens <= config.max_tokens
            and overlap_tokens <= config.overlap_tokens
        ):
            continue
        packed[-1] = replace(
            final,
            text=expanded_text,
            char_start=new_char_start,
            sentence_start=previous.sentence_end,
            overlap_actual_tokens=overlap_tokens,
            overlap_sentence_count=0,
            sentence_fragment_index=None,
            sentence_fragment_count=None,
            sentence_token_start=None,
            sentence_token_end=None,
            short_tail_adjusted=True,
            short_tail_token_overlap_fallback=True,
            short_tail_adjustment_mode="safe_token_overlap_fallback",
            short_tail_original_new_token_count=new_tokens,
            short_tail_context_added_tokens=overlap_tokens,
        )
        return packed

    raise ValueError("512토큰 상한을 지키면서 짧은 마지막 청크를 보정하지 못했습니다")


def pack_sentence_spans(
    stream_text: str | Sequence[SentenceSpan],
    sentences: Sequence[SentenceSpan] | None = None,
    codec: TokenCodec | None = None,
    config: AdvancedChunkConfig | None = None,
) -> list[PackedTextChunk]:
    """KSS 문장을 512 안에 묶고 가능한 온전한 문장 suffix를 반복한다."""
    selected_config = config or AdvancedChunkConfig()
    if codec is None:
        raise ValueError("문장 packing에 TokenCodec이 필요합니다")
    if isinstance(stream_text, str):
        if sentences is None:
            raise ValueError("stream_text와 함께 SentenceSpan 목록이 필요합니다")
        selected_text = stream_text
        selected_sentences = list(sentences)
    else:
        if sentences is not None:
            raise ValueError("SentenceSpan 목록을 두 번 전달할 수 없습니다")
        selected_sentences = list(stream_text)
        selected_text = "".join(sentence.text for sentence in selected_sentences)

    validate_advanced_config(selected_config)
    if "".join(sentence.text for sentence in selected_sentences) != selected_text:
        raise ValueError("문장 span을 합친 결과가 stream 원문과 다릅니다")
    selected_budget_text = normalize_text_for_embedding(selected_text)
    if len(selected_budget_text) != len(selected_text):
        raise ValueError("임베딩 정규화가 원문 문자 offset을 바꿨습니다")
    packed: list[PackedTextChunk] = []
    start_index = 0
    pending_overlap_tokens = 0
    pending_overlap_sentences = 0

    while start_index < len(selected_sentences):
        first = selected_sentences[start_index]
        first_budget_text = selected_budget_text[first.char_start : first.char_end]
        if len(codec.encode(first_budget_text)) > selected_config.max_tokens:
            fragments = _split_oversized_sentence(
                first,
                first_budget_text,
                codec,
                selected_config,
            )
            packed.extend(fragments)
            start_index += 1
            pending_overlap_tokens = 0
            pending_overlap_sentences = 0
            continue

        end_index = start_index
        while end_index < len(selected_sentences):
            candidate = selected_budget_text[
                selected_sentences[start_index].char_start : selected_sentences[
                    end_index
                ].char_end
            ]
            if len(codec.encode(candidate)) > selected_config.max_tokens:
                break
            end_index += 1
        if end_index == start_index:
            raise ValueError("문장 하나를 토큰 상한 안에 넣지 못했습니다")

        raw_text = selected_text[
            selected_sentences[start_index].char_start : selected_sentences[
                end_index - 1
            ].char_end
        ]
        packed.append(
            PackedTextChunk(
                text=raw_text,
                char_start=selected_sentences[start_index].char_start,
                char_end=selected_sentences[end_index - 1].char_end,
                sentence_start=selected_sentences[start_index].sentence_index,
                sentence_end=selected_sentences[end_index - 1].sentence_index,
                overlap_target_tokens=selected_config.overlap_tokens,
                overlap_actual_tokens=pending_overlap_tokens,
                overlap_sentence_count=pending_overlap_sentences,
                oversized_sentence_split=False,
            )
        )
        if end_index >= len(selected_sentences):
            break
        if (
            len(
                codec.encode(
                    selected_budget_text[
                        selected_sentences[end_index].char_start : selected_sentences[
                            end_index
                        ].char_end
                    ]
                )
            )
            > selected_config.max_tokens
        ):
            start_index = end_index
            pending_overlap_tokens = 0
            pending_overlap_sentences = 0
            continue
        (
            next_start,
            pending_overlap_tokens,
            pending_overlap_sentences,
        ) = _whole_sentence_overlap_start(
            selected_budget_text,
            selected_sentences,
            start_index,
            end_index,
            end_index,
            codec,
            selected_config,
        )
        start_index = next_start
    return _repair_short_final_chunk(
        packed,
        selected_text,
        selected_budget_text,
        selected_sentences,
        codec,
        selected_config,
    )


def _source_blocks_for_char_range(
    stream: AdvancedTextStream,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """문자 구간과 실제로 겹치는 원본 블록을 순서대로 찾는다."""
    selected = [
        block
        for block_start, block_end, block in stream.block_char_spans
        if block_start < end and block_end > start
    ]
    if not selected:
        # 합성한 블록 사이 개행만 가리키는 조각은 만들어지지 않아야 한다.
        raise ValueError("텍스트 청크와 겹치는 원본 블록이 없습니다")
    return selected


def _range(values: Sequence[Mapping[str, Any]], field: str) -> tuple[Any, Any]:
    """null을 제외한 첫 위치와 마지막 위치를 반환한다."""
    selected = [value.get(field) for value in values if value.get(field) is not None]
    return (selected[0], selected[-1]) if selected else (None, None)


def inherited_advanced_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """사업·출처 metadata를 중첩 없이 청크 최상위에 복사한다."""
    business = document.get("business_metadata") or {}
    if not isinstance(business, Mapping):
        raise ValueError("business_metadata는 JSON 객체여야 합니다")
    record = {
        "source_id": document["source_id"],
        "document_id": document["document_id"],
        "source_sha256": document["source_sha256"],
        "source_filename": document["source_filename"],
        "source_relative_path": document.get("source_relative_path"),
        "filename_aliases": list(document.get("filename_aliases") or []),
        "file_type": document["file_type"],
        "metadata_validation_status": document.get("metadata_validation_status"),
    }
    record.update(
        {
            field: business.get(field, document.get(field))
            for field in BUSINESS_METADATA_FIELDS
        }
    )
    return record


def _tokenize_bm25(
    tokenizer: Bm25Tokenizer | Callable[[str], Sequence[Any]], text: str
) -> list[str]:
    """Kiwi 결과에서 팀 합의대로 조사(J*)와 어미(E*)만 제외한다."""
    if hasattr(tokenizer, "tokenize"):
        values = tokenizer.tokenize(text)  # type: ignore[union-attr]
    elif callable(tokenizer):
        values = tokenizer(text)
    else:
        raise TypeError("Kiwi tokenizer는 tokenize 메서드 또는 callable이어야 합니다")
    tokens: list[str] = []
    for value in values:
        if isinstance(value, str):
            form = value
            keep = True
        else:
            form = str(getattr(value, "form", ""))
            tag = str(getattr(value, "tag", ""))
            keep = not tag.startswith(BM25_EXCLUDED_POS_PREFIXES)
        normalized = form.strip().casefold()
        if keep and normalized:
            tokens.append(normalized)
    return tokens


def _quality_flags(
    source_blocks: Sequence[Mapping[str, Any]],
    extra: Iterable[str] = (),
) -> list[str]:
    """원본과 새 품질 flag를 입력 순서대로 중복 제거한다."""
    return unique_in_order(
        flag
        for flag in [
            *(
                flag
                for block in source_blocks
                for flag in (block.get("quality_flags") or [])
            ),
            *extra,
        ]
        if flag
    )


def build_advanced_chunk_record(
    *,
    document: Mapping[str, Any],
    source_blocks: Sequence[dict[str, Any]],
    embedding_text: str,
    raw_text: str | None = None,
    content_type: str,
    stream_id: str,
    stream_order: int,
    stream_part_index: int,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
    kiwi_tokenizer: Bm25Tokenizer | Callable[[str], Sequence[Any]] | None,
    overlap_actual_tokens: int = 0,
    overlap_sentence_count: int = 0,
    boundary_type: str | None = None,
    boundary_id: str | None = None,
    sentence_fields: Mapping[str, Any] | None = None,
    table_fields: Mapping[str, Any] | None = None,
    extra_quality_flags: Iterable[str] = (),
) -> dict[str, Any]:
    """일반 텍스트와 표가 공유하는 prefix 없는 JSON 청크를 만든다."""
    if not source_blocks:
        raise ValueError("청크의 원본 블록이 비어 있습니다")
    if not embedding_text.strip():
        raise ValueError("빈 임베딩 본문 청크를 만들 수 없습니다")
    if FORBIDDEN_IMAGE_PAYLOAD.search(embedding_text):
        raise ValueError("청크에 이미지 payload 또는 Base64가 포함됐습니다")
    selected_raw_text = embedding_text if raw_text is None else raw_text
    token_count = len(codec.encode(embedding_text))
    if token_count > config.max_tokens:
        raise ValueError(
            f"Advanced 청크가 토큰 상한을 넘었습니다: {token_count} > {config.max_tokens}"
        )
    is_text = content_type == "text"
    if is_text and kiwi_tokenizer is None:
        raise ValueError("일반 텍스트 BM25용 Kiwi tokenizer가 없습니다")
    if is_text and embedding_text != normalize_text_for_embedding(selected_raw_text):
        raise ValueError(
            "일반 텍스트 embedding_text에 줄바꿈이 남았거나 raw_text 정규화와 다릅니다"
        )
    bm25_tokens = (
        _tokenize_bm25(kiwi_tokenizer, embedding_text)
        if is_text and kiwi_tokenizer is not None
        else []
    )
    section_start, section_end = _range(source_blocks, "section_idx")
    para_start, para_end = _range(source_blocks, "para_idx")
    page_start, page_end = _range(source_blocks, "page")
    record = {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": INPUT_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "chunk_id": None,
        "chunk_order": None,
        "strategy_id": config.strategy_id,
        "chunk_size_tokens": config.max_tokens,
        "chunk_overlap_target_tokens": config.overlap_tokens,
        "min_tail_tokens": config.min_tail_tokens,
        "overlap_actual_tokens": overlap_actual_tokens,
        "overlap_sentence_count": overlap_sentence_count,
        "tokenizer_model": codec.model_name,
        "tokenizer_encoding": codec.encoding_name,
        "tokenizer_version": codec.version,
        "token_count": token_count,
        "token_count_basis": "embedding_text",
        "overlap_token_basis": (
            "normalized_embedding_text" if is_text else "embedding_text"
        ),
        **inherited_advanced_metadata(document),
        "source_block_ids": [block["block_id"] for block in source_blocks],
        "source_block_order_start": int(source_blocks[0]["block_order"]),
        "source_block_order_end": int(source_blocks[-1]["block_order"]),
        "source_index_policies": unique_in_order(
            block["index_policy"] for block in source_blocks
        ),
        "source_block_types": unique_in_order(
            block["block_type"] for block in source_blocks
        ),
        "content_type": content_type,
        # 일반 텍스트 raw는 KSS span 원문을 보존하고, embedding에는 팀
        # 합의대로 줄바꿈을 공백으로 바꾼 문자열만 넣는다. 표는 Markdown
        # 행 구조가 검색 의미이므로 두 필드가 같다.
        "raw_text": selected_raw_text,
        "embedding_text": embedding_text,
        "embedding_text_normalization": (
            TEXT_EMBEDDING_NORMALIZATION_ID if is_text else "preserve_markdown_newlines"
        ),
        "vectorize_field": "embedding_text",
        "embedding_prefix_included": False,
        "stream_id": stream_id,
        "stream_order": stream_order,
        "stream_part_index": stream_part_index,
        "kss_applied": is_text,
        "kss_boundary_type": boundary_type,
        "kss_boundary_id": boundary_id,
        "bm25_eligible": is_text,
        "bm25_tokenizer": "kiwipiepy" if is_text else None,
        "bm25_tokenizer_version": EXPECTED_KIWI_VERSION if is_text else None,
        "bm25_pos_policy": BM25_POS_POLICY_ID if is_text else None,
        "bm25_excluded_pos_prefixes": (
            list(BM25_EXCLUDED_POS_PREFIXES) if is_text else []
        ),
        "bm25_token_normalization": BM25_TOKEN_NORMALIZATION if is_text else None,
        "bm25_source_field": "embedding_text" if is_text else None,
        "bm25_tokens": bm25_tokens,
        "bm25_token_count": len(bm25_tokens),
        "section_path": source_blocks[0].get("section_path") or "본문",
        "section_idx_start": section_start,
        "section_idx_end": section_end,
        "para_idx_start": para_start,
        "para_idx_end": para_end,
        "page_start": page_start,
        "page_end": page_end,
        # 텍스트 청크는 source stream의 정확한 문자 범위를 기록해 독립
        # validator가 raw_text와 원본 블록 연결을 다시 검증할 수 있다.
        "stream_char_start": None,
        "stream_char_end": None,
        "table_id": None,
        "render_mode": None,
        "table_part_index": None,
        "table_part_count": None,
        "table_segment_index": None,
        "table_segment_count": None,
        "table_segment_part_index": None,
        "table_segment_part_count": None,
        "table_row_start": None,
        "table_row_end": None,
        "table_header_text": None,
        "table_header_repeated": False,
        "table_overlap_mode": None,
        "quality_flags": _quality_flags(source_blocks, extra_quality_flags),
    }
    if sentence_fields:
        record.update(sentence_fields)
    if table_fields:
        record.update(table_fields)
    return record


def _call_sentence_splitter(
    splitter: SentenceSplitter | Callable[[str], Sequence[str]],
    text: str,
) -> Sequence[str]:
    """주입한 KSS wrapper 또는 callable을 한 경계에 한 번 호출한다."""
    if callable(splitter):
        return splitter(text)
    raise TypeError("sentence_splitter는 callable이어야 합니다")


def chunk_advanced_text_stream(
    document: Mapping[str, Any],
    stream: AdvancedTextStream,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
    sentence_splitter: SentenceSplitter | Callable[[str], Sequence[str]],
    kiwi_tokenizer: Bm25Tokenizer | Callable[[str], Sequence[Any]],
) -> list[dict[str, Any]]:
    """한 위치 stream에 KSS→문장 packing→Kiwi를 순서대로 적용한다."""
    kss_input, sanitized_character_count = _sanitize_kss_input(stream.text)
    kss_sentences = _call_sentence_splitter(sentence_splitter, kss_input)
    alignment_fallback = False
    try:
        sentences = align_kss_sentences(
            stream.text,
            kss_sentences,
            boundary_id=stream.boundary_id,
            context=(
                f"source_id={document['source_id']}, boundary={stream.boundary_id}, "
                f"blocks={[block['block_id'] for block in stream.blocks]}"
            ),
        )
    except KssAlignmentError:
        # pecab이 표시 문자 외의 원문까지 바꾼 예외 문단은 잘못된 KSS 경계를
        # 억지로 적용하지 않는다. 한 문장 span으로 원문을 100% 보존하고,
        # 512를 넘을 때만 기존 UTF-8 안전 토큰 fallback이 나눈다.
        alignment_fallback = True
        sentences = [
            SentenceSpan(
                boundary_id=stream.boundary_id,
                sentence_index=1,
                char_start=0,
                char_end=len(stream.text),
                normalized_text=stream.text,
                raw_text=stream.text,
            )
        ]
    packed = pack_sentence_spans(stream.text, sentences, codec, config)
    alignment_flags: list[str] = []
    if sanitized_character_count:
        alignment_flags.append("kss_input_sanitized_private_format_or_decorative")
    if alignment_fallback:
        alignment_flags.append("kss_alignment_fallback_whole_boundary")
    alignment_status = (
        "whole_boundary_fallback"
        if alignment_fallback
        else "sanitized_aligned"
        if sanitized_character_count
        else "aligned"
    )
    chunks: list[dict[str, Any]] = []
    covered_char_end = 0
    for part_index, part in enumerate(packed, start=1):
        source_blocks = _source_blocks_for_char_range(
            stream,
            part.char_start,
            part.char_end,
        )
        extra_flags = [*alignment_flags]
        if part.oversized_sentence_split:
            extra_flags.append("oversized_kss_sentence_token_split")
        if part.short_tail_adjusted:
            extra_flags.append("short_final_text_chunk_adjusted")
        if part.short_tail_token_overlap_fallback:
            extra_flags.append("short_tail_token_overlap_fallback")
        if part.short_tail_adjustment_mode == "merged_with_previous":
            extra_flags.append("short_tail_merged_with_previous")
        new_char_start = max(covered_char_end, part.char_start)
        new_content_token_count = len(
            codec.encode(
                normalize_text_for_embedding(
                    stream.text[new_char_start : part.char_end]
                )
            )
        )
        covered_char_end = max(covered_char_end, part.char_end)
        embedding_text = normalize_text_for_embedding(part.text)
        chunks.append(
            build_advanced_chunk_record(
                document=document,
                source_blocks=source_blocks,
                raw_text=part.text,
                embedding_text=embedding_text,
                content_type="text",
                stream_id=stream.stream_id,
                stream_order=stream.stream_order,
                stream_part_index=part_index,
                codec=codec,
                config=config,
                kiwi_tokenizer=kiwi_tokenizer,
                overlap_actual_tokens=part.overlap_actual_tokens,
                overlap_sentence_count=part.overlap_sentence_count,
                boundary_type=stream.boundary_type,
                boundary_id=stream.boundary_id,
                sentence_fields={
                    "sentence_start": part.sentence_start,
                    "sentence_end": part.sentence_end,
                    "sentence_count": part.sentence_end - part.sentence_start + 1,
                    "oversized_sentence_split": part.oversized_sentence_split,
                    "sentence_fragment_index": part.sentence_fragment_index,
                    "sentence_fragment_count": part.sentence_fragment_count,
                    "sentence_token_start": part.sentence_token_start,
                    "sentence_token_end": part.sentence_token_end,
                    "short_tail_adjusted": part.short_tail_adjusted,
                    "short_tail_token_overlap_fallback": (
                        part.short_tail_token_overlap_fallback
                    ),
                    "short_tail_adjustment_mode": part.short_tail_adjustment_mode,
                    "short_tail_original_new_token_count": (
                        part.short_tail_original_new_token_count
                    ),
                    "short_tail_context_added_tokens": (
                        part.short_tail_context_added_tokens
                    ),
                    "new_content_token_count": new_content_token_count,
                    "stream_char_start": part.char_start,
                    "stream_char_end": part.char_end,
                    "kss_input_sanitized": bool(sanitized_character_count),
                    "kss_stream_sanitized_character_count": (sanitized_character_count),
                    "kss_alignment_fallback": alignment_fallback,
                    "kss_alignment_status": alignment_status,
                },
                extra_quality_flags=extra_flags,
            )
        )
    return chunks


def _split_for_rendered_budget(
    source_text: str,
    render: Callable[[str], str],
    codec: TokenCodec,
    max_tokens: int,
) -> list[str]:
    """render 결과가 상한을 넘지 않도록 원문을 UTF-8 안전 구간으로 나눈다."""
    token_map = TokenTextMap(source_text, codec)
    fragments: list[str] = []
    start = 0
    while start < len(token_map):
        best_end = start
        for end in token_map.safe_token_indices:
            if end <= start:
                continue
            fragment = token_map.slice(start, end)
            if len(codec.encode(render(fragment))) > max_tokens:
                break
            best_end = end
        if best_end <= start:
            raise ValueError(
                "Markdown wrapper 안에 표 내용을 한 글자도 넣지 못했습니다"
            )
        fragments.append(token_map.slice(start, best_end))
        start = best_end
    return fragments


def _one_cell_table(fragment: str, label: str = "표 내용") -> str:
    """긴 표 header/row 조각을 유효한 한 셀 Markdown으로 감싼다."""
    escaped = escape_markdown_fallback_cell(fragment)
    return f"| {label} |\n| --- |\n| {escaped} |"


def _fallback_table_segment_parts(
    source_text: str,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> list[str]:
    """헤더 예산이 고갈된 표 전체를 손실 없는 한 셀 Markdown들로 나눈다."""
    return [
        _one_cell_table(fragment)
        for fragment in _split_for_rendered_budget(
            source_text,
            _one_cell_table,
            codec,
            config.max_tokens,
        )
    ]


def _chunk_table_segment_texts(
    segment: Any,
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> list[dict[str, Any]]:
    """한 Markdown 표 segment를 온전한 행과 반복 헤더 중심으로 나눈다."""
    header = segment_header_text(segment)
    source_text = "\n".join(
        [*segment.context_lines, *segment.header_lines, *segment.data_rows]
    )
    if len(codec.encode(header)) > config.max_tokens:
        return [
            {
                "text": text,
                "row_start": None,
                "row_end": None,
                "header_text": "| 표 내용 |\n| --- |",
                "header_repeated": part_index > 1,
                "quality_flags": [
                    "oversized_table_header_split",
                    "table_structure_flattened_fallback",
                ],
            }
            for part_index, text in enumerate(
                _fallback_table_segment_parts(source_text, codec, config), start=1
            )
        ]

    # 세 번째 값은 원본 한 행을 토큰 fallback으로 쪼갠 unit인지 표시한다.
    # 문자열 모양으로 추정하면 escape된 ``|`` 때문에 오판할 수 있다.
    rows: list[tuple[int, str, bool]] = []
    for row_number, row in enumerate(segment.data_rows, start=1):
        if len(codec.encode(f"{header}\n{row}")) <= config.max_tokens:
            rows.append((row_number, row, False))
            continue

        # 헤더와 원본 행을 그대로 넣을 수 없을 때만 한 셀 Markdown fallback을
        # 쓴다. 먼저 원본 헤더를 반복해 문맥을 보존하고, 예산이 없으면 표
        # 전체 fallback으로 전환한다.
        def render(fragment: str, header_text: str = header) -> str:
            """긴 한 행 조각에 기존 표 헤더를 반복해 Markdown을 만든다."""
            return f"{header_text}\n| {escape_markdown_fallback_cell(fragment)} |"

        try:
            fragments = _split_for_rendered_budget(
                row,
                render,
                codec,
                config.max_tokens,
            )
        except ValueError:
            return [
                {
                    "text": text,
                    "row_start": None,
                    "row_end": None,
                    "header_text": "| 표 내용 |\n| --- |",
                    "header_repeated": part_index > 1,
                    "quality_flags": [
                        "table_header_budget_exhausted_fallback",
                        "table_structure_flattened_fallback",
                    ],
                }
                for part_index, text in enumerate(
                    _fallback_table_segment_parts(source_text, codec, config), start=1
                )
            ]
        # fragment_index는 동일 원본 행의 순서를 유지하기 위해 별도 튜플에
        # 넣지 않고 row 번호와 함께 연속 units로 기록한다.
        for fragment in fragments:
            # header가 제목·설명 등 3줄 이상이어도 전체 render 문자열을
            # split해서 꺼내지 않는다. 분할된 한 셀 행만 저장한 뒤 packing
            # 단계에서 header를 정확히 한 번 다시 붙인다.
            fallback_row = f"| {escape_markdown_fallback_cell(fragment)} |"
            rows.append((row_number, fallback_row, True))

    if not rows:
        return [
            {
                "text": header,
                "row_start": None,
                "row_end": None,
                "header_text": header,
                "header_repeated": False,
                "quality_flags": [],
            }
        ]

    parts: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(rows):
        selected: list[tuple[int, str, bool]] = []
        while cursor < len(rows):
            candidate = "\n".join(
                [header, *(row for _, row, _ in [*selected, rows[cursor]])]
            )
            if len(codec.encode(candidate)) > config.max_tokens:
                break
            selected.append(rows[cursor])
            cursor += 1
        if not selected:
            raise ValueError("표 행을 512토큰 예산 안에 넣지 못했습니다")
        row_numbers = [row_number for row_number, _, _ in selected]
        oversized = any(is_oversized for _, _, is_oversized in selected)
        parts.append(
            {
                "text": "\n".join([header, *(row for _, row, _ in selected)]),
                "row_start": min(row_numbers),
                "row_end": max(row_numbers),
                "header_text": header,
                "header_repeated": bool(parts),
                "quality_flags": ["oversized_table_row_split"] if oversized else [],
            }
        )
    return parts


def chunk_advanced_table_block(
    document: Mapping[str, Any],
    block: dict[str, Any],
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> list[dict[str, Any]]:
    """표 Markdown만 행 단위로 나누고 KSS·Kiwi는 호출하지 않는다."""
    if not _is_advanced_table(block):
        raise ValueError(f"Advanced Dense 표 계약이 아닙니다: {block.get('block_id')}")
    markdown = str(block["table_markdown"])
    if FORBIDDEN_IMAGE_PAYLOAD.search(markdown):
        raise ValueError(
            f"표 Markdown에 이미지 payload가 있습니다: {block['block_id']}"
        )
    if HTML_TABLE_TAG.search(markdown):
        raise ValueError(
            f"표 임베딩 본문에 HTML 표 태그가 있습니다: {block['block_id']}"
        )
    segments = parse_markdown_table_segments(markdown)
    if not segments:
        raise ValueError(f"표 Markdown을 해석할 수 없습니다: {block['block_id']}")

    segment_parts: list[tuple[Any, dict[str, Any], int, int]] = []
    for segment in segments:
        parts = _chunk_table_segment_texts(segment, codec, config)
        for part_index, part in enumerate(parts, start=1):
            segment_parts.append((segment, part, part_index, len(parts)))

    chunks: list[dict[str, Any]] = []
    for table_part_index, (
        segment,
        part,
        segment_part_index,
        segment_part_count,
    ) in enumerate(segment_parts, start=1):
        chunks.append(
            build_advanced_chunk_record(
                document=document,
                source_blocks=[block],
                embedding_text=str(part["text"]),
                content_type="table",
                stream_id=str(block.get("stream_id") or block["block_id"]),
                stream_order=int(block.get("stream_order") or block["block_order"]),
                stream_part_index=table_part_index,
                codec=codec,
                config=config,
                kiwi_tokenizer=None,
                table_fields={
                    "table_id": block["table_id"],
                    "render_mode": "gfm",
                    "table_part_index": table_part_index,
                    "table_part_count": len(segment_parts),
                    "table_segment_index": segment.segment_index,
                    "table_segment_count": len(segments),
                    "table_segment_part_index": segment_part_index,
                    "table_segment_part_count": segment_part_count,
                    "table_row_start": part["row_start"],
                    "table_row_end": part["row_end"],
                    "table_header_text": part["header_text"],
                    "table_header_repeated": part["header_repeated"],
                    "table_overlap_mode": (
                        "header_repeat_only" if part["header_repeated"] else "none"
                    ),
                },
                extra_quality_flags=part["quality_flags"],
            )
        )
    return chunks


def build_advanced_chunk_corpus(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    codec: TokenCodec | None = None,
    config: AdvancedChunkConfig | None = None,
    sentence_splitter: SentenceSplitter | Callable[[str], Sequence[str]] | None = None,
    kiwi_tokenizer: Bm25Tokenizer | Callable[[str], Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Advanced 문서·블록 전체를 결정적인 순서의 청크 corpus로 만든다."""
    selected_config = config or AdvancedChunkConfig()
    validate_advanced_config(selected_config)
    selected_codec = codec or TiktokenCodec(
        selected_config.model_name,
        selected_config.encoding_name,
    )
    selected_splitter = sentence_splitter or KssSentenceSplitter()
    selected_kiwi = kiwi_tokenizer or KiwiBm25Tokenizer()

    documents_by_id: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        source_id = str(document.get("source_id") or "")
        if not source_id or source_id in documents_by_id:
            raise ValueError(
                f"Advanced 문서 source_id가 없거나 중복됐습니다: {source_id}"
            )
        if document.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"Advanced 입력 문서 스키마가 다릅니다: {document.get('schema_version')}"
            )
        documents_by_id[source_id] = document

    blocks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_block_ids: set[str] = set()
    for block in blocks:
        block_id = str(block.get("block_id") or "")
        source_id = str(block.get("source_id") or "")
        if not block_id or block_id in seen_block_ids:
            raise ValueError(f"Advanced block_id가 없거나 중복됐습니다: {block_id}")
        if source_id not in documents_by_id:
            raise ValueError(f"블록의 문서가 없습니다: {block_id}")
        if block.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError(f"Advanced 입력 블록 스키마가 다릅니다: {block_id}")
        if block.get("embedding_prefix_included") is not False:
            raise ValueError(f"입력 블록에 metadata prefix 표시가 있습니다: {block_id}")
        seen_block_ids.add(block_id)
        blocks_by_document[source_id].append(block)

    corpus: list[dict[str, Any]] = []
    # 입력 JSONL 행 순서가 달라도 canonical corpus 순서는 source_id로 같다.
    for document in sorted(documents, key=lambda row: str(row["source_id"])):
        source_id = str(document["source_id"])
        document_chunks: list[dict[str, Any]] = []
        streams = build_advanced_streams(document, blocks_by_document[source_id])
        for stream in streams:
            if isinstance(stream, AdvancedTextStream):
                document_chunks.extend(
                    chunk_advanced_text_stream(
                        document,
                        stream,
                        selected_codec,
                        selected_config,
                        selected_splitter,
                        selected_kiwi,
                    )
                )
            else:
                document_chunks.extend(
                    chunk_advanced_table_block(
                        document,
                        stream,
                        selected_codec,
                        selected_config,
                    )
                )
        for order, chunk in enumerate(document_chunks, start=1):
            chunk["chunk_order"] = order
            chunk["chunk_id"] = (
                f"{source_id}:ADV2:A{selected_config.max_tokens}"
                f"O{selected_config.overlap_tokens}:C{order:06d}"
            )
        corpus.extend(document_chunks)
    return corpus


def chunk_advanced_corpus(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    codec: TokenCodec | None = None,
    config: AdvancedChunkConfig | None = None,
    sentence_splitter: SentenceSplitter | Callable[[str], Sequence[str]] | None = None,
    kiwi_tokenizer: Bm25Tokenizer | Callable[[str], Sequence[Any]] | None = None,
) -> AdvancedChunkingResult:
    """여러 Advanced 입력을 청킹하고 검증·요약까지 한 번에 수행한다."""
    selected_config = config or AdvancedChunkConfig()
    selected_codec = codec or TiktokenCodec(
        selected_config.model_name,
        selected_config.encoding_name,
    )
    chunks = build_advanced_chunk_corpus(
        documents,
        blocks,
        selected_codec,
        selected_config,
        sentence_splitter,
        kiwi_tokenizer,
    )
    validation = validate_advanced_chunks(
        documents,
        blocks,
        chunks,
        selected_codec,
        selected_config,
    )
    if not validation["overall_pass"]:
        failed = [name for name, passed in validation["gates"].items() if not passed]
        raise ValueError(f"Advanced 청킹 품질 검증 실패: {', '.join(failed)}")
    return AdvancedChunkingResult(
        chunks=tuple(chunks),
        summary=build_advanced_summary(
            documents,
            blocks,
            chunks,
            validation,
            selected_codec,
            selected_config,
        ),
    )


def validate_advanced_chunks(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> dict[str, Any]:
    """Advanced 위치·표·KSS·Kiwi·metadata 불변식을 독립적으로 검사한다."""
    documents_by_id = {str(row["source_id"]): row for row in documents}
    blocks_by_id = {str(row["block_id"]): row for row in blocks}
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)
    text_streams_by_id: dict[tuple[str, str], AdvancedTextStream] = {}
    table_streams_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    page_markers_by_source: dict[str, set[str]] = defaultdict(set)
    page_marker_ids: set[str] = set()
    for source_id, document in documents_by_id.items():
        source_blocks = blocks_by_source.get(source_id, [])
        detected = _find_pdf_page_marker_blocks(document, source_blocks)
        page_marker_ids.update(detected)
        page_markers_by_source[source_id].update(detected)
        for stream in build_advanced_streams(document, source_blocks):
            if isinstance(stream, AdvancedTextStream):
                text_streams_by_id[(source_id, stream.stream_id)] = stream
            else:
                table_streams_by_id[(source_id, str(stream["stream_id"]))] = stream

    def source_link_contract(chunk: Mapping[str, Any]) -> bool:
        """청크의 출처·순서·유형 연결이 실제 입력 블록과 같은지 본다."""
        source_id = str(chunk.get("source_id") or "")
        block_ids = chunk.get("source_block_ids")
        if (
            source_id not in documents_by_id
            or not isinstance(block_ids, list)
            or not block_ids
            or any(str(block_id) not in blocks_by_id for block_id in block_ids)
        ):
            return False
        linked = [blocks_by_id[str(block_id)] for block_id in block_ids]
        orders = [int(block.get("block_order") or 0) for block in linked]
        if (
            any(str(block.get("source_id") or "") != source_id for block in linked)
            or orders != sorted(set(orders))
            or chunk.get("source_block_order_start") != orders[0]
            or chunk.get("source_block_order_end") != orders[-1]
        ):
            return False
        if chunk.get("content_type") == "text":
            return all(
                _is_advanced_text(block)
                and str(block["block_id"])
                not in page_markers_by_source.get(source_id, set())
                for block in linked
            )
        if chunk.get("content_type") == "table":
            return (
                len(linked) == 1
                and _is_advanced_table(linked[0])
                and chunk.get("table_id") == linked[0].get("table_id")
            )
        return False

    def raw_text_matches_source_stream(chunk: Mapping[str, Any]) -> bool:
        """텍스트 raw가 기록된 stream 문자 범위와 블록 집합에 일치하는지 본다."""
        if chunk.get("content_type") != "text":
            return True
        source_id = str(chunk.get("source_id") or "")
        stream_id = str(chunk.get("stream_id") or "")
        stream = text_streams_by_id.get((source_id, stream_id))
        start = chunk.get("stream_char_start")
        end = chunk.get("stream_char_end")
        if (
            stream is None
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(stream.text)
            or chunk.get("stream_order") != stream.stream_order
            or chunk.get("kss_boundary_type") != stream.boundary_type
            or chunk.get("kss_boundary_id") != stream.boundary_id
            or chunk.get("raw_text") != stream.text[start:end]
        ):
            return False
        linked_ids = [
            str(block["block_id"])
            for block in _source_blocks_for_char_range(stream, start, end)
        ]
        return linked_ids == [
            str(block_id) for block_id in chunk.get("source_block_ids") or []
        ]

    eligible_ids = {
        str(block["block_id"])
        for block in blocks
        if _is_advanced_text(block) or _is_advanced_table(block)
    }
    required_eligible_ids = eligible_ids - page_marker_ids
    nonindexable_ids = set(blocks_by_id) - eligible_ids
    covered_ids = {
        str(block_id)
        for chunk in chunks
        for block_id in chunk.get("source_block_ids") or []
    }
    chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    covered_page_marker_ids = covered_ids & page_marker_ids
    metadata_only_page_marker_ids = page_marker_ids - covered_ids

    orders_valid = True
    for source_id, source_chunks in _group_chunks_by_source(chunks).items():
        expected_orders = list(range(1, len(source_chunks) + 1))
        actual_orders = [int(chunk["chunk_order"]) for chunk in source_chunks]
        expected_ids = [
            f"{source_id}:ADV2:A{config.max_tokens}"
            f"O{config.overlap_tokens}:C{order:06d}"
            for order in expected_orders
        ]
        orders_valid &= actual_orders == expected_orders
        orders_valid &= [chunk["chunk_id"] for chunk in source_chunks] == expected_ids

    source_links_valid = all(source_link_contract(chunk) for chunk in chunks)
    raw_text_source_valid = all(
        raw_text_matches_source_stream(chunk) for chunk in chunks
    )
    token_counts_valid = all(
        1 <= int(chunk["token_count"]) <= config.max_tokens
        and int(chunk["token_count"]) == len(codec.encode(chunk["embedding_text"]))
        for chunk in chunks
    )
    embedding_text_contract_valid = all(
        (
            chunk.get("embedding_text")
            == normalize_text_for_embedding(str(chunk.get("raw_text") or ""))
            and not any(
                separator in str(chunk.get("embedding_text") or "")
                for separator in TEXT_LINE_SEPARATOR_CHARS
            )
            and chunk.get("embedding_text_normalization")
            == TEXT_EMBEDDING_NORMALIZATION_ID
        )
        if chunk.get("content_type") == "text"
        else (
            chunk.get("embedding_text") == chunk.get("raw_text")
            and chunk.get("embedding_text_normalization")
            == "preserve_markdown_newlines"
        )
        for chunk in chunks
    )
    schema_contract_valid = all(
        chunk.get("schema_version") == SCHEMA_VERSION
        and chunk.get("source_schema_version") == INPUT_SCHEMA_VERSION
        and chunk.get("strategy_id") == config.strategy_id
        and chunk.get("corpus_id") == CORPUS_ID
        and chunk.get("token_count_basis") == "embedding_text"
        and chunk.get("overlap_token_basis")
        == (
            "normalized_embedding_text"
            if chunk.get("content_type") == "text"
            else "embedding_text"
        )
        for chunk in chunks
    )
    prefix_absent = all(
        chunk.get("embedding_prefix_included") is False
        and not METADATA_PREFIX.search(str(chunk.get("embedding_text") or ""))
        for chunk in chunks
    )

    locations_valid = True
    table_contract_valid = True
    bm25_contract_valid = True
    kss_metadata_valid = True
    tail_metadata_valid = True
    metadata_valid = True
    for chunk in chunks:
        document = documents_by_id[str(chunk["source_id"])]
        source_blocks = [
            blocks_by_id[str(value)] for value in chunk["source_block_ids"]
        ]
        file_type = str(document.get("file_type") or "").casefold()
        if file_type == "pdf":
            pages = {block.get("page") for block in source_blocks}
            locations_valid &= len(pages) == 1 and next(iter(pages), None) is not None
            locations_valid &= chunk.get("page_start") == chunk.get("page_end")
        else:
            locations_valid &= chunk.get("page_start") is None
            locations_valid &= chunk.get("page_end") is None
            if chunk["content_type"] == "text":
                boundary_ids = {block.get("kss_boundary_id") for block in source_blocks}
                locations_valid &= len(boundary_ids) == 1
                locations_valid &= chunk.get("kss_boundary_id") in boundary_ids

        if chunk["content_type"] == "table":
            table_contract_valid &= chunk.get("kss_applied") is False
            table_contract_valid &= chunk.get("bm25_eligible") is False
            table_contract_valid &= chunk.get("bm25_tokens") == []
            table_contract_valid &= chunk.get("bm25_pos_policy") is None
            table_contract_valid &= chunk.get("bm25_excluded_pos_prefixes") == []
            table_contract_valid &= chunk.get("bm25_token_normalization") is None
            table_contract_valid &= chunk.get("bm25_source_field") is None
            table_contract_valid &= chunk.get("table_id") == source_blocks[0].get(
                "table_id"
            )
            table_contract_valid &= not HTML_TABLE_TAG.search(chunk["embedding_text"])
            table_contract_valid &= bool(
                parse_markdown_table_segments(chunk["embedding_text"])
            )
        else:
            bm25_contract_valid &= chunk.get("kss_applied") is True
            bm25_contract_valid &= chunk.get("bm25_eligible") is True
            bm25_contract_valid &= chunk.get("bm25_pos_policy") == BM25_POS_POLICY_ID
            bm25_contract_valid &= chunk.get("bm25_excluded_pos_prefixes") == list(
                BM25_EXCLUDED_POS_PREFIXES
            )
            bm25_contract_valid &= (
                chunk.get("bm25_token_normalization") == BM25_TOKEN_NORMALIZATION
            )
            bm25_contract_valid &= chunk.get("bm25_source_field") == "embedding_text"
            bm25_contract_valid &= isinstance(chunk.get("bm25_tokens"), list)
            bm25_contract_valid &= chunk.get("bm25_token_count") == len(
                chunk.get("bm25_tokens") or []
            )
            sanitized = chunk.get("kss_input_sanitized")
            sanitized_count = chunk.get("kss_stream_sanitized_character_count")
            alignment_fallback = chunk.get("kss_alignment_fallback")
            alignment_status = chunk.get("kss_alignment_status")
            quality_flags = set(chunk.get("quality_flags") or [])
            kss_metadata_valid &= isinstance(sanitized, bool)
            kss_metadata_valid &= (
                isinstance(sanitized_count, int) and sanitized_count >= 0
            )
            kss_metadata_valid &= sanitized is bool(sanitized_count)
            kss_metadata_valid &= isinstance(alignment_fallback, bool)
            expected_status = (
                "whole_boundary_fallback"
                if alignment_fallback
                else "sanitized_aligned"
                if sanitized
                else "aligned"
            )
            kss_metadata_valid &= alignment_status == expected_status
            kss_metadata_valid &= (
                "kss_input_sanitized_private_format_or_decorative" in quality_flags
            ) is sanitized
            kss_metadata_valid &= (
                "kss_alignment_fallback_whole_boundary" in quality_flags
            ) is alignment_fallback
            adjustment_mode = str(chunk.get("short_tail_adjustment_mode") or "none")
            adjusted = chunk.get("short_tail_adjusted")
            token_fallback = chunk.get("short_tail_token_overlap_fallback")
            original_new_tokens = chunk.get("short_tail_original_new_token_count")
            context_added_tokens = chunk.get("short_tail_context_added_tokens")
            allowed_modes = {
                "none",
                "merged_with_previous",
                "whole_sentence_rebalance",
                "existing_overlap_context",
                "safe_token_overlap_fallback",
            }
            tail_metadata_valid &= adjustment_mode in allowed_modes
            tail_metadata_valid &= adjusted is (adjustment_mode != "none")
            tail_metadata_valid &= token_fallback is (
                adjustment_mode == "safe_token_overlap_fallback"
            )
            tail_metadata_valid &= (
                isinstance(chunk.get("new_content_token_count"), int)
                and int(chunk.get("new_content_token_count") or 0) >= 0
            )
            if adjustment_mode == "none":
                tail_metadata_valid &= original_new_tokens is None
                tail_metadata_valid &= context_added_tokens == 0
            else:
                tail_metadata_valid &= (
                    isinstance(original_new_tokens, int)
                    and 0 <= original_new_tokens < config.min_tail_tokens
                )
                tail_metadata_valid &= (
                    isinstance(context_added_tokens, int) and context_added_tokens >= 0
                )
            if adjustment_mode in {
                "existing_overlap_context",
                "safe_token_overlap_fallback",
            }:
                tail_metadata_valid &= int(context_added_tokens or 0) > 0
            elif adjustment_mode != "none":
                tail_metadata_valid &= context_added_tokens == 0

        business = document.get("business_metadata") or {}
        metadata_valid &= all(
            chunk.get(field) == business.get(field, document.get(field))
            for field in BUSINESS_METADATA_FIELDS
        )
        metadata_valid &= chunk.get("source_filename") == document.get(
            "source_filename"
        )

    text_chunks_by_stream: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for chunk in chunks:
        if chunk.get("content_type") == "text":
            text_chunks_by_stream[
                (str(chunk.get("source_id")), str(chunk.get("stream_id")))
            ].append(chunk)
    short_tail_contract_valid = True
    all_nonfirst_chunks_add_content = True
    text_stream_coverage_valid = set(text_chunks_by_stream) == set(text_streams_by_id)
    invalid_text_stream_ids: list[str] = []
    for stream_key, stream_chunks in text_chunks_by_stream.items():
        stream = text_streams_by_id.get(stream_key)
        ordered_stream_chunks = sorted(
            stream_chunks,
            key=lambda row: int(row.get("stream_part_index") or 0),
        )
        stream_valid = stream is not None
        stream_valid &= [
            int(chunk.get("stream_part_index") or 0) for chunk in ordered_stream_chunks
        ] == list(range(1, len(ordered_stream_chunks) + 1))
        covered_char_end = 0
        if stream is not None:
            for part_index, chunk in enumerate(ordered_stream_chunks, start=1):
                start = chunk.get("stream_char_start")
                end = chunk.get("stream_char_end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(stream.text)
                    or (part_index == 1 and start != 0)
                    or (part_index > 1 and start > covered_char_end)
                ):
                    stream_valid = False
                    continue
                new_char_start = max(covered_char_end, start)
                expected_new_tokens = len(
                    codec.encode(
                        normalize_text_for_embedding(stream.text[new_char_start:end])
                    )
                )
                expected_overlap_tokens = len(
                    codec.encode(
                        normalize_text_for_embedding(
                            stream.text[start : min(covered_char_end, end)]
                        )
                    )
                )
                stream_valid &= (
                    chunk.get("new_content_token_count") == expected_new_tokens
                )
                stream_valid &= (
                    chunk.get("overlap_actual_tokens") == expected_overlap_tokens
                )
                covered_char_end = max(covered_char_end, end)
            stream_valid &= covered_char_end == len(stream.text)
        text_stream_coverage_valid &= stream_valid
        if not stream_valid:
            invalid_text_stream_ids.append(f"{stream_key[0]}::{stream_key[1]}")
        all_nonfirst_chunks_add_content &= all(
            int(chunk.get("new_content_token_count") or 0) > 0
            for chunk in ordered_stream_chunks[1:]
        )
        if len(stream_chunks) < 2:
            continue
        final_chunk = max(
            stream_chunks,
            key=lambda row: int(row.get("stream_part_index") or 0),
        )
        new_tokens = int(final_chunk.get("new_content_token_count") or 0)
        adjustment_mode = str(final_chunk.get("short_tail_adjustment_mode") or "none")
        if new_tokens >= config.min_tail_tokens:
            continue
        short_tail_contract_valid &= (
            adjustment_mode
            in {"existing_overlap_context", "safe_token_overlap_fallback"}
            and int(final_chunk.get("token_count") or 0) >= config.min_tail_tokens
            and int(final_chunk.get("short_tail_context_added_tokens") or 0) > 0
        )

    # 표는 KSS·Kiwi를 쓰지 않아 원본 Markdown에서 결정적으로 재생성할 수
    # 있다. 저장된 표 청크를 다시 만든 기대값과 비교해 본문 변조·행 유실을
    # 단순 Markdown 문법 검사보다 강하게 탐지한다.
    table_chunks_by_stream: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for chunk in chunks:
        if chunk.get("content_type") == "table":
            table_chunks_by_stream[
                (str(chunk.get("source_id")), str(chunk.get("stream_id")))
            ].append(chunk)
    table_source_contract_valid = set(table_chunks_by_stream) == set(
        table_streams_by_id
    )
    invalid_table_stream_ids: list[str] = []
    for stream_key, actual_chunks in table_chunks_by_stream.items():
        table_stream = table_streams_by_id.get(stream_key)
        document = documents_by_id.get(stream_key[0])
        stream_valid = table_stream is not None and document is not None
        if stream_valid and table_stream is not None and document is not None:
            expected_chunks = chunk_advanced_table_block(
                document,
                table_stream,
                codec,
                config,
            )
            ordered_actual = sorted(
                actual_chunks,
                key=lambda row: int(row.get("stream_part_index") or 0),
            )
            stream_valid &= len(ordered_actual) == len(expected_chunks)
            for actual, expected in zip(ordered_actual, expected_chunks):
                stream_valid &= all(
                    actual.get(field) == value
                    for field, value in expected.items()
                    if field not in {"chunk_id", "chunk_order"}
                )
        table_source_contract_valid &= stream_valid
        if not stream_valid:
            invalid_table_stream_ids.append(f"{stream_key[0]}::{stream_key[1]}")

    gates = {
        "document_ids_are_unique": len(documents_by_id) == len(documents),
        "block_ids_are_unique": len(blocks_by_id) == len(blocks),
        "chunk_ids_are_unique": len(chunk_ids) == len(set(chunk_ids)),
        "chunk_orders_and_ids_are_contiguous": bool(orders_valid),
        "chunk_schema_strategy_and_corpus_are_v2": bool(schema_contract_valid),
        "all_required_dense_text_and_tables_are_covered": (
            covered_ids >= required_eligible_ids
        ),
        "pdf_page_marker_blocks_are_metadata_only": not covered_page_marker_ids,
        "no_nonindexable_or_image_block_is_covered": not (
            covered_ids & nonindexable_ids
        ),
        "source_block_links_are_valid": source_links_valid,
        "text_raw_matches_source_stream_span": raw_text_source_valid,
        "text_streams_are_contiguously_covered": bool(text_stream_coverage_valid),
        "table_chunks_match_source_markdown": bool(table_source_contract_valid),
        "token_counts_and_512_limit_are_valid": token_counts_valid,
        "text_newlines_are_excluded_from_embedding_only": bool(
            embedding_text_contract_valid
        ),
        "pdf_page_and_hwp_paragraph_boundaries_are_preserved": bool(locations_valid),
        "tables_use_markdown_without_kss_or_bm25": bool(table_contract_valid),
        "text_only_has_kiwi_bm25_tokens": bool(bm25_contract_valid),
        "kss_sanitization_metadata_is_consistent": bool(kss_metadata_valid),
        "tail_adjustment_metadata_is_consistent": bool(tail_metadata_valid),
        "short_final_text_chunks_are_adjusted": bool(short_tail_contract_valid),
        "every_nonfirst_text_chunk_adds_new_content": bool(
            all_nonfirst_chunks_add_content
        ),
        "embedding_prefix_is_absent": prefix_absent,
        "business_and_file_metadata_are_preserved": bool(metadata_valid),
        "no_image_payload_or_base64": all(
            not FORBIDDEN_IMAGE_PAYLOAD.search(str(chunk["embedding_text"]))
            for chunk in chunks
        ),
        "all_embedding_text_is_nonempty": all(
            bool(str(chunk["embedding_text"]).strip()) for chunk in chunks
        ),
    }
    return {
        "overall_pass": all(gates.values()),
        "gates": gates,
        "diagnostics": {
            "missing_required_eligible_block_ids": sorted(
                required_eligible_ids - covered_ids
            ),
            "covered_page_marker_block_ids": sorted(covered_page_marker_ids),
            "metadata_only_page_marker_block_ids": sorted(
                metadata_only_page_marker_ids
            ),
            "covered_nonindexable_block_ids": sorted(nonindexable_ids & covered_ids),
            "invalid_text_stream_ids": sorted(invalid_text_stream_ids),
            "invalid_table_stream_ids": sorted(invalid_table_stream_ids),
        },
    }


def _group_chunks_by_source(
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """입력 순서를 유지한 source_id별 청크 목록을 만든다."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk["source_id"])].append(chunk)
    return dict(grouped)


def build_advanced_summary(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    validation: Mapping[str, Any],
    codec: TokenCodec,
    config: AdvancedChunkConfig,
) -> dict[str, Any]:
    """실행 시각과 무관한 Advanced 청킹 품질 요약을 만든다."""
    token_counts = [int(chunk["token_count"]) for chunk in chunks]
    overlaps = [
        int(chunk["overlap_actual_tokens"])
        for chunk in chunks
        if chunk["content_type"] == "text"
    ]
    by_type = Counter(str(chunk["content_type"]) for chunk in chunks)
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)
    page_marker_ids: set[str] = set()
    page_marker_pages: set[tuple[str, int]] = set()
    page_markers_by_previous_type: Counter[str] = Counter()
    page_marker_refs: list[dict[str, Any]] = []
    for document in documents:
        source_id = str(document["source_id"])
        source_blocks = sorted(
            blocks_by_source.get(source_id, []),
            key=lambda row: int(row["block_order"]),
        )
        detected = _find_pdf_page_marker_blocks(document, source_blocks)
        page_marker_ids.update(detected)
        for index, block in enumerate(source_blocks):
            block_id = str(block["block_id"])
            if block_id not in detected:
                continue
            page = block.get("page")
            if isinstance(page, int):
                page_marker_pages.add((source_id, page))
            previous_type = "none"
            for previous in reversed(source_blocks[:index]):
                if previous.get("page") != page:
                    break
                if str(previous.get("block_id") or "") in detected:
                    continue
                if previous.get("dense_eligible") is True:
                    previous_type = str(previous.get("content_type") or "unknown")
                    break
            page_markers_by_previous_type[previous_type] += 1
            page_marker_refs.append(
                {
                    "source_id": source_id,
                    "block_id": block_id,
                    "block_order": int(block["block_order"]),
                    "physical_page": page,
                    "printed_page_numbers": list(detected[block_id]),
                    "raw_text": str(block.get("text") or ""),
                    "previous_content_type": previous_type,
                }
            )
    covered_ids = {
        str(block_id)
        for chunk in chunks
        for block_id in chunk.get("source_block_ids") or []
    }
    covered_page_marker_ids = covered_ids & page_marker_ids
    tail_modes = Counter(
        str(chunk.get("short_tail_adjustment_mode") or "none")
        for chunk in chunks
        if chunk.get("content_type") == "text"
    )
    new_content_counts = [
        int(chunk.get("new_content_token_count") or 0)
        for chunk in chunks
        if chunk.get("content_type") == "text"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": INPUT_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "strategy_id": config.strategy_id,
        "max_tokens": config.max_tokens,
        "overlap_target_tokens": config.overlap_tokens,
        "min_tail_tokens": config.min_tail_tokens,
        "overlap_policy": "largest_whole_sentence_suffix_at_or_below_target",
        "short_tail_policy": ("merge_then_rebalance_then_safe_overlap_by_new_content"),
        "page_marker_policy": "always_metadata_only_never_in_embedding_text",
        "page_marker_detector_id": PAGE_MARKER_DETECTOR_ID,
        "table_overlap_policy": "header_repeat_only",
        "embedding_text_field": "embedding_text",
        "text_embedding_normalization": TEXT_EMBEDDING_NORMALIZATION_ID,
        "table_embedding_normalization": "preserve_markdown_newlines",
        "token_count_basis": "embedding_text",
        "overlap_token_basis": "normalized_embedding_text",
        "tail_token_basis": "normalized_embedding_text",
        "bm25_source_field": "embedding_text",
        "tokenizer_model": codec.model_name,
        "tokenizer_encoding": codec.encoding_name,
        "tokenizer_version": codec.version,
        "kss_version": EXPECTED_KSS_VERSION,
        "kss_backend": KssSentenceSplitter.backend,
        "kss_num_workers": KssSentenceSplitter.num_workers,
        "kiwipiepy_version": EXPECTED_KIWI_VERSION,
        "bm25_pos_policy": BM25_POS_POLICY_ID,
        "bm25_excluded_pos_prefixes": list(BM25_EXCLUDED_POS_PREFIXES),
        "bm25_token_normalization": BM25_TOKEN_NORMALIZATION,
        "document_count": len(documents),
        "block_count": len(blocks),
        "eligible_text_block_count": sum(_is_advanced_text(row) for row in blocks),
        "required_vector_text_block_count": (
            sum(_is_advanced_text(row) for row in blocks) - len(page_marker_ids)
        ),
        "eligible_table_block_count": sum(_is_advanced_table(row) for row in blocks),
        "chunk_count": len(chunks),
        "chunks_by_content_type": dict(sorted(by_type.items())),
        "oversized_sentence_chunk_count": sum(
            bool(chunk.get("oversized_sentence_split")) for chunk in chunks
        ),
        "kss_input_sanitized_chunk_count": sum(
            bool(chunk.get("kss_input_sanitized")) for chunk in chunks
        ),
        "kss_input_sanitized_stream_count": len(
            {
                (str(chunk.get("source_id")), str(chunk.get("stream_id")))
                for chunk in chunks
                if chunk.get("kss_input_sanitized")
            }
        ),
        "kss_alignment_fallback_chunk_count": sum(
            bool(chunk.get("kss_alignment_fallback")) for chunk in chunks
        ),
        "kss_alignment_fallback_stream_count": len(
            {
                (str(chunk.get("source_id")), str(chunk.get("stream_id")))
                for chunk in chunks
                if chunk.get("kss_alignment_fallback")
            }
        ),
        "short_tail_adjusted_chunk_count": sum(
            bool(chunk.get("short_tail_adjusted")) for chunk in chunks
        ),
        "short_tail_adjustment_mode_counts": dict(sorted(tail_modes.items())),
        "short_tail_token_overlap_fallback_count": sum(
            bool(chunk.get("short_tail_token_overlap_fallback")) for chunk in chunks
        ),
        "page_marker_block_count": len(page_marker_ids),
        "page_marker_pdf_page_count": len(page_marker_pages),
        "page_marker_vectorized_block_count": len(covered_page_marker_ids),
        "page_marker_metadata_only_block_count": len(
            page_marker_ids - covered_page_marker_ids
        ),
        "page_marker_refs": page_marker_refs,
        "page_markers_by_previous_content_type": dict(
            sorted(page_markers_by_previous_type.items())
        ),
        "split_table_count": len(
            {
                str(chunk["table_id"])
                for chunk in chunks
                if chunk["content_type"] == "table"
                and int(chunk.get("table_part_count") or 0) > 1
            }
        ),
        "oversized_table_row_chunk_count": sum(
            "oversized_table_row_split" in (chunk.get("quality_flags") or [])
            for chunk in chunks
        ),
        "text_chunks_with_zero_bm25_tokens": sum(
            chunk["content_type"] == "text" and not chunk.get("bm25_tokens")
            for chunk in chunks
        ),
        "bm25_token_total": sum(
            int(chunk.get("bm25_token_count") or 0) for chunk in chunks
        ),
        "new_content_token_min": min(new_content_counts, default=0),
        "new_content_token_mean": (
            round(statistics.fmean(new_content_counts), 2)
            if new_content_counts
            else 0.0
        ),
        "new_content_token_max": max(new_content_counts, default=0),
        "token_min": min(token_counts, default=0),
        "token_mean": (
            round(statistics.fmean(token_counts), 2) if token_counts else 0.0
        ),
        "token_max": max(token_counts, default=0),
        "overlap_actual_min": min(overlaps, default=0),
        "overlap_actual_mean": (
            round(statistics.fmean(overlaps), 2) if overlaps else 0.0
        ),
        "overlap_actual_max": max(overlaps, default=0),
        "validation": dict(validation),
    }


__all__ = [
    "AdvancedChunkConfig",
    "AdvancedChunkingResult",
    "AdvancedTextStream",
    "BM25_EXCLUDED_POS_PREFIXES",
    "BM25_POS_POLICY_ID",
    "BM25_TOKEN_NORMALIZATION",
    "CORPUS_ID",
    "EXPECTED_KIWI_VERSION",
    "EXPECTED_KSS_VERSION",
    "KssAlignmentError",
    "KssSentenceSplitter",
    "KiwiBm25Tokenizer",
    "KSS_ALIGNMENT_DECORATIVE_CHARS",
    "PAGE_MARKER_DETECTOR_ID",
    "PackedTextChunk",
    "SCHEMA_VERSION",
    "STRATEGY_ID",
    "SentenceSpan",
    "TEXT_EMBEDDING_NORMALIZATION_ID",
    "TiktokenCodec",
    "align_kss_sentences",
    "build_advanced_chunk_corpus",
    "build_advanced_chunk_record",
    "build_advanced_streams",
    "build_advanced_summary",
    "build_advanced_text_streams",
    "chunk_advanced_corpus",
    "chunk_advanced_table_block",
    "chunk_advanced_text_stream",
    "extract_page_marker_numbers",
    "normalize_text_for_embedding",
    "pack_sentence_spans",
    "validate_advanced_chunks",
]
