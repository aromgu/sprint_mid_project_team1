"""구조 전처리 결과를 Naive RAG용 512/102 토큰 청크로 나눈다.

이 모듈은 :mod:`src.preprocessing.clean_text`가 만든 메모리 결과만 입력으로
받는다. 일반 본문은 같은 PDF 페이지 또는 HWP 섹션 안에서 이어 붙인 뒤
LangChain ``RecursiveCharacterTextSplitter``로 문단→줄→공백→문자
경계 순서로 나누고, 표는 본문과 섞지 않고 Markdown 행 단위로 나눈다.

중요한 청킹 원칙
-----------------
* ``cl100k_base`` 기준 최종 임베딩 문자열이 최대 512토큰이어야 한다.
* 같은 본문 stream의 인접 청크는 원문 토큰 축에서 102토큰을 겹친다.
* Recursive splitter가 의미 경계를 고르고, 기존 토큰 맵이 UTF-8 안전성과
  정확한 오버랩 및 원본 위치 좌표를 보장한다.
* PDF 페이지, HWP 섹션, 표 경계가 바뀌면 overlap을 넘기지 않는다.
* ``index_policy``가 ``index`` 또는 ``flatten``인 블록만 청킹한다.
* 표는 HTML 없이 Markdown 행과 헤더를 보존한다.
* 이미지 placeholder·Base64·data URI는 임베딩 문자열에 넣지 않는다.
"""

from __future__ import annotations

import bisect
import importlib.metadata
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocessing.clean_text import PreprocessingResult

SCHEMA_VERSION = "rfp_naive_chunk_v1"
STRATEGY_ID = "naive_langchain_recursive_cl100k_base_512_102_v2"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_ENCODING = "cl100k_base"
EXPECTED_TIKTOKEN_VERSION = "0.13.0"
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 102
RECURSIVE_SEPARATORS = ("\n\n", "\n", " ", "")
TEXT_SPLITTER_NAME = "RecursiveCharacterTextSplitter"
INDEXABLE_POLICIES = frozenset({"index", "flatten"})

MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(image://[^)]+\)", re.IGNORECASE)
IMAGE_URI = re.compile(r"image://[^\s)]+", re.IGNORECASE)
FORBIDDEN_PAYLOAD = re.compile(r"(?:data\s*:\s*image|base64\s*,)", re.IGNORECASE)
HTML_TABLE_TAG = re.compile(
    r"</?(?:table|caption|tr|th|td|img|p|li|br)\b",
    re.IGNORECASE,
)
MARKDOWN_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


class TokenCodec(Protocol):
    """청킹 코드가 사용하는 최소 토크나이저 인터페이스다."""

    model_name: str
    encoding_name: str
    version: str

    def encode(self, text: str) -> list[int]:
        """문자열을 토큰 ID 목록으로 바꾼다."""

    def token_bytes(self, token_id: int) -> bytes:
        """한 토큰이 나타내는 원시 바이트를 반환한다."""


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """첫 번째 Naive 청킹 실험의 재현 가능한 설정이다."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    model_name: str = DEFAULT_MODEL
    encoding_name: str = DEFAULT_ENCODING
    strategy_id: str = STRATEGY_ID


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """임베딩 후보 청크와 결정적 품질 요약을 함께 반환한다."""

    chunks: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextStream:
    """같은 위치 경계 안에서 이어 붙인 일반 텍스트 블록 묶음이다."""

    stream_id: str
    stream_order: int
    blocks: tuple[dict[str, Any], ...]
    text: str
    block_char_spans: tuple[tuple[int, int, dict[str, Any]], ...]


@dataclass(frozen=True, slots=True)
class MarkdownTableSegment:
    """캡션·헤더·행으로 구성된 Markdown 표 한 개다."""

    segment_index: int
    context_lines: tuple[str, ...]
    header_lines: tuple[str, ...]
    data_rows: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableUnit:
    """표의 온전한 행 또는 너무 긴 한 행을 나눈 조각이다."""

    row_number: int
    fragment_index: int
    text: str
    oversized_row_split: bool


class TableRowBudgetError(ValueError):
    """반복 헤더 때문에 표 행 조각을 한 글자도 넣지 못할 때 발생한다."""


class TiktokenCodec:
    """OpenAI 임베딩 모델과 동일한 tiktoken 인코딩을 제공한다."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        encoding_name: str = DEFAULT_ENCODING,
    ) -> None:
        """모델 매핑과 패키지 버전을 확인하고 인코딩을 불러온다."""
        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - 설치 안내 경로
            raise RuntimeError(
                "tiktoken이 없습니다. 프로젝트 폴더에서 uv sync를 실행하세요."
            ) from error

        resolved_name = tiktoken.encoding_name_for_model(model_name)
        if resolved_name != encoding_name:
            raise RuntimeError(
                f"모델 토크나이저 매핑이 바뀌었습니다: {model_name} -> "
                f"{resolved_name} (기대값: {encoding_name})"
            )
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.model_name = model_name
        self.encoding_name = encoding_name
        self.version = importlib.metadata.version("tiktoken")
        if self.version != EXPECTED_TIKTOKEN_VERSION:
            raise RuntimeError(
                f"tiktoken 버전이 다릅니다: {self.version} "
                f"(기대값: {EXPECTED_TIKTOKEN_VERSION})"
            )

    def encode(self, text: str) -> list[int]:
        """특수 토큰 모양의 문자열도 문서 원문으로 보고 토큰화한다."""
        return self._encoding.encode_ordinary(text)

    def token_bytes(self, token_id: int) -> bytes:
        """UTF-8 문자 경계를 검증할 수 있도록 단일 토큰 바이트를 반환한다."""
        return self._encoding.decode_single_token_bytes(token_id)


class TokenTextMap:
    """원문 토큰 위치와 손실 없는 UTF-8 문자 위치를 연결한다."""

    def __init__(self, text: str, codec: TokenCodec) -> None:
        """토큰 바이트가 원문 UTF-8과 정확히 같은지 즉시 검증한다."""
        self.text = text
        self.token_ids = tuple(codec.encode(text))
        token_byte_offsets = [0]
        rebuilt = bytearray()
        for token_id in self.token_ids:
            token_value = codec.token_bytes(token_id)
            rebuilt.extend(token_value)
            token_byte_offsets.append(len(rebuilt))

        self.utf8 = text.encode("utf-8")
        if bytes(rebuilt) != self.utf8:
            raise ValueError("토크나이저 바이트와 원문 UTF-8이 일치하지 않습니다")
        self.token_byte_offsets = tuple(token_byte_offsets)

        byte_to_char = {0: 0}
        byte_offset = 0
        for char_index, char in enumerate(text, start=1):
            byte_offset += len(char.encode("utf-8"))
            byte_to_char[byte_offset] = char_index
        self.byte_to_char = byte_to_char
        self.char_to_byte = {
            char_index: offset for offset, char_index in byte_to_char.items()
        }
        self.safe_token_indices = tuple(
            index
            for index, offset in enumerate(self.token_byte_offsets)
            if offset in self.byte_to_char
        )
        self.safe_token_index_set = frozenset(self.safe_token_indices)
        self.token_to_char = {
            index: self.byte_to_char[self.token_byte_offsets[index]]
            for index in self.safe_token_indices
        }
        self.char_to_token = {
            char_index: index for index, char_index in self.token_to_char.items()
        }

    def __len__(self) -> int:
        """원문 전체의 토큰 수를 반환한다."""
        return len(self.token_ids)

    def slice(self, start_token: int, end_token: int) -> str:
        """안전한 토큰 경계 사이의 원문을 대체문자 없이 반환한다."""
        if (
            start_token not in self.safe_token_index_set
            or end_token not in self.safe_token_index_set
        ):
            raise ValueError("UTF-8 문자 중간에서 원문을 자를 수 없습니다")
        start_byte = self.token_byte_offsets[start_token]
        end_byte = self.token_byte_offsets[end_token]
        return self.utf8[start_byte:end_byte].decode("utf-8")

    def byte_span(self, start_token: int, end_token: int) -> tuple[int, int]:
        """토큰 구간을 원문의 UTF-8 바이트 구간으로 변환한다."""
        return self.token_byte_offsets[start_token], self.token_byte_offsets[end_token]


def validate_config(config: ChunkConfig) -> None:
    """청크 크기와 중복 설정이 진행 가능한지 검사한다."""
    if config.max_tokens <= 0:
        raise ValueError("max_tokens는 양수여야 합니다")
    if not 0 <= config.overlap_tokens < config.max_tokens:
        raise ValueError("overlap_tokens는 0 이상이고 max_tokens보다 작아야 합니다")
    if not config.model_name or not config.encoding_name or not config.strategy_id:
        raise ValueError("모델·인코딩·전략 ID는 비어 있을 수 없습니다")


def clean_chunk_text(value: Any) -> str:
    """이미지 참조를 제거하되 나머지 원문 공백과 Markdown은 보존한다."""
    text = str(value or "")
    if FORBIDDEN_PAYLOAD.search(text):
        raise ValueError("청킹 입력에서 금지된 이미지 payload를 발견했습니다")
    text = MARKDOWN_IMAGE.sub("", text)
    return IMAGE_URI.sub("", text)


def block_chunk_text(block: Mapping[str, Any]) -> str:
    """블록 유형에 맞는 청킹 원문을 고른다.

    표는 최신 팀 결정대로 Markdown 표시 내용을 사용한다. 일반 본문은 검색용
    평문을 사용하고, 이미지 placeholder는 어느 유형에서도 제거한다.
    """
    if block.get("block_type") == "table":
        display = clean_chunk_text(block.get("display_content"))
        if HTML_TABLE_TAG.search(display):
            raise ValueError("Naive RAG 표에 HTML 태그가 포함되어 있습니다")
        return display
    return clean_chunk_text(block.get("retrieval_text"))


def is_indexable(block: Mapping[str, Any]) -> bool:
    """검색 정책이 index/flatten이고 실제 청킹 텍스트가 있는지 판단한다."""
    return block.get("index_policy") in INDEXABLE_POLICIES and bool(
        block_chunk_text(block).strip()
    )


def select_indexable_blocks(
    blocks: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """짧은 텍스트도 버리지 않고 검색 대상 블록만 선택한다."""
    return [block for block in blocks if is_indexable(block)]


def boundary_key(
    document: Mapping[str, Any], block: Mapping[str, Any]
) -> tuple[Any, ...]:
    """PDF는 페이지, HWP/HWPX는 논리 섹션을 절대 경계로 삼는다."""
    if str(document.get("file_type") or "").casefold() == "pdf":
        return (document["source_id"], "page", block.get("page"))
    return (
        document["source_id"],
        "section",
        block.get("section_idx"),
        block.get("section_path") or "본문",
    )


def should_break_on_excluded(block: Mapping[str, Any]) -> bool:
    """이미지·표·문서 부속물이 빠진 자리에서는 본문 stream을 끊는다."""
    return (
        block.get("block_type") in {"picture", "table"}
        or block.get("scope") != "body"
        or bool(block.get("furniture_type"))
    )


def join_text_blocks(
    blocks: Sequence[dict[str, Any]],
) -> tuple[str, tuple[tuple[int, int, dict[str, Any]], ...]]:
    """일반 텍스트 블록을 두 줄 간격으로 잇고 각 원문 위치를 기록한다."""
    parts: list[str] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for block_index, block in enumerate(blocks):
        if block_index:
            parts.append("\n\n")
            cursor += 2
        text = block_chunk_text(block)
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, block))
    return "".join(parts), tuple(spans)


def build_streams(
    document: Mapping[str, Any],
    blocks: Sequence[dict[str, Any]],
) -> list[dict[str, Any] | TextStream]:
    """문서 순서를 유지하면서 일반 텍스트 묶음과 독립 표를 만든다."""
    streams: list[dict[str, Any] | TextStream] = []
    current_blocks: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    stream_order = 0

    def flush_text() -> None:
        """모아 둔 일반 텍스트 블록을 위치 stream 하나로 확정한다."""
        nonlocal current_blocks, current_key, stream_order
        if not current_blocks:
            return
        stream_order += 1
        text, spans = join_text_blocks(current_blocks)
        streams.append(
            TextStream(
                stream_id=f"{document['source_id']}:S{stream_order:06d}",
                stream_order=stream_order,
                blocks=tuple(current_blocks),
                text=text,
                block_char_spans=spans,
            )
        )
        current_blocks = []
        current_key = None

    for block in sorted(blocks, key=lambda row: int(row["block_order"])):
        if not is_indexable(block):
            if should_break_on_excluded(block):
                flush_text()
            continue
        if block["block_type"] == "table":
            flush_text()
            stream_order += 1
            table_stream = dict(block)
            table_stream["stream_id"] = f"{document['source_id']}:S{stream_order:06d}"
            table_stream["stream_order"] = stream_order
            streams.append(table_stream)
            continue

        key = boundary_key(document, block)
        if current_blocks and key != current_key:
            flush_text()
        current_key = key
        current_blocks.append(block)
    flush_text()
    return streams


def location_label(document: Mapping[str, Any], block: Mapping[str, Any]) -> str:
    """검색 문맥에 넣을 사람 친화적인 PDF/HWP 위치를 만든다."""
    if str(document.get("file_type") or "").casefold() == "pdf":
        return f"PDF {block['page']}쪽"
    return str(block.get("section_path") or f"섹션 {block.get('section_idx', 0)}")


def build_context_prefix(
    document: Mapping[str, Any],
    block: Mapping[str, Any],
    content_type: str,
) -> str:
    """임베딩 문자열에 포함할 최소 출처 문맥을 만든다."""
    type_label = "표" if content_type.startswith("table") else "본문"
    return (
        f"[문서] {document['source_filename']}\n"
        f"[위치] {location_label(document, block)}\n"
        f"[유형] {type_label}"
    )


def make_retrieval_text(prefix: str, raw_text: str) -> str:
    """실제 임베딩에 전달할 문맥 포함 문자열을 만든다."""
    return f"{prefix}\n\n{raw_text}"


def find_max_safe_end(
    token_map: TokenTextMap,
    start_token: int,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
    overlap_tokens: int,
) -> int:
    """최종 문자열이 토큰 상한 이하인 가장 먼 UTF-8 경계를 찾는다."""
    if len(token_map) == start_token:
        return start_token
    remaining_text = token_map.slice(start_token, len(token_map))
    if (
        len(codec.encode(make_retrieval_text(prefix, remaining_text)))
        <= config.max_tokens
    ):
        return len(token_map)

    prefix_tokens = len(codec.encode(f"{prefix}\n\n"))
    estimated_body = max(1, config.max_tokens - prefix_tokens)
    upper_token = min(len(token_map) - 1, start_token + estimated_body + 16)
    upper_position = bisect.bisect_right(token_map.safe_token_indices, upper_token)
    candidates = token_map.safe_token_indices[:upper_position]

    for end_token in reversed(candidates):
        if end_token <= start_token:
            break
        if overlap_tokens and end_token - overlap_tokens <= start_token:
            continue
        if (
            overlap_tokens
            and end_token - overlap_tokens not in token_map.safe_token_index_set
        ):
            continue
        raw_text = token_map.slice(start_token, end_token)
        if (
            len(codec.encode(make_retrieval_text(prefix, raw_text)))
            <= config.max_tokens
        ):
            return end_token
    raise ValueError("문서 문맥이 너무 길어 한 토큰도 안전하게 청킹할 수 없습니다")


def build_recursive_character_splitter(
    codec: TokenCodec,
    chunk_size: int,
    overlap_tokens: int,
) -> RecursiveCharacterTextSplitter:
    """프로젝트 토크나이저를 사용하는 LangChain 재귀 분할기를 만든다.

    테스트용 문자 코덱과 운영용 tiktoken 코덱이 같은 경로를 사용하도록
    ``from_tiktoken_encoder`` 대신 프로젝트 ``codec.encode``를 길이 함수로
    주입한다. 문맥 prefix는 본문 밖에서 반복되므로 호출자가 차감한 실제
    본문 예산을 ``chunk_size``로 전달해야 한다.
    """
    if chunk_size <= 0:
        raise ValueError("Recursive splitter의 chunk_size는 양수여야 합니다")
    if not 0 <= overlap_tokens < chunk_size:
        raise ValueError("Recursive splitter의 overlap은 chunk_size보다 작아야 합니다")
    return RecursiveCharacterTextSplitter(
        separators=list(RECURSIVE_SEPARATORS),
        keep_separator="end",
        is_separator_regex=False,
        chunk_size=chunk_size,
        chunk_overlap=overlap_tokens,
        length_function=lambda value: len(codec.encode(value)),
        strip_whitespace=False,
    )


def choose_recursive_end(
    token_map: TokenTextMap,
    start_token: int,
    max_end_token: int,
    overlap_tokens: int,
    codec: TokenCodec,
) -> int:
    """LangChain이 고른 의미 경계를 가장 가까운 안전한 토큰 경계로 맞춘다.

    ``find_max_safe_end``가 문맥을 포함한 512토큰 상한을 먼저 계산한다.
    LangChain splitter는 남은 원문에서 문단→줄→공백→문자 순으로 첫
    경계를 고르고, 이 함수는 UTF-8 문자 중간을 자르지 않는 토큰 경계로만
    결과를 정렬한다. 다음 청크의 정확한 토큰 오버랩은 바깥 루프가 맡는다.
    """
    if max_end_token == len(token_map):
        return max_end_token

    safe_body = token_map.slice(start_token, max_end_token)
    body_budget = len(codec.encode(safe_body))
    splitter = build_recursive_character_splitter(
        codec,
        body_budget,
        overlap_tokens,
    )
    # 첫 경계를 얻는 데 뒤쪽 stream 전체를 매번 다시 분할할 필요는 없다.
    # 본문 예산을 처음 초과하는 안전한 경계까지만 probe해 대용량 문서에서도
    # 청크 수에 대해 제곱으로 느려지는 일을 막는다.
    probe_end_token = max_end_token
    for candidate in token_map.safe_token_indices:
        if candidate <= max_end_token:
            continue
        probe_text = token_map.slice(start_token, candidate)
        probe_end_token = candidate
        if len(codec.encode(probe_text)) > body_budget:
            break
    remaining_text = token_map.slice(start_token, probe_end_token)
    parts = splitter.split_text(remaining_text)
    if not parts:
        raise ValueError("RecursiveCharacterTextSplitter가 빈 결과를 반환했습니다")

    first_part = parts[0]
    if not remaining_text.startswith(first_part):
        # strip_whitespace=False와 keep_separator='end' 계약이 바뀌면 원문
        # 좌표를 안전하게 보존할 수 없으므로 조용히 진행하지 않는다.
        raise ValueError("Recursive splitter 결과가 원문 시작 위치와 다릅니다")

    start_char = token_map.token_to_char[start_token]
    target_char = start_char + len(first_part)
    for boundary_token in reversed(token_map.safe_token_indices):
        if boundary_token > max_end_token or boundary_token <= start_token:
            continue
        if token_map.token_to_char[boundary_token] > target_char:
            continue
        if overlap_tokens and (
            boundary_token - overlap_tokens <= start_token
            or boundary_token - overlap_tokens not in token_map.safe_token_index_set
        ):
            continue
        return boundary_token
    return max_end_token


def split_text_token_ranges(
    text: str,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
    overlap_tokens: int | None = None,
) -> tuple[TokenTextMap, list[tuple[int, int, int]]]:
    """원문을 손실 없이 나누고 ``(시작, 끝, 이전 중복)``을 반환한다."""
    overlap = config.overlap_tokens if overlap_tokens is None else overlap_tokens
    if not text:
        return TokenTextMap(text, codec), []
    token_map = TokenTextMap(text, codec)
    ranges: list[tuple[int, int, int]] = []
    start_token = 0
    previous_end = 0
    while start_token < len(token_map):
        max_end = find_max_safe_end(
            token_map,
            start_token,
            prefix,
            codec,
            config,
            overlap,
        )
        end_token = choose_recursive_end(
            token_map,
            start_token,
            max_end,
            overlap,
            codec,
        )
        previous_overlap = previous_end - start_token if ranges else 0
        ranges.append((start_token, end_token, previous_overlap))
        if end_token >= len(token_map):
            break
        next_start = end_token - overlap
        if next_start <= start_token:
            raise ValueError("중복 설정 때문에 청크가 앞으로 진행하지 못합니다")
        previous_end = end_token
        start_token = next_start
    return token_map, ranges


def blocks_for_token_range(
    stream: TextStream,
    token_map: TokenTextMap,
    start_token: int,
    end_token: int,
) -> list[dict[str, Any]]:
    """청크 구간과 실제로 겹치는 원본 블록을 순서대로 찾는다."""
    start_byte, end_byte = token_map.byte_span(start_token, end_token)
    selected: list[dict[str, Any]] = []
    for start_char, end_char, block in stream.block_char_spans:
        block_start_byte = token_map.char_to_byte[start_char]
        block_end_byte = token_map.char_to_byte[end_char]
        if block_start_byte < end_byte and block_end_byte > start_byte:
            selected.append(block)
    return selected


def non_null_range(blocks: Sequence[Mapping[str, Any]], field: str) -> tuple[Any, Any]:
    """위치 필드의 null을 제외한 첫 값과 마지막 값을 반환한다."""
    values = [block.get(field) for block in blocks if block.get(field) is not None]
    return (values[0], values[-1]) if values else (None, None)


def unique_in_order(values: Iterable[Any]) -> list[Any]:
    """입력 순서를 유지하면서 중복값을 한 번만 남긴다."""
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        marker = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def inherited_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """원본 경로 객체나 미검증 본문 없이 안전한 출처 필드만 복사한다."""
    return {
        "source_id": document["source_id"],
        "document_id": document["document_id"],
        "source_sha256": document["source_sha256"],
        "source_filename": document["source_filename"],
        "source_relative_path": document.get("source_relative_path"),
        "filename_aliases": list(document.get("filename_aliases") or []),
        "file_type": document["file_type"],
    }


def build_chunk_record(
    *,
    document: Mapping[str, Any],
    source_blocks: Sequence[dict[str, Any]],
    raw_text: str,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
    content_type: str,
    stream_id: str,
    stream_order: int,
    stream_part_index: int,
    overlap_from_previous_tokens: int,
    stream_token_start: int | None,
    stream_token_end: int | None,
    table_fields: Mapping[str, Any] | None = None,
    extra_quality_flags: Iterable[str] = (),
) -> dict[str, Any]:
    """본문과 표가 공유하는 JSON 직렬화 가능한 청크 레코드를 만든다."""
    if not source_blocks:
        raise ValueError("청크의 원본 블록이 비어 있습니다")
    if not raw_text.strip():
        raise ValueError("빈 raw_text 청크를 만들 수 없습니다")

    retrieval_text = make_retrieval_text(prefix, raw_text)
    token_count = len(codec.encode(retrieval_text))
    if token_count > config.max_tokens:
        raise ValueError(
            f"청크가 토큰 상한을 넘었습니다: {token_count} > {config.max_tokens}"
        )
    if FORBIDDEN_PAYLOAD.search(retrieval_text) or IMAGE_URI.search(retrieval_text):
        raise ValueError("청크에 이미지 payload 또는 placeholder가 포함되었습니다")

    section_start, section_end = non_null_range(source_blocks, "section_idx")
    para_start, para_end = non_null_range(source_blocks, "para_idx")
    page_start, page_end = non_null_range(source_blocks, "page")
    quality_flags = unique_in_order(
        flag
        for flag in [
            *(
                flag
                for block in source_blocks
                for flag in (block.get("quality_flags") or [])
            ),
            *extra_quality_flags,
        ]
        if flag
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": None,
        "chunk_order": None,
        "strategy_id": config.strategy_id,
        "chunk_size_tokens": config.max_tokens,
        "chunk_overlap_tokens": config.overlap_tokens,
        "tokenizer_model": codec.model_name,
        "tokenizer_encoding": codec.encoding_name,
        "tokenizer_version": codec.version,
        "token_count": token_count,
        "content_token_count": len(codec.encode(raw_text)),
        "overlap_from_previous_tokens": overlap_from_previous_tokens,
        **inherited_metadata(document),
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
        "raw_text": raw_text,
        "retrieval_text": retrieval_text,
        "stream_id": stream_id,
        "stream_order": stream_order,
        "stream_part_index": stream_part_index,
        "stream_token_start": stream_token_start,
        "stream_token_end": stream_token_end,
        "section_path": source_blocks[0].get("section_path") or "본문",
        "section_idx_start": section_start,
        "section_idx_end": section_end,
        "para_idx_start": para_start,
        "para_idx_end": para_end,
        "page_start": page_start,
        "page_end": page_end,
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
        "quality_flags": quality_flags,
    }
    if table_fields:
        record.update(table_fields)
    return record


def chunk_text_stream(
    document: Mapping[str, Any],
    stream: TextStream,
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """같은 위치의 일반 블록을 Recursive 512/102 토큰 청크로 만든다."""
    anchor = stream.blocks[0]
    prefix = build_context_prefix(document, anchor, "text")
    token_map, ranges = split_text_token_ranges(stream.text, prefix, codec, config)
    chunks: list[dict[str, Any]] = []
    for part_index, (start_token, end_token, overlap) in enumerate(ranges, start=1):
        source_blocks = blocks_for_token_range(
            stream,
            token_map,
            start_token,
            end_token,
        )
        chunks.append(
            build_chunk_record(
                document=document,
                source_blocks=source_blocks,
                raw_text=token_map.slice(start_token, end_token),
                prefix=prefix,
                codec=codec,
                config=config,
                content_type="text",
                stream_id=stream.stream_id,
                stream_order=stream.stream_order,
                stream_part_index=part_index,
                overlap_from_previous_tokens=overlap,
                stream_token_start=start_token,
                stream_token_end=end_token,
            )
        )
    return chunks


def is_markdown_table_row(line: str) -> bool:
    """파이프로 시작하고 끝나는 GFM 표 행인지 확인한다."""
    stripped = line.strip()
    return len(stripped) >= 2 and stripped.startswith("|") and stripped.endswith("|")


def is_markdown_separator_row(line: str) -> bool:
    """GFM 헤더 아래의 ``| --- |`` 구분 행인지 확인한다."""
    if not is_markdown_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(
        MARKDOWN_SEPARATOR_CELL.fullmatch(cell) for cell in cells
    )


def escape_markdown_fallback_cell(value: str) -> str:
    """비정형 표 평문을 한 셀짜리 Markdown 행으로 안전하게 바꾼다."""
    return value.replace("\\", "\\\\").replace("|", r"\|").replace("\n", " / ")


def fallback_markdown_table(value: str) -> str:
    """표시 내용이 비어 있거나 깨진 경우에도 HTML 대신 Markdown을 만든다."""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return "| 내용 |\n| --- |\n| (내용 없음) |"
    rows = "\n".join(f"| {escape_markdown_fallback_cell(line)} |" for line in lines)
    return f"| 내용 |\n| --- |\n{rows}"


def parse_markdown_table_segments(value: str) -> list[MarkdownTableSegment]:
    """중첩 표 표시를 캡션과 독립 Markdown 표 단위로 나눈다."""
    lines = value.splitlines()
    segments: list[MarkdownTableSegment] = []
    pending_context: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not is_markdown_table_row(line):
            if line.strip():
                pending_context.append(line.strip())
            index += 1
            continue

        table_rows: list[str] = []
        while index < len(lines) and is_markdown_table_row(lines[index]):
            table_rows.append(lines[index].strip())
            index += 1

        if len(table_rows) >= 2 and is_markdown_separator_row(table_rows[1]):
            header_lines = (table_rows[0], table_rows[1])
            data_rows = tuple(table_rows[2:])
        else:
            header_lines = ("| 내용 |", "| --- |")
            data_rows = tuple(
                f"| {escape_markdown_fallback_cell(row)} |" for row in table_rows
            )

        segments.append(
            MarkdownTableSegment(
                segment_index=len(segments) + 1,
                context_lines=tuple(pending_context),
                header_lines=header_lines,
                data_rows=data_rows,
            )
        )
        pending_context = []

    if pending_context:
        fallback = fallback_markdown_table("\n".join(pending_context)).splitlines()
        segments.append(
            MarkdownTableSegment(
                segment_index=len(segments) + 1,
                context_lines=(),
                header_lines=(fallback[0], fallback[1]),
                data_rows=tuple(fallback[2:]),
            )
        )
    return segments


def segment_header_text(segment: MarkdownTableSegment) -> str:
    """캡션과 GFM 헤더·구분 행을 반복 가능한 문자열로 합친다."""
    return "\n".join([*segment.context_lines, *segment.header_lines])


def table_raw_text(header_text: str, units: Sequence[TableUnit]) -> str:
    """표 헤더와 선택한 행 조각을 Markdown 문자열로 합친다."""
    lines = [header_text, *(unit.text for unit in units)]
    return "\n".join(line for line in lines if line)


def split_oversized_table_row(
    row_text: str,
    row_number: int,
    header_text: str,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[TableUnit]:
    """한 표 행이 예산을 넘을 때만 행 내부를 Recursive 방식으로 나눈다."""
    row_prefix = f"{prefix}\n\n{header_text}" if header_text else prefix
    try:
        token_map, ranges = split_text_token_ranges(
            row_text,
            row_prefix,
            codec,
            config,
            overlap_tokens=0,
        )
    except ValueError as error:
        if str(error) != "문서 문맥이 너무 길어 한 토큰도 안전하게 청킹할 수 없습니다":
            raise
        raise TableRowBudgetError(
            "반복할 표 헤더가 행 분할에 필요한 토큰 예산을 모두 사용했습니다"
        ) from error
    return [
        TableUnit(
            row_number=row_number,
            fragment_index=fragment_index,
            text=token_map.slice(start, end),
            oversized_row_split=True,
        )
        for fragment_index, (start, end, _) in enumerate(ranges, start=1)
    ]


def table_header_exhausts_row_budget(
    segment: MarkdownTableSegment,
    header_text: str,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
) -> bool:
    """헤더 뒤에 행의 첫 안전 문자조차 들어갈 수 없는지 확인한다.

    일부 복합 표는 거의 모든 내용을 첫 행(헤더)에 담고 마지막에 짧은
    데이터 행 하나만 둔다. 헤더 자체가 512토큰 이하여도 남은 예산이 0에
    가까우면 행 내부 분할을 시작할 수 없으므로, 이때는 표 전체를 한 셀
    Markdown으로 평탄화해 RCTS로 나눈다.
    """
    for row_number, row_text in enumerate(segment.data_rows, start=1):
        complete_row = table_raw_text(
            header_text,
            [TableUnit(row_number, 1, row_text, False)],
        )
        if (
            len(codec.encode(make_retrieval_text(prefix, complete_row)))
            <= config.max_tokens
        ):
            continue

        token_map = TokenTextMap(row_text, codec)
        first_safe_end = next(
            (index for index in token_map.safe_token_indices if index > 0),
            None,
        )
        if first_safe_end is None:
            continue
        first_fragment = table_raw_text(
            header_text,
            [
                TableUnit(
                    row_number,
                    1,
                    token_map.slice(0, first_safe_end),
                    True,
                )
            ],
        )
        if (
            len(codec.encode(make_retrieval_text(prefix, first_fragment)))
            > config.max_tokens
        ):
            return True
    return False


def prepare_table_units(
    segment: MarkdownTableSegment,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
) -> tuple[str, list[TableUnit], list[str]]:
    """Markdown 헤더를 고정하고 나머지를 온전한 행 단위로 준비한다."""
    header_text = segment_header_text(segment)
    if len(codec.encode(make_retrieval_text(prefix, header_text))) > config.max_tokens:
        raise ValueError("표 캡션과 헤더만으로 토큰 상한을 초과했습니다")

    units: list[TableUnit] = []
    quality_flags: list[str] = []
    for row_offset, row_text in enumerate(segment.data_rows, start=1):
        candidate = table_raw_text(
            header_text,
            [TableUnit(row_offset, 1, row_text, False)],
        )
        if (
            len(codec.encode(make_retrieval_text(prefix, candidate)))
            <= config.max_tokens
        ):
            units.append(TableUnit(row_offset, 1, row_text, False))
            continue
        units.extend(
            split_oversized_table_row(
                row_text,
                row_offset,
                header_text,
                prefix,
                codec,
                config,
            )
        )
        quality_flags.append("oversized_table_row_split")
    return header_text, units, unique_in_order(quality_flags)


def select_table_overlap_units(
    previous_units: Sequence[TableUnit],
    codec: TokenCodec,
    overlap_tokens: int,
) -> list[TableUnit]:
    """직전 part 끝에서 overlap 이하인 온전한 행만 반복한다."""
    if overlap_tokens <= 0:
        return []
    selected: list[TableUnit] = []
    for unit in reversed(previous_units):
        candidate = [unit, *selected]
        token_count = len(codec.encode("\n".join(item.text for item in candidate)))
        if token_count > overlap_tokens:
            break
        selected = candidate
    return selected


def oversized_header_source_text(segment: MarkdownTableSegment) -> str:
    """너무 긴 표 헤더를 한 셀짜리 Markdown으로 옮길 평문으로 만든다."""
    header_row = segment.header_lines[0].strip()
    if header_row.startswith("|") and header_row.endswith("|"):
        header_row = header_row[1:-1].strip()
    values = [*segment.context_lines, header_row, *segment.data_rows]
    return escape_markdown_fallback_cell("\n".join(value for value in values if value))


def split_oversized_header_ranges(
    value: str,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
) -> tuple[TokenTextMap, list[tuple[int, int]]]:
    """긴 단일 헤더를 유효한 한 셀 Markdown 표 예산에 맞춰 나눈다."""
    token_map = TokenTextMap(value, codec)
    ranges: list[tuple[int, int]] = []
    start_token = 0
    while start_token < len(token_map):
        max_end = start_token
        for end_token in token_map.safe_token_indices:
            if end_token <= start_token:
                continue
            fragment = token_map.slice(start_token, end_token)
            raw_text = f"| 표 내용 |\n| --- |\n| {fragment} |"
            if (
                len(codec.encode(make_retrieval_text(prefix, raw_text)))
                > config.max_tokens
            ):
                break
            max_end = end_token
        if max_end <= start_token:
            raise ValueError("긴 표 헤더를 Markdown 토큰 예산 안에 넣을 수 없습니다")
        end_token = choose_recursive_end(
            token_map,
            start_token,
            max_end,
            0,
            codec,
        )
        ranges.append((start_token, end_token))
        start_token = end_token
    return token_map, ranges


def chunk_oversized_table_header(
    *,
    document: Mapping[str, Any],
    block: dict[str, Any],
    segment: MarkdownTableSegment,
    segment_count: int,
    prefix: str,
    codec: TokenCodec,
    config: ChunkConfig,
    content_type: str,
    fallback_quality_flag: str = "oversized_table_header_split",
) -> list[dict[str, Any]]:
    """표 구조가 예산에 맞지 않으면 한 셀 Markdown 여러 개로 보존한다."""
    source_text = oversized_header_source_text(segment)
    token_map, ranges = split_oversized_header_ranges(
        source_text,
        prefix,
        codec,
        config,
    )
    chunks: list[dict[str, Any]] = []
    header_text = "| 표 내용 |\n| --- |"
    for part_index, (start_token, end_token) in enumerate(ranges, start=1):
        raw_text = f"{header_text}\n| {token_map.slice(start_token, end_token)} |"
        chunks.append(
            build_chunk_record(
                document=document,
                source_blocks=[block],
                raw_text=raw_text,
                prefix=prefix,
                codec=codec,
                config=config,
                content_type=content_type,
                stream_id=block["stream_id"],
                stream_order=int(block["stream_order"]),
                stream_part_index=part_index,
                overlap_from_previous_tokens=0,
                stream_token_start=None,
                stream_token_end=None,
                table_fields={
                    "table_id": block["table_id"],
                    "render_mode": "gfm",
                    "table_segment_index": segment.segment_index,
                    "table_segment_count": segment_count,
                    "table_segment_part_index": part_index,
                    "table_segment_part_count": len(ranges),
                    "table_row_start": None,
                    "table_row_end": None,
                    "table_header_text": header_text,
                    "table_header_repeated": part_index > 1,
                    "table_overlap_mode": (
                        "header_repeat_only" if part_index > 1 else "none"
                    ),
                },
                extra_quality_flags=(
                    fallback_quality_flag,
                    "table_structure_flattened_fallback",
                ),
            )
        )
    return chunks


def chunk_table_segment(
    *,
    document: Mapping[str, Any],
    block: dict[str, Any],
    segment: MarkdownTableSegment,
    segment_count: int,
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """Markdown 표 한 개를 행 경계와 헤더 반복을 지켜 나눈다."""
    content_type = "table_flattened" if block["index_policy"] == "flatten" else "table"
    prefix = build_context_prefix(document, block, content_type)
    if (
        len(codec.encode(make_retrieval_text(prefix, segment_header_text(segment))))
        > config.max_tokens
    ):
        return chunk_oversized_table_header(
            document=document,
            block=block,
            segment=segment,
            segment_count=segment_count,
            prefix=prefix,
            codec=codec,
            config=config,
            content_type=content_type,
        )
    header_text = segment_header_text(segment)
    if table_header_exhausts_row_budget(
        segment,
        header_text,
        prefix,
        codec,
        config,
    ):
        return chunk_oversized_table_header(
            document=document,
            block=block,
            segment=segment,
            segment_count=segment_count,
            prefix=prefix,
            codec=codec,
            config=config,
            content_type=content_type,
            fallback_quality_flag="table_header_budget_exhausted_fallback",
        )
    try:
        header_text, units, preparation_flags = prepare_table_units(
            segment,
            prefix,
            codec,
            config,
        )
    except TableRowBudgetError:
        # BPE 토큰은 앞 문맥에 따라 경계가 달라질 수 있다. 첫 행 조각은
        # 들어가더라도 다음 조각이 헤더 뒤에 들어가지 않는 경우, 표 전체를
        # 한 셀 GFM으로 옮겨 내용 손실 없이 Recursive 방식으로 다시 나눈다.
        return chunk_oversized_table_header(
            document=document,
            block=block,
            segment=segment,
            segment_count=segment_count,
            prefix=prefix,
            codec=codec,
            config=config,
            content_type=content_type,
            fallback_quality_flag="table_header_budget_exhausted_fallback",
        )

    # 빈 표도 헤더 골격 자체를 검색 가능한 Markdown 청크 하나로 남긴다.
    if not units:
        return [
            build_chunk_record(
                document=document,
                source_blocks=[block],
                raw_text=header_text,
                prefix=prefix,
                codec=codec,
                config=config,
                content_type=content_type,
                stream_id=block["stream_id"],
                stream_order=int(block["stream_order"]),
                stream_part_index=1,
                overlap_from_previous_tokens=0,
                stream_token_start=None,
                stream_token_end=None,
                table_fields={
                    "table_id": block["table_id"],
                    "render_mode": "gfm",
                    "table_segment_index": segment.segment_index,
                    "table_segment_count": segment_count,
                    "table_segment_part_index": 1,
                    "table_segment_part_count": 1,
                    "table_row_start": None,
                    "table_row_end": None,
                    "table_header_text": header_text,
                    "table_header_repeated": False,
                    "table_overlap_mode": "none",
                },
                extra_quality_flags=preparation_flags,
            )
        ]

    parts: list[dict[str, Any]] = []
    next_new_index = 0
    previous_new_units: list[TableUnit] = []
    while next_new_index < len(units):
        overlap_units = (
            select_table_overlap_units(
                previous_new_units,
                codec,
                config.overlap_tokens,
            )
            if parts
            else []
        )
        while overlap_units:
            probe = table_raw_text(
                header_text,
                [*overlap_units, units[next_new_index]],
            )
            if (
                len(codec.encode(make_retrieval_text(prefix, probe)))
                <= config.max_tokens
            ):
                break
            overlap_units = overlap_units[1:]

        new_units: list[TableUnit] = []
        cursor = next_new_index
        while cursor < len(units):
            candidate_units = [*overlap_units, *new_units, units[cursor]]
            candidate_text = table_raw_text(header_text, candidate_units)
            if (
                len(codec.encode(make_retrieval_text(prefix, candidate_text)))
                > config.max_tokens
            ):
                break
            new_units.append(units[cursor])
            cursor += 1
        if not new_units:
            raise ValueError(
                f"표 행을 토큰 예산 안에 넣지 못했습니다: {block['block_id']}"
            )

        present_units = [*overlap_units, *new_units]
        overlap_count = (
            len(codec.encode("\n".join(unit.text for unit in overlap_units)))
            if overlap_units
            else 0
        )
        header_repeated = bool(parts)
        if overlap_units and header_repeated:
            overlap_mode = "whole_row_plus_header_repeat"
        elif overlap_units:
            overlap_mode = "whole_row_only"
        elif header_repeated:
            overlap_mode = "header_repeat_only"
        else:
            overlap_mode = "none"

        quality_flags = list(preparation_flags)
        if any(unit.oversized_row_split for unit in new_units):
            quality_flags.append("oversized_table_row_split")
        parts.append(
            build_chunk_record(
                document=document,
                source_blocks=[block],
                raw_text=table_raw_text(header_text, present_units),
                prefix=prefix,
                codec=codec,
                config=config,
                content_type=content_type,
                stream_id=block["stream_id"],
                stream_order=int(block["stream_order"]),
                stream_part_index=len(parts) + 1,
                overlap_from_previous_tokens=overlap_count,
                stream_token_start=None,
                stream_token_end=None,
                table_fields={
                    "table_id": block["table_id"],
                    "render_mode": "gfm",
                    "table_segment_index": segment.segment_index,
                    "table_segment_count": segment_count,
                    "table_segment_part_index": len(parts) + 1,
                    "table_segment_part_count": None,
                    "table_row_start": min(unit.row_number for unit in present_units),
                    "table_row_end": max(unit.row_number for unit in present_units),
                    "table_header_text": header_text,
                    "table_header_repeated": header_repeated,
                    "table_overlap_mode": overlap_mode,
                },
                extra_quality_flags=quality_flags,
            )
        )
        previous_new_units = new_units
        next_new_index = cursor

    for part in parts:
        part["table_segment_part_count"] = len(parts)
    return parts


def chunk_table_block(
    document: Mapping[str, Any],
    block: dict[str, Any],
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """표를 본문과 분리하고 Markdown 표별·행별로 나눈다."""
    markdown = block_chunk_text(block)
    segments = parse_markdown_table_segments(markdown)
    if not segments:
        segments = parse_markdown_table_segments(
            fallback_markdown_table(str(block.get("retrieval_text") or ""))
        )
    if not segments:
        raise ValueError(
            f"검색 대상 표를 Markdown으로 만들 수 없습니다: {block['block_id']}"
        )

    chunks: list[dict[str, Any]] = []
    for segment in segments:
        segment_chunks = chunk_table_segment(
            document=document,
            block=block,
            segment=segment,
            segment_count=len(segments),
            codec=codec,
            config=config,
        )
        for segment_chunk in segment_chunks:
            segment_chunk["stream_part_index"] = len(chunks) + 1
            chunks.append(segment_chunk)

    for part_index, chunk in enumerate(chunks, start=1):
        chunk["table_part_index"] = part_index
        chunk["table_part_count"] = len(chunks)
    return chunks


def chunk_document(
    document: Mapping[str, Any],
    blocks: Sequence[dict[str, Any]],
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """한 문서의 stream을 순서대로 청킹하고 결정적 ID를 부여한다."""
    validate_config(config)
    chunks: list[dict[str, Any]] = []
    for stream in build_streams(document, blocks):
        if isinstance(stream, TextStream):
            chunks.extend(chunk_text_stream(document, stream, codec, config))
        else:
            chunks.extend(chunk_table_block(document, stream, codec, config))

    for chunk_order, chunk in enumerate(chunks, start=1):
        chunk["chunk_order"] = chunk_order
        chunk["chunk_id"] = (
            f"{document['source_id']}:N{config.max_tokens}O{config.overlap_tokens}:"
            f"C{chunk_order:06d}"
        )
    return chunks


def build_chunk_corpus(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    codec: TokenCodec,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """모든 문서를 source_id 순서로 청킹해 입력 순서와 무관하게 만든다."""
    validate_config(config)
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)

    source_ids = [str(document["source_id"]) for document in documents]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("같은 source_id의 전처리 결과가 두 번 입력되었습니다")

    chunks: list[dict[str, Any]] = []
    for document in sorted(documents, key=lambda row: str(row["source_id"])):
        source_blocks = sorted(
            blocks_by_source.get(str(document["source_id"]), []),
            key=lambda row: int(row["block_order"]),
        )
        chunks.extend(chunk_document(document, source_blocks, codec, config))
    return chunks


def validate_chunk_corpus(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    codec: TokenCodec,
    config: ChunkConfig,
) -> dict[str, Any]:
    """누락·토큰 초과·경계 침범·이미지 혼입을 문서 단위로 검사한다."""
    documents_by_id = {str(row["source_id"]): row for row in documents}
    blocks_by_id = {str(row["block_id"]): row for row in blocks}
    eligible_ids = {str(row["block_id"]) for row in blocks if is_indexable(row)}
    excluded_ids = {
        str(row["block_id"]) for row in blocks if row.get("index_policy") == "exclude"
    }
    covered_ids = {
        str(block_id) for chunk in chunks for block_id in chunk["source_block_ids"]
    }
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]

    orders_valid = True
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[str(chunk["source_id"])].append(chunk)
    for source_id, source_chunks in chunks_by_source.items():
        ordered = sorted(source_chunks, key=lambda row: int(row["chunk_order"]))
        expected_orders = list(range(1, len(ordered) + 1))
        expected_ids = [
            f"{source_id}:N{config.max_tokens}O{config.overlap_tokens}:C{order:06d}"
            for order in expected_orders
        ]
        orders_valid &= [row["chunk_order"] for row in ordered] == expected_orders
        orders_valid &= [row["chunk_id"] for row in ordered] == expected_ids

    source_links_valid = all(
        all(
            block_id in blocks_by_id
            and blocks_by_id[block_id]["source_id"] == chunk["source_id"]
            and is_indexable(blocks_by_id[block_id])
            for block_id in chunk["source_block_ids"]
        )
        for chunk in chunks
    )
    token_counts_valid = all(
        chunk["token_count"] == len(codec.encode(chunk["retrieval_text"]))
        and chunk["content_token_count"] == len(codec.encode(chunk["raw_text"]))
        and 1 <= chunk["token_count"] <= config.max_tokens
        for chunk in chunks
    )
    payloads_absent = all(
        not FORBIDDEN_PAYLOAD.search(chunk["retrieval_text"])
        and not IMAGE_URI.search(chunk["retrieval_text"])
        for chunk in chunks
    )

    locations_valid = True
    tables_valid = True
    for chunk in chunks:
        source_blocks = [
            blocks_by_id[block_id] for block_id in chunk["source_block_ids"]
        ]
        document = documents_by_id[str(chunk["source_id"])]
        if str(document["file_type"]).casefold() == "pdf":
            pages = {block.get("page") for block in source_blocks}
            locations_valid &= len(pages) == 1 and None not in pages
        else:
            sections = {
                (block.get("section_idx"), block.get("section_path"))
                for block in source_blocks
            }
            locations_valid &= len(sections) == 1
            locations_valid &= chunk["page_start"] is None
            locations_valid &= chunk["page_end"] is None

        if chunk["content_type"].startswith("table"):
            tables_valid &= len(source_blocks) == 1
            tables_valid &= source_blocks[0]["block_type"] == "table"
            tables_valid &= chunk["table_id"] == source_blocks[0]["table_id"]
            tables_valid &= chunk["render_mode"] == "gfm"
            tables_valid &= not HTML_TABLE_TAG.search(chunk["raw_text"])
            tables_valid &= any(
                is_markdown_table_row(line) for line in chunk["raw_text"].splitlines()
            )
        else:
            tables_valid &= all(
                block["block_type"] != "table" for block in source_blocks
            )
            tables_valid &= chunk["table_id"] is None

    text_overlap_valid = True
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)
    text_sources: dict[str, TokenTextMap] = {}
    for document in documents:
        for stream in build_streams(
            document,
            blocks_by_source.get(str(document["source_id"]), []),
        ):
            if isinstance(stream, TextStream):
                text_sources[stream.stream_id] = TokenTextMap(stream.text, codec)

    stream_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        stream_groups[str(chunk["stream_id"])].append(chunk)
    for stream_id, token_map in text_sources.items():
        ordered = sorted(
            stream_groups.get(stream_id, []),
            key=lambda row: int(row["stream_part_index"]),
        )
        if not ordered:
            text_overlap_valid = False
            continue
        text_overlap_valid &= ordered[0]["stream_token_start"] == 0
        text_overlap_valid &= ordered[-1]["stream_token_end"] == len(token_map)
        for chunk in ordered:
            start = int(chunk["stream_token_start"])
            end = int(chunk["stream_token_end"])
            text_overlap_valid &= chunk["raw_text"] == token_map.slice(start, end)
        for previous, current in zip(ordered, ordered[1:]):
            text_overlap_valid &= (
                current["stream_token_start"]
                == previous["stream_token_end"] - config.overlap_tokens
            )
            text_overlap_valid &= (
                current["overlap_from_previous_tokens"] == config.overlap_tokens
            )

    gates = {
        "document_ids_are_unique": len(documents_by_id) == len(documents),
        "block_ids_are_unique": len(blocks_by_id) == len(blocks),
        "chunk_ids_are_unique": len(chunk_ids) == len(set(chunk_ids)),
        "chunk_orders_and_ids_are_contiguous": bool(orders_valid),
        "all_indexable_blocks_are_covered": covered_ids >= eligible_ids,
        "no_excluded_block_is_covered": not (covered_ids & excluded_ids),
        "source_block_links_are_valid": source_links_valid,
        "token_counts_and_limit_are_valid": token_counts_valid,
        "pdf_page_and_hwp_section_boundaries_are_preserved": bool(locations_valid),
        "tables_are_standalone_gfm_markdown": bool(tables_valid),
        "text_overlap_and_full_coverage_are_valid": bool(text_overlap_valid),
        "no_image_payload_or_placeholder": payloads_absent,
        "all_raw_text_is_nonempty": all(
            bool(str(chunk["raw_text"]).strip()) for chunk in chunks
        ),
        "no_replacement_character": all(
            "\ufffd" not in chunk["retrieval_text"] for chunk in chunks
        ),
    }
    return {
        "overall_pass": all(gates.values()),
        "gates": gates,
        "diagnostics": {
            "missing_indexable_block_ids": sorted(eligible_ids - covered_ids),
            "covered_excluded_block_ids": sorted(excluded_ids & covered_ids),
        },
    }


def build_summary(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    validation: Mapping[str, Any],
    codec: TokenCodec,
    config: ChunkConfig,
) -> dict[str, Any]:
    """실행 시각 없이도 동일 입력이면 같은 청킹 품질 요약을 만든다."""
    token_counts = [int(chunk["token_count"]) for chunk in chunks]
    by_type = Counter(str(chunk["content_type"]) for chunk in chunks)
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": config.strategy_id,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "tokenizer_model": codec.model_name,
        "tokenizer_encoding": codec.encoding_name,
        "tokenizer_version": codec.version,
        "text_splitter_name": TEXT_SPLITTER_NAME,
        "text_splitter_version": importlib.metadata.version("langchain-text-splitters"),
        "document_count": len(documents),
        "block_count": len(blocks),
        "indexable_block_count": sum(is_indexable(block) for block in blocks),
        "excluded_block_count": sum(
            block.get("index_policy") == "exclude" for block in blocks
        ),
        "chunk_count": len(chunks),
        "chunks_by_content_type": dict(sorted(by_type.items())),
        "token_min": min(token_counts, default=0),
        "token_mean": (
            round(statistics.fmean(token_counts), 2) if token_counts else 0.0
        ),
        "token_max": max(token_counts, default=0),
        "validation": dict(validation),
    }


def chunk_preprocessing_results(
    results: Iterable[PreprocessingResult],
    *,
    codec: TokenCodec | None = None,
    config: ChunkConfig | None = None,
) -> ChunkingResult:
    """여러 전처리 결과를 결정적 순서의 청크 corpus로 변환한다."""
    selected_config = config or ChunkConfig()
    validate_config(selected_config)
    selected_codec = codec or TiktokenCodec(
        selected_config.model_name,
        selected_config.encoding_name,
    )
    result_list = list(results)
    if any(not isinstance(result, PreprocessingResult) for result in result_list):
        raise TypeError("results의 모든 항목은 PreprocessingResult여야 합니다")

    documents = [result.document for result in result_list]
    blocks = [block for result in result_list for block in result.blocks]
    chunks = build_chunk_corpus(documents, blocks, selected_codec, selected_config)
    validation = validate_chunk_corpus(
        documents,
        blocks,
        chunks,
        selected_codec,
        selected_config,
    )
    if not validation["overall_pass"]:
        failed = [name for name, passed in validation["gates"].items() if not passed]
        raise ValueError(f"청킹 품질 검증에 실패했습니다: {', '.join(failed)}")
    return ChunkingResult(
        chunks=tuple(chunks),
        summary=build_summary(
            documents,
            blocks,
            chunks,
            validation,
            selected_codec,
            selected_config,
        ),
    )


def chunk_preprocessing_result(
    result: PreprocessingResult,
    *,
    codec: TokenCodec | None = None,
    config: ChunkConfig | None = None,
) -> ChunkingResult:
    """전처리 결과 한 문서를 Naive RAG 청크로 변환한다."""
    return chunk_preprocessing_results([result], codec=codec, config=config)
