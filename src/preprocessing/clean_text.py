"""RFP 원본 문서를 검색 가능한 구조로 전처리한다.

loader가 만든 :class:`SourceDocument`를 받아 HWP/HWPX 또는 PDF를 파싱하고,
본문·표·이미지의 순서와 위치를 보존한 메모리 결과를 반환한다. 이 모듈은
원본이나 전처리 결과를 파일로 저장하지 않는다.

중요한 데이터 원칙
--------------------
* HWP/HWPX는 ``rhwp.parse(...).to_ir()``로 구조를 읽는다.
* PDF 페이지 번호는 사람이 세는 방식과 같은 1부터 시작한다.
* 단순·병합·중첩 표를 모두 GFM Markdown으로 표현한다.
* 이미지는 바이트/Base64 없이 ``image://`` 참조와 메타데이터만 남긴다.
* 화면 표시용 내용과 검색·임베딩용 텍스트를 분리한다.
"""

from __future__ import annotations

import importlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.loader.load_documents import SourceDocument, sha256_file

SCHEMA_VERSION = "rfp_structured_preprocessing_v2"
HWP_FILE_TYPES = frozenset({"hwp", "hwpx"})
PDF_FILE_TYPE = "pdf"
PDF_TABLE_TEXT_COVERAGE_THRESHOLD = 0.8
PDF_TABLE_TEXT_FALLBACK_MIN_CHARS = 40

# 표가 실제 업무 내용을 담는지 판단할 때 사용하는 대표적인 RFP 표현이다.
TABLE_CONTENT_SIGNAL = re.compile(
    r"(?:요구\s*사항|요구사항\s*ID|REQ[-_ ]?\d+|구분|항목|세부내용|기능|성능|보안|"
    r"일정|기간|금액|예산|평가|배점|제출서류|산출물)",
    re.IGNORECASE,
)
TOP_HEADING = re.compile(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|제\s*\d+\s*장)\s*[.)]?\s*.+$")
SAFE_HTTP_URL = re.compile(r"(?i)\bhttps?(?:\\:|:)//[^;\s\"'<>]+")
DATA_URI = re.compile(r"\bdata:[^\s,]*,", re.IGNORECASE)
HTML_TABLE_TAG = re.compile(
    r"</?(?:table|caption|tr|th|td|img|p|li|br)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """문서 한 개를 전처리한 결과다.

    tuple을 사용해 호출한 쪽에서 실수로 블록 순서를 바꾸는 일을 줄인다.
    각 레코드는 이후 chunking 단계가 JSON으로 직렬화하기 쉬운 dict 형태다.
    """

    document: dict[str, Any]
    blocks: tuple[dict[str, Any], ...]
    tables: tuple[dict[str, Any], ...]
    images: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedAnalysisSource:
    """원본 대신 파서가 읽을 검증된 복구·변환 파일 정보다.

    단순 경로만 받으면 다른 문서를 잘못 연결할 수 있다. 그래서 복구 이력에서
    확인한 원본 ID·원본 SHA·분석본 SHA와 검증 출처를 모두 요구한다.
    """

    path: str | Path
    sha256: str
    original_source_id: str
    original_source_sha256: str
    verification_source: str


class PreprocessingDependencyError(RuntimeError):
    """문서 형식에 필요한 파서가 설치되지 않았을 때 발생한다."""


def normalize_text(value: str | None) -> str:
    """원문 의미를 바꾸지 않고 문자·줄바꿈·과도한 빈 줄만 정리한다.

    문장부호와 숫자는 그대로 두며, 여러 번 실행해도 결과가 달라지지 않는다.
    ``str.strip()``을 쓰지 않아 목록이나 예시 코드의 첫 들여쓰기도 보존한다.
    """
    text = unicodedata.normalize("NFC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    result: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        result.append("" if is_blank else line)
        previous_blank = is_blank

    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()
    return "\n".join(result)


def compact_text(value: str | None) -> str:
    """표 분류와 검색에 쓰도록 여러 공백과 줄바꿈을 한 칸으로 줄인다."""
    return " ".join(normalize_text(value).split())


def kind_name(block: Any) -> str:
    """rhwp 블록 종류를 버전 차이에 강한 소문자 문자열로 통일한다."""
    kind = getattr(block, "kind", type(block).__name__)
    return str(getattr(kind, "value", kind)).casefold()


def provenance(block: Any) -> tuple[int | None, int | None]:
    """HWP 블록의 원본 섹션 번호와 문단 번호를 안전하게 읽는다."""
    source_position = getattr(block, "prov", None)
    return (
        getattr(source_position, "section_idx", None),
        getattr(source_position, "para_idx", None),
    )


def child_blocks(block: Any) -> list[Any]:
    """블록의 직접 자식을 문서 순서대로 반환한다.

    표는 셀을 행·열 순서로 펼친다. 표와 이미지의 캡션은 부모 메타데이터이므로
    여기서 다시 순회하지 않아 같은 내용이 두 번 생기는 것을 막는다.
    """
    if kind_name(block) == "table":
        cells = sorted(
            getattr(block, "cells", []) or [],
            key=lambda cell: (
                int(getattr(cell, "row", 0) or 0),
                int(getattr(cell, "col", 0) or 0),
            ),
        )
        return [
            child for cell in cells for child in (getattr(cell, "blocks", []) or [])
        ]
    return list(getattr(block, "blocks", []) or [])


def walk_blocks_with_depth(
    blocks: Iterable[Any], table_depth: int = 0
) -> Iterable[tuple[Any, int]]:
    """블록을 원래 순서대로 순회하며 현재 표 중첩 깊이도 반환한다."""
    for block in blocks:
        yield block, table_depth
        next_depth = table_depth + 1 if kind_name(block) == "table" else table_depth
        yield from walk_blocks_with_depth(child_blocks(block), next_depth)


def block_plain_text(block: Any) -> str:
    """블록에서 사람이 읽을 수 있는 기본 텍스트를 고른다."""
    kind = kind_name(block)
    if kind == "formula":
        return str(getattr(block, "text_alt", "") or getattr(block, "script", "") or "")
    if kind == "field":
        return str(getattr(block, "cached_value", "") or "")
    if kind == "toc":
        return "\n".join(
            str(getattr(entry, "text", "") or "")
            for entry in (getattr(block, "entries", []) or [])
            if getattr(entry, "text", "")
        )
    return str(getattr(block, "text", "") or "")


def list_item_display_text(block: Any) -> str:
    """목록의 들여쓰기·번호/글머리표와 본문을 함께 보존한다."""
    enumerated = bool(getattr(block, "enumerated", False))
    marker = normalize_text(str(getattr(block, "marker", "") or ""))
    marker = marker or ("1." if enumerated else "-")
    level = max(_optional_int(getattr(block, "level", 0)) or 0, 0)
    text = normalize_text(block_plain_text(block))
    return f"{'  ' * level}{marker} {text}".rstrip()


def note_display_text(block: Any) -> str:
    """각주·미주 번호와 안쪽 내용을 함께 표시한다."""
    kind = kind_name(block)
    label = "각주" if kind == "footnote" else "미주"
    number = _optional_int(getattr(block, "number", None))
    heading = f"[{label} {number}]" if number is not None else f"[{label}]"
    child_text = normalize_text(
        "\n\n".join(flatten_block_for_retrieval(child) for child in child_blocks(block))
    )
    return f"{heading}\n\n{child_text}" if child_text else heading


def toc_entries_metadata(block: Any) -> list[dict[str, Any]]:
    """목차 항목의 표시 계층과 저장 시점 이동 정보를 보존한다."""
    entries: list[dict[str, Any]] = []
    for entry in getattr(block, "entries", []) or []:
        section_idx, para_idx = provenance(entry)
        entries.append(
            {
                "text": normalize_text(str(getattr(entry, "text", "") or "")),
                "level": _optional_int(getattr(entry, "level", None)),
                "target_bookmark_name": str(
                    getattr(entry, "target_bookmark_name", "") or ""
                )
                or None,
                "target_section_idx": _optional_int(
                    getattr(entry, "target_section_idx", None)
                ),
                "cached_page": _optional_int(getattr(entry, "cached_page", None)),
                "is_stale": bool(getattr(entry, "is_stale", False)),
                "section_idx": section_idx,
                "para_idx": para_idx,
            }
        )
    return entries


def has_nested_table(block: Any) -> bool:
    """표 셀 안에 또 다른 표가 있는지 확인한다."""
    return any(
        kind_name(child) == "table"
        for child, _ in walk_blocks_with_depth(child_blocks(block))
    )


def table_markdown_mode(_block: Any) -> str:
    """Naive RAG의 모든 표는 구조와 관계없이 GFM Markdown을 사용한다."""
    return "gfm"


def field_parts(block: Any) -> dict[str, str]:
    """HWP 필드를 표시용·검색용·감사용 값으로 분리한다.

    안전한 HTTP(S) 링크는 검색에 남긴다. 계산식과 입력 제어문은 일반 문장처럼
    임베딩하지 않고, 원래 명령은 최대 500자 감사 메타데이터로만 보존한다.
    """
    field_kind_value = getattr(block, "field_kind", "")
    field_kind = str(
        getattr(field_kind_value, "value", field_kind_value) or "unknown"
    ).casefold()
    cached = normalize_text(str(getattr(block, "cached_value", "") or ""))
    raw = normalize_text(str(getattr(block, "raw_instruction", "") or ""))[:500]

    if cached:
        return {
            "display_text": cached,
            "retrieval_text": cached,
            "field_kind": field_kind,
            "field_disposition": "cached_value",
            "raw_instruction": raw,
        }

    url_match = SAFE_HTTP_URL.search(raw)
    if url_match:
        url = url_match.group(0).replace(r"\:", ":")
        return {
            "display_text": url,
            "retrieval_text": url,
            "field_kind": field_kind,
            "field_disposition": "safe_hyperlink_target",
            "raw_instruction": raw,
        }

    if "calc" in field_kind or raw.lstrip().startswith("="):
        display = f"[계산 필드: {raw}]" if raw else "[계산 필드]"
        disposition = "calculation_not_indexed"
    elif "clickhere" in field_kind or raw.casefold().startswith("clickhere"):
        display = "[입력 안내 필드]"
        disposition = "clickhere_control_not_indexed"
    else:
        display = f"[{field_kind} 필드]"
        disposition = "unknown_field_not_indexed"

    return {
        "display_text": display,
        "retrieval_text": "",
        "field_kind": field_kind,
        "field_disposition": disposition,
        "raw_instruction": raw,
    }


def block_display_text(block: Any) -> str:
    """필드 제어문을 구분하면서 화면에 보여 줄 텍스트를 반환한다."""
    if kind_name(block) == "field":
        return field_parts(block)["display_text"]
    return normalize_text(block_plain_text(block))


def flatten_block_for_retrieval(block: Any) -> str:
    """블록 하나를 중복 없이 검색용 평문으로 바꾼다."""
    kind = kind_name(block)
    if kind == "picture":
        return ""
    if kind == "table":
        return table_retrieval_text(block)
    if kind == "field":
        return compact_text(field_parts(block)["retrieval_text"])
    if kind == "list_item":
        return compact_text(list_item_display_text(block))
    if kind in {"footnote", "endnote"}:
        return compact_text(note_display_text(block))

    own = compact_text(block_plain_text(block))
    if own:
        # paragraph.text는 자식 텍스트를 이미 합쳐 가진 경우가 많다.
        return own
    children = (flatten_block_for_retrieval(child) for child in child_blocks(block))
    return compact_text("\n".join(value for value in children if value))


def table_cell_text(cell: Any) -> str:
    """표 셀의 문단과 중첩 표를 원래 순서대로 한 번만 평탄화한다."""
    values = [
        flatten_block_for_retrieval(block)
        for block in (getattr(cell, "blocks", []) or [])
    ]
    return compact_text(" ".join(value for value in values if value))


def caption_text(block: Any) -> str:
    """표·이미지의 평문/구조화 캡션을 실제 자식 텍스트로 읽는다."""
    caption_value = getattr(block, "caption", None)
    if isinstance(caption_value, str) and compact_text(caption_value):
        return normalize_text(caption_value)

    structured_caption = (
        caption_value
        if caption_value is not None and not isinstance(caption_value, str)
        else getattr(block, "caption_block", None)
    )
    if structured_caption is None:
        return ""
    return normalize_text(flatten_block_for_retrieval(structured_caption))


def caption_metadata(block: Any) -> tuple[str | None, int | None, int | None]:
    """구조화 캡션의 방향과 원문 section/paragraph 위치를 반환한다."""
    caption_value = getattr(block, "caption", None)
    structured_caption = (
        caption_value
        if caption_value is not None and not isinstance(caption_value, str)
        else getattr(block, "caption_block", None)
    )
    if structured_caption is None:
        return None, None, None
    section_idx, para_idx = provenance(structured_caption)
    direction = str(getattr(structured_caption, "direction", "") or "") or None
    return direction, section_idx, para_idx


def table_retrieval_text(block: Any) -> str:
    """표시용 Markdown 문법과 무관한 행 단위 검색 텍스트를 만든다."""
    rows = int(getattr(block, "rows", 0) or 0)
    cells = sorted(
        getattr(block, "cells", []) or [],
        key=lambda cell: (
            int(getattr(cell, "row", 0) or 0),
            int(getattr(cell, "col", 0) or 0),
        ),
    )
    values_by_row: dict[int, list[str]] = defaultdict(list)
    for cell in cells:
        row = int(getattr(cell, "row", 0) or 0)
        values_by_row[row].append(table_cell_text(cell))

    lines: list[str] = []
    caption = compact_text(caption_text(block))
    if caption:
        lines.append(caption)
    for row_index in range(rows):
        row_text = " | ".join(values_by_row.get(row_index, []))
        if row_text.strip(" |"):
            lines.append(row_text)
    return normalize_text("\n".join(lines))


def table_has_picture(block: Any) -> bool:
    """텍스트가 없어도 이미지가 든 배치용 표인지 확인한다."""
    return any(
        kind_name(child) == "picture"
        for child, _ in walk_blocks_with_depth(child_blocks(block))
    )


def classify_table(block: Any) -> tuple[str, str, str]:
    """HWP 표를 content/layout/empty와 검색 정책으로 분류한다."""
    text = table_retrieval_text(block)
    compact = compact_text(text)
    has_picture = table_has_picture(block)
    if not compact and not has_picture:
        return "empty", "exclude", "empty_table"
    if not compact and has_picture:
        return "layout", "exclude", "picture_only_layout_table"

    rows = int(getattr(block, "rows", 0) or 0)
    cols = int(getattr(block, "cols", 0) or 0)
    cells = list(getattr(block, "cells", []) or [])
    nonempty_cells = sum(bool(table_cell_text(cell)) for cell in cells)
    has_header = any(
        str(getattr(cell, "role", "") or "").casefold()
        in {"header", "column_header", "row_header"}
        for cell in cells
    )
    has_caption = bool(caption_text(block))
    grid_like = rows >= 2 and cols >= 2 and nonempty_cells >= 2
    text_rich = len(compact) >= 80 and (rows >= 2 or cols >= 2)
    if (
        has_header
        or has_caption
        or TABLE_CONTENT_SIGNAL.search(compact)
        or grid_like
        or text_rich
    ):
        return "content", "index", "semantic_table_content"
    return "layout", "flatten", "layout_table_text_flattened"


def escape_markdown_cell(value: str) -> str:
    """표 셀을 HTML 없이 한 줄로 만들고 Markdown 구분자를 이스케이프한다."""
    single_line = re.sub(r"\n+", " / ", normalize_text(value))
    return (
        single_line.replace("\\", "\\\\")
        .replace("|", r"\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def picture_alt(block: Any, picture_id: str) -> str:
    """설명이나 캡션을 이미지 대체 텍스트로 사용한다."""
    description = compact_text(str(getattr(block, "description", "") or ""))
    caption = compact_text(caption_text(block))
    return description or caption or f"이미지 {picture_id}"


def render_picture_placeholder(block: Any, picture_id: str) -> str:
    """이미지 바이트 대신 안정적인 ``image://`` 참조만 만든다."""
    uri = f"image://{picture_id}"
    alt = picture_alt(block, picture_id)
    escaped_alt = (
        alt.replace("\\", "\\\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"![{escaped_alt}]({uri})"


def render_markdown_cell_block(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> str:
    """셀 자식을 HTML 없이 Markdown 한 조각으로 바꾼다.

    중첩 표는 셀 안에 다시 그리지 않고 ID만 남긴다. 실제 표는 부모 표 아래에
    별도 Markdown 표로 펼쳐 이미지와 표 내용이 중복되는 일을 막는다.
    """
    kind = kind_name(block)
    if kind == "table":
        return f"[중첩 표: {table_ids[id(block)]}]"
    if kind == "picture":
        return render_picture_placeholder(block, picture_ids[id(block)])
    if kind == "list_item":
        own = list_item_display_text(block)
    elif kind in {"footnote", "endnote"}:
        own = note_display_text(block)
    else:
        own = block_display_text(block)

    parts = [own] if own else []
    for child in child_blocks(block):
        child_kind = kind_name(child)
        # 일반 문단의 text에는 보통 자식 텍스트가 이미 들어 있다. 표와 그림은
        # 별도 구조이므로 항상 추가하고, own이 없을 때만 일반 자식을 펼친다.
        if child_kind in {"table", "picture"} or not own:
            rendered = render_markdown_cell_block(child, table_ids, picture_ids)
            if rendered:
                parts.append(rendered)
    return normalize_text("\n".join(parts))


def direct_nested_tables(block: Any) -> list[Any]:
    """현재 표에 직접 속한 중첩 표만 문서 순서대로 반환한다."""
    nested: list[Any] = []
    seen: set[int] = set()

    def visit(children: Iterable[Any]) -> None:
        for child in children:
            if kind_name(child) == "table":
                if id(child) not in seen:
                    nested.append(child)
                    seen.add(id(child))
                # 더 안쪽 표는 이 표를 렌더링할 때 처리한다.
                continue
            visit(child_blocks(child))

    visit(child_blocks(block))
    return nested


def render_table_gfm(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> str:
    """단순·병합·중첩 표를 모두 GFM Markdown으로 만든다."""
    cells = list(getattr(block, "cells", []) or [])
    declared_rows = max(int(getattr(block, "rows", 0) or 0), 0)
    declared_cols = max(int(getattr(block, "cols", 0) or 0), 0)
    rows = max(
        declared_rows,
        max(
            (
                int(getattr(cell, "row", 0) or 0)
                + max(int(getattr(cell, "row_span", 1) or 1), 1)
                for cell in cells
            ),
            default=0,
        ),
        1,
    )
    cols = max(
        declared_cols,
        max(
            (
                int(getattr(cell, "col", 0) or 0)
                + max(int(getattr(cell, "col_span", 1) or 1), 1)
                for cell in cells
            ),
            default=0,
        ),
        1,
    )
    grid = [["" for _ in range(cols)] for _ in range(rows)]

    ordered_cells = sorted(
        cells,
        key=lambda cell: (
            int(getattr(cell, "row", 0) or 0),
            int(getattr(cell, "col", 0) or 0),
        ),
    )
    for cell in ordered_cells:
        row = int(getattr(cell, "row", 0) or 0)
        col = int(getattr(cell, "col", 0) or 0)
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        row_span = max(int(getattr(cell, "row_span", 1) or 1), 1)
        col_span = max(int(getattr(cell, "col_span", 1) or 1), 1)
        values = [
            render_markdown_cell_block(child, table_ids, picture_ids)
            for child in (getattr(cell, "blocks", []) or [])
        ]
        content = "\n".join(value for value in values if value)
        if row_span > 1 or col_span > 1:
            content = f"[병합 {row_span}행×{col_span}열] {content}".rstrip()
        escaped_content = escape_markdown_cell(content)
        merge_reference = escape_markdown_cell(
            f"[병합 {row_span}행×{col_span}열 계속: {row + 1}행 {col + 1}열 참조]"
        )

        # GFM에는 rowspan/colspan이 없다. 원점에는 전체 내용을, 나머지 칸에는
        # 원점 참조를 넣는다. 이미지 URI까지 반복해 출력하지 않기 위함이다.
        for target_row in range(row, min(row + row_span, rows)):
            for target_col in range(col, min(col + col_span, cols)):
                if target_row == row and target_col == col:
                    grid[target_row][target_col] = escaped_content
                elif not grid[target_row][target_col]:
                    grid[target_row][target_col] = merge_reference

    lines = [
        "| " + " | ".join(grid[0]) + " |",
        "| " + " | ".join("---" for _ in grid[0]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in grid[1:])
    sections: list[str] = ["\n".join(lines)]
    caption = caption_text(block)
    if caption:
        safe_caption = caption.replace("<", "&lt;").replace(">", "&gt;")
        sections[0] = f"{safe_caption}\n\n{sections[0]}"

    for nested in direct_nested_tables(block):
        nested_id = table_ids[id(nested)]
        nested_markdown = render_table_gfm(nested, table_ids, picture_ids)
        sections.append(f"**중첩 표 `{nested_id}`**\n\n{nested_markdown}")
    return "\n\n".join(sections)


def render_table(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> tuple[str, str]:
    """표시 내용을 예외 없이 GFM Markdown으로 반환한다."""
    return render_table_gfm(block, table_ids, picture_ids), "gfm"


def render_generic_block(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> str:
    """표·이미지를 포함한 최상위 IR 블록을 화면 표시 형식으로 만든다."""
    kind = kind_name(block)
    if kind == "table":
        return render_table(block, table_ids, picture_ids)[0]
    if kind == "picture":
        return render_picture_placeholder(block, picture_ids[id(block)])
    if kind == "list_item":
        return list_item_display_text(block)
    if kind in {"footnote", "endnote"}:
        return note_display_text(block)

    own = block_display_text(block)
    parts = [own] if own else []
    for child in child_blocks(block):
        child_kind = kind_name(child)
        # own에 일반 자식 문단이 이미 합쳐진 경우가 많아 구조 객체만 덧붙인다.
        if child_kind in {"table", "picture"} or not own:
            rendered = render_generic_block(child, table_ids, picture_ids)
            if rendered:
                parts.append(rendered)
    return normalize_text("\n\n".join(parts))


def heading_from_text(value: str) -> str | None:
    """로마숫자나 '제 N 장'으로 시작하는 큰 제목을 섹션명으로 찾는다."""
    for line in normalize_text(value).splitlines()[:3]:
        candidate = " ".join(line.split())
        if len(candidate) <= 200 and TOP_HEADING.match(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# HWP/HWPX 구조 처리
# ---------------------------------------------------------------------------


def furniture_roots_with_type(ir: Any) -> list[tuple[Any, str]]:
    """머리말·꼬리말·각주·미주를 종류와 함께 고정된 순서로 반환한다."""
    furniture = getattr(ir, "furniture", None)
    if furniture is None:
        return []

    result: list[tuple[Any, str]] = []
    for attribute, label in (
        ("page_headers", "header"),
        ("page_footers", "footer"),
        ("footnotes", "footnote"),
        ("endnotes", "endnote"),
    ):
        roots = getattr(furniture, attribute, []) or []
        result.extend((block, label) for block in roots)
    return result


def _optional_int(value: Any) -> int | None:
    """파서 메타데이터의 선택적 숫자를 JSON에 안전한 int로 바꾼다."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_image_uri(value: Any) -> str:
    """로컬 경로나 payload를 노출하지 않고 rhwp의 ``bin://``만 허용한다."""
    uri = str(value or "")
    return uri if uri.casefold().startswith("bin://") else ""


def build_hwp_structure_maps(
    ir: Any,
    source_id: str,
) -> tuple[
    dict[int, str],
    dict[int, str],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """rhwp IR을 순회해 표·이미지 ID와 메타데이터를 만든다.

    과거 EDA CSV에 기대지 않고 현재 IR에서 직접 계산한다. Python 객체의
    ``id``는 실행 중 연결표로만 쓰며 반환 레코드에는 저장하지 않는다.
    """
    body_roots = list(getattr(ir, "body", []) or [])
    furniture_roots = [block for block, _ in furniture_roots_with_type(ir)]
    all_root_ids = {id(block) for block in [*body_roots, *furniture_roots]}

    encountered_tables: list[tuple[Any, str, int]] = []
    encountered_images: list[tuple[Any, str, int]] = []
    for scope, roots in (("body", body_roots), ("furniture", furniture_roots)):
        for block, depth in walk_blocks_with_depth(roots):
            kind = kind_name(block)
            if kind == "table":
                encountered_tables.append((block, scope, depth))
            elif kind == "picture":
                encountered_images.append((block, scope, depth))

    table_ids: dict[int, str] = {}
    picture_ids: dict[int, str] = {}
    table_by_object: dict[int, dict[str, Any]] = {}
    image_by_object: dict[int, dict[str, Any]] = {}

    for index, (block, scope, depth) in enumerate(encountered_tables):
        table_id = f"{source_id}:{scope}:T{index + 1:06d}"
        table_ids[id(block)] = table_id
        section_idx, para_idx = provenance(block)
        cells = list(getattr(block, "cells", []) or [])
        is_top_level = id(block) in all_root_ids
        table_class, base_policy, base_reason = classify_table(block)
        caption_direction, caption_section_idx, caption_para_idx = caption_metadata(
            block
        )

        if scope != "body":
            index_policy, index_reason = "exclude", "document_furniture"
        elif not is_top_level:
            index_policy, index_reason = "exclude", "embedded_in_parent_block"
        else:
            index_policy, index_reason = base_policy, base_reason

        table_by_object[id(block)] = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "document_id": source_id,
            "table_id": table_id,
            "source_table_index": index,
            "scope": scope,
            "is_top_level": is_top_level,
            "nested_depth": depth,
            "parent_block_id": None,
            "parent_table_id": None,
            "section_idx": section_idx,
            "para_idx": para_idx,
            # rhwp IR에는 신뢰할 수 있는 페이지 매핑이 없어 추측하지 않는다.
            "page": None,
            "bbox": None,
            "rows": int(getattr(block, "rows", 0) or 0),
            "cols": int(getattr(block, "cols", 0) or 0),
            "cell_count": len(cells),
            "merged_cell_count": sum(
                int(getattr(cell, "row_span", 1) or 1) > 1
                or int(getattr(cell, "col_span", 1) or 1) > 1
                for cell in cells
            ),
            "empty_cell_count": sum(not table_cell_text(cell) for cell in cells),
            "header_cell_count": sum(
                str(getattr(cell, "role", "") or "").casefold()
                in {"header", "column_header", "row_header"}
                for cell in cells
            ),
            "has_caption": bool(caption_text(block)),
            "caption_direction": caption_direction,
            "caption_section_idx": caption_section_idx,
            "caption_para_idx": caption_para_idx,
            "has_nested_table": has_nested_table(block),
            "render_mode": table_markdown_mode(block),
            "table_class": table_class,
            "retrieval_chars": len(table_retrieval_text(block)),
            "index_policy": index_policy,
            "index_reason": index_reason,
        }

    for index, (block, scope, depth) in enumerate(encountered_images):
        picture_id = f"{source_id}:{scope}:I{index + 1:06d}"
        picture_ids[id(block)] = picture_id
        section_idx, para_idx = provenance(block)
        image_reference = getattr(block, "image", None)
        description = compact_text(str(getattr(block, "description", "") or ""))
        raw_image_uri = str(getattr(image_reference, "uri", "") or "")
        safe_image_uri = _safe_image_uri(raw_image_uri)
        caption_direction, caption_section_idx, caption_para_idx = caption_metadata(
            block
        )

        image_by_object[id(block)] = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "document_id": source_id,
            "picture_id": picture_id,
            "source_picture_index": index,
            "scope": scope,
            "is_top_level": id(block) in all_root_ids,
            "nested_depth": depth,
            "parent_block_id": None,
            "parent_table_id": None,
            "section_idx": section_idx,
            "para_idx": para_idx,
            "page": None,
            "bbox": None,
            "original_ir_uri": safe_image_uri,
            "original_ir_uri_redacted": bool(raw_image_uri and not safe_image_uri),
            "mime_type": str(getattr(image_reference, "mime_type", "") or ""),
            "width": _optional_int(getattr(image_reference, "width", None)),
            "height": _optional_int(getattr(image_reference, "height", None)),
            "dpi": _optional_int(getattr(image_reference, "dpi", None)),
            "alt_text": picture_alt(block, picture_id),
            "has_caption": bool(caption_text(block)),
            "caption_direction": caption_direction,
            "caption_section_idx": caption_section_idx,
            "caption_para_idx": caption_para_idx,
            "has_description": bool(description),
            # 이미지 바이트를 읽지 않으므로 크기와 해시는 모른다고 명시한다.
            "binary_status": "not_read_by_preprocessor",
            "binary_size_bytes": None,
            "binary_sha256": "",
            "index_enabled": False,
            "index_policy": "exclude",
            "index_reason": "image_metadata_only",
            "payload_stored": False,
        }

    return table_ids, picture_ids, table_by_object, image_by_object


def assign_structure_relationships(
    block: Any,
    owner_block_id: str,
    table_by_object: dict[int, dict[str, Any]],
    image_by_object: dict[int, dict[str, Any]],
    parent_table_id: str | None = None,
) -> None:
    """중첩 표·이미지에 부모 블록과 부모 표 ID를 연결한다."""
    kind = kind_name(block)
    next_parent_table_id = parent_table_id
    if kind == "table":
        table = table_by_object[id(block)]
        table["parent_block_id"] = owner_block_id
        table["parent_table_id"] = parent_table_id
        next_parent_table_id = table["table_id"]
    elif kind == "picture":
        picture = image_by_object[id(block)]
        picture["parent_block_id"] = owner_block_id
        picture["parent_table_id"] = parent_table_id

    for child in child_blocks(block):
        assign_structure_relationships(
            child,
            owner_block_id,
            table_by_object,
            image_by_object,
            next_parent_table_id,
        )


def make_hwp_blocks(
    ir: Any,
    source_id: str,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
    table_by_object: dict[int, dict[str, Any]],
    image_by_object: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """HWP 본문과 문서 부속물을 원래 순서의 최상위 블록으로 만든다.

    표 셀 안의 표·이미지는 부모 블록 안에 한 번만 렌더링하고 독립 블록으로
    다시 만들지 않는다. 세부 위치는 table/image 레코드의 부모 ID로 추적한다.
    """
    roots: list[tuple[Any, str, str | None]] = [
        (block, "body", None) for block in (getattr(ir, "body", []) or [])
    ]
    roots.extend(
        (block, "furniture", furniture_type)
        for block, furniture_type in furniture_roots_with_type(ir)
    )

    blocks: list[dict[str, Any]] = []
    section_path = "본문"
    for order, (root, scope, furniture_type) in enumerate(roots, start=1):
        block_id = f"{source_id}:B{order:06d}"
        kind = kind_name(root)
        section_idx, para_idx = provenance(root)
        display_content = render_generic_block(root, table_ids, picture_ids)
        retrieval_text = flatten_block_for_retrieval(root)

        if scope == "body":
            detected_heading = heading_from_text(retrieval_text)
            if detected_heading:
                section_path = detected_heading

        table_id = table_ids.get(id(root))
        picture_id = picture_ids.get(id(root))
        render_mode: str | None = None
        quality_flags: list[str] = []
        field_metadata = field_parts(root) if kind == "field" else None
        list_marker = (
            str(getattr(root, "marker", "") or "") or None
            if kind == "list_item"
            else None
        )
        list_enumerated = (
            bool(getattr(root, "enumerated", False)) if kind == "list_item" else None
        )
        list_level = (
            _optional_int(getattr(root, "level", 0)) if kind == "list_item" else None
        )
        marker_prov = (
            getattr(root, "marker_prov", None)
            if kind in {"footnote", "endnote"}
            else None
        )
        toc_entries = toc_entries_metadata(root) if kind == "toc" else []

        if kind == "table":
            table = table_by_object[id(root)]
            index_policy = table["index_policy"]
            index_reason = table["index_reason"]
            render_mode = table["render_mode"]
            if table["table_class"] != "content":
                quality_flags.append(f"{table['table_class']}_table")
        elif kind == "picture":
            retrieval_text = ""
            index_policy = "exclude"
            index_reason = "image_metadata_only"
        elif kind == "field":
            assert field_metadata is not None
            if scope == "furniture":
                index_policy = "exclude"
                index_reason = "document_furniture"
            elif retrieval_text:
                index_policy = "index"
                index_reason = f"field_{field_metadata['field_disposition']}"
            else:
                index_policy = "exclude"
                index_reason = f"field_{field_metadata['field_disposition']}"
                quality_flags.append(field_metadata["field_disposition"])
        elif scope == "furniture":
            index_policy = "exclude"
            index_reason = "document_furniture"
        elif retrieval_text:
            index_policy = "index"
            index_reason = "body_text"
        else:
            index_policy = "exclude"
            index_reason = "empty_nontext_block"

        blocks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "document_id": source_id,
                "block_id": block_id,
                "block_order": order,
                "parent_block_id": None,
                "scope": scope,
                "furniture_type": furniture_type,
                "block_type": kind,
                "display_content": display_content,
                "retrieval_text": normalize_text(retrieval_text),
                "index_policy": index_policy,
                "index_reason": index_reason,
                "section_path": section_path if scope == "body" else furniture_type,
                "section_idx": section_idx,
                "para_idx": para_idx,
                "page": None,
                "bbox": None,
                "table_id": table_id,
                "picture_id": picture_id,
                "nested_depth": 0,
                "render_mode": render_mode,
                "field_kind": field_metadata["field_kind"] if field_metadata else None,
                "field_disposition": (
                    field_metadata["field_disposition"] if field_metadata else None
                ),
                "field_raw_instruction": (
                    field_metadata["raw_instruction"] if field_metadata else None
                ),
                "list_marker": list_marker,
                "list_enumerated": list_enumerated,
                "list_level": list_level,
                "note_number": (
                    _optional_int(getattr(root, "number", None))
                    if kind in {"footnote", "endnote"}
                    else None
                ),
                "note_marker_section_idx": getattr(marker_prov, "section_idx", None),
                "note_marker_para_idx": getattr(marker_prov, "para_idx", None),
                "formula_script": (
                    str(getattr(root, "script", "") or "")
                    if kind == "formula"
                    else None
                ),
                "formula_script_kind": (
                    str(getattr(root, "script_kind", "") or "") or None
                    if kind == "formula"
                    else None
                ),
                "formula_inline": (
                    bool(getattr(root, "inline", False)) if kind == "formula" else None
                ),
                "toc_entry_count": len(toc_entries) if kind == "toc" else None,
                "toc_entries": toc_entries,
                "caption_direction": (
                    str(getattr(root, "direction", "") or "") or None
                    if kind == "caption"
                    else None
                ),
                "quality_flags": quality_flags,
            }
        )
        assign_structure_relationships(
            root,
            block_id,
            table_by_object,
            image_by_object,
        )

    return blocks


def process_hwp_document(
    source: SourceDocument,
    analysis_path: Path,
    rhwp_module: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """HWP/HWPX 하나를 ``to_ir()`` 한 번으로 구조화한다."""
    parsed_document = rhwp_module.parse(str(analysis_path))
    ir = parsed_document.to_ir()
    table_ids, picture_ids, table_by_object, image_by_object = build_hwp_structure_maps(
        ir, source.source_id
    )
    blocks = make_hwp_blocks(
        ir,
        source.source_id,
        table_ids,
        picture_ids,
        table_by_object,
        image_by_object,
    )
    tables = sorted(
        table_by_object.values(), key=lambda record: record["source_table_index"]
    )
    images = sorted(
        image_by_object.values(), key=lambda record: record["source_picture_index"]
    )
    stats = {
        "parser": "rhwp.to_ir",
        "page_count": getattr(parsed_document, "page_count", None),
        "section_count": getattr(parsed_document, "section_count", None),
        "paragraph_count": getattr(parsed_document, "paragraph_count", None),
        "body_table_count": sum(row["scope"] == "body" for row in tables),
        "furniture_table_count": sum(row["scope"] == "furniture" for row in tables),
        "body_picture_count": sum(row["scope"] == "body" for row in images),
        "furniture_picture_count": sum(row["scope"] == "furniture" for row in images),
        "picture_placeholder_count": sum(
            block["display_content"].count("image://") for block in blocks
        ),
        "quality_flags": [],
    }
    return blocks, tables, images, stats


# ---------------------------------------------------------------------------
# PDF 페이지·표·이미지 처리
# ---------------------------------------------------------------------------


def point_in_bbox(x: float, y: float, bbox: Sequence[float]) -> bool:
    """단어 중심점이 PDF 표 사각형 안에 있는지 확인한다."""
    x0, top, x1, bottom = map(float, bbox)
    return x0 <= x <= x1 and top <= y <= bottom


def deduplicate_pdf_tables(tables: Sequence[Any]) -> list[Any]:
    """같은 위치에서 중복 검출된 PDF 표를 한 번만 남긴다."""
    result: list[Any] = []
    seen: set[tuple[float, float, float, float]] = set()
    ordered = sorted(
        tables,
        key=lambda item: (float(item.bbox[1]), float(item.bbox[0])),
    )
    for table in ordered:
        key = tuple(round(float(value), 2) for value in table.bbox)
        if key not in seen:
            seen.add(key)
            result.append(table)
    return result


def pdf_matrix_text(matrix: Sequence[Sequence[Any]]) -> str:
    """PDF 표 행렬을 검색용 행 단위 평문으로 바꾼다."""
    lines: list[str] = []
    for row in matrix:
        cells = [compact_text("" if value is None else str(value)) for value in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return normalize_text("\n".join(lines))


def comparable_text_char_count(value: str | None) -> int:
    """공백·문장부호를 제외한 글자 수로 PDF 추출량을 비교한다."""
    return sum(character.isalnum() for character in normalize_text(value))


def pdf_table_text_coverage(
    matrix: Sequence[Sequence[Any]],
    words: Sequence[dict[str, Any]],
    bbox: Sequence[float],
) -> float | None:
    """표 영역 원문 대비 셀 행렬이 보존한 텍스트 비율을 추정한다.

    pdfplumber가 병합 셀의 테두리는 찾았지만 셀 본문을 놓치면 표 영역 전체의
    단어가 본문에서도 제거될 수 있다. 작은 표는 글자 수 차이에 민감하므로
    충분한 원문이 있는 표만 검사한다.
    """
    inside_text: list[str] = []
    for word in words:
        center_x = (float(word.get("x0", 0) or 0) + float(word.get("x1", 0) or 0)) / 2
        center_y = (
            float(word.get("top", 0) or 0) + float(word.get("bottom", 0) or 0)
        ) / 2
        if point_in_bbox(center_x, center_y, bbox):
            inside_text.append(str(word.get("text") or ""))

    source_chars = comparable_text_char_count(" ".join(inside_text))
    if source_chars < PDF_TABLE_TEXT_FALLBACK_MIN_CHARS:
        return None

    extracted_chars = comparable_text_char_count(pdf_matrix_text(matrix))
    return min(extracted_chars / source_chars, 1.0)


def render_pdf_table(matrix: Sequence[Sequence[Any]]) -> str:
    """PDF 표 행렬을 GFM Markdown으로 만든다."""
    rows = [
        [escape_markdown_cell("" if value is None else str(value)) for value in row]
        for row in matrix
    ]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        # 검출됐지만 내용 추출에 실패한 표도 HTML이나 빈 문자열로 바꾸지 않는다.
        return "|  |\n| --- |"
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def classify_pdf_table(
    matrix: Sequence[Sequence[Any]],
) -> tuple[str, str, str]:
    """PDF 표도 HWP와 같은 content/layout/empty 규칙으로 분류한다."""
    text = pdf_matrix_text(matrix)
    compact = compact_text(text)
    if not compact:
        return "empty", "exclude", "empty_table"

    rows = len(matrix)
    cols = max((len(row) for row in matrix), default=0)
    nonempty = sum(
        bool(compact_text("" if value is None else str(value)))
        for row in matrix
        for value in row
    )
    if (
        TABLE_CONTENT_SIGNAL.search(compact)
        or (rows >= 2 and cols >= 2 and nonempty >= 2)
        or len(compact) >= 80
    ):
        return "content", "index", "semantic_table_content"
    return "layout", "flatten", "layout_table_text_flattened"


def group_pdf_words_into_lines(
    words: Sequence[dict[str, Any]], tolerance: float = 3.0
) -> list[tuple[float, str]]:
    """y좌표가 가까운 PDF 단어를 한 줄로 묶어 읽기 순서를 복원한다."""
    lines: list[dict[str, Any]] = []
    ordered_words = sorted(
        words,
        key=lambda item: (
            float(item.get("top", 0) or 0),
            float(item.get("x0", 0) or 0),
        ),
    )
    for word in ordered_words:
        top = float(word.get("top", 0) or 0)
        if not lines or abs(top - float(lines[-1]["top"])) > tolerance:
            lines.append({"top": top, "words": [word]})
        else:
            lines[-1]["words"].append(word)

    result: list[tuple[float, str]] = []
    for line in lines:
        ordered = sorted(line["words"], key=lambda item: float(item.get("x0", 0) or 0))
        text = compact_text(" ".join(str(item.get("text") or "") for item in ordered))
        if text:
            result.append((float(line["top"]), text))
    return result


def pdf_image_bbox(
    image: dict[str, Any], page_height: float
) -> tuple[float, float, float, float]:
    """pdfplumber 이미지 좌표를 top/bottom 좌표계로 통일한다."""
    x0 = float(image.get("x0", 0) or 0)
    x1 = float(image.get("x1", x0) or x0)
    if image.get("top") is not None and image.get("bottom") is not None:
        top = float(image["top"])
        bottom = float(image["bottom"])
    else:
        y0 = float(image.get("y0", 0) or 0)
        y1 = float(image.get("y1", y0) or y0)
        top, bottom = page_height - y1, page_height - y0
    return x0, top, x1, bottom


def process_pdf_document(
    source: SourceDocument,
    pdfplumber_module: Any,
    *,
    analysis_path: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """PDF를 본문·표·이미지의 세로 위치 순서대로 블록화한다."""
    source_id = source.source_id
    source_path = analysis_path or source.source_path
    blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    section_path = "본문"
    table_error_count = 0
    word_error_count = 0
    fallback_text_error_count = 0
    table_text_fallback_count = 0
    table_text_fallback_pages: set[int] = set()
    document_quality_flags: set[str] = set()

    with pdfplumber_module.open(source_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                detected_tables = deduplicate_pdf_tables(page.find_tables() or [])
            except Exception:
                detected_tables = []
                table_error_count += 1
                document_quality_flags.add("pdf_table_detection_failed")

            extracted_tables: list[tuple[Any, Sequence[Sequence[Any]]]] = []
            for detected_table in detected_tables:
                try:
                    extracted_tables.append(
                        (detected_table, detected_table.extract() or [])
                    )
                except Exception:
                    # 추출 실패 표의 영역에서 본문 단어를 제거하면 내용이 사라진다.
                    table_error_count += 1
                    document_quality_flags.add("pdf_table_extraction_failed")

            try:
                words = list(
                    page.extract_words(
                        use_text_flow=True,
                        keep_blank_chars=False,
                    )
                    or []
                )
            except Exception:
                words = []
                word_error_count += 1
                document_quality_flags.add("pdf_word_extraction_failed")

            prepared_tables: list[
                tuple[Any, Sequence[Sequence[Any]], float | None, bool]
            ] = []
            for detected_table, matrix in extracted_tables:
                bbox = tuple(map(float, detected_table.bbox))
                coverage = pdf_table_text_coverage(matrix, words, bbox)
                use_text_fallback = (
                    coverage is not None
                    and coverage < PDF_TABLE_TEXT_COVERAGE_THRESHOLD
                )
                if use_text_fallback:
                    table_text_fallback_count += 1
                    table_text_fallback_pages.add(page_number)
                    document_quality_flags.add("pdf_table_text_fallback")
                prepared_tables.append(
                    (detected_table, matrix, coverage, use_text_fallback)
                )

            if words or prepared_tables:
                # 좌표 기반 읽기 순서는 다단 문서에서 완벽하지 않을 수 있음을 표시한다.
                document_quality_flags.add("pdf_coordinate_order_heuristic")

            # 완전한 표 안의 단어는 표 행렬에 이미 있으므로 본문에서 제외한다.
            # 불완전한 표의 단어는 표별 fallback 블록으로 묶어 긴 셀 내용과
            # 다단 페이지의 열 경계를 함께 보존한다.
            outside_words: list[dict[str, Any]] = []
            fallback_words_by_table: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for word in words:
                center_x = (
                    float(word.get("x0", 0) or 0) + float(word.get("x1", 0) or 0)
                ) / 2
                center_y = (
                    float(word.get("top", 0) or 0) + float(word.get("bottom", 0) or 0)
                ) / 2
                fallback_matches: list[tuple[float, int]] = []
                inside_trusted_table = False
                for table_index, (
                    detected_table,
                    matrix,
                    _,
                    use_text_fallback,
                ) in enumerate(prepared_tables):
                    bbox = tuple(map(float, detected_table.bbox))
                    if not point_in_bbox(center_x, center_y, bbox):
                        continue
                    if use_text_fallback:
                        area = max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)
                        fallback_matches.append((area, table_index))
                    elif pdf_matrix_text(matrix):
                        inside_trusted_table = True

                if fallback_matches:
                    # 겹친 표에서는 더 구체적인 작은 영역에 단어를 한 번만 배정한다.
                    _, selected_table_index = min(fallback_matches)
                    fallback_words_by_table[selected_table_index].append(word)
                elif not inside_trusted_table:
                    outside_words.append(word)

            # (세로 위치, 같은 위치 우선순위, 종류, 값, 품질 플래그)
            events: list[tuple[float, int, str, Any, list[str]]] = []
            for top, line_text in group_pdf_words_into_lines(outside_words):
                events.append((top, 0, "text", line_text, []))
            for table_index, (
                detected_table,
                matrix,
                _,
                use_text_fallback,
            ) in enumerate(prepared_tables):
                table_flags = ["pdf_table_text_fallback"] if use_text_fallback else []
                table_top = float(detected_table.bbox[1])
                events.append(
                    (
                        table_top,
                        1,
                        "table",
                        (detected_table, matrix),
                        table_flags,
                    )
                )
                if use_text_fallback:
                    fallback_lines = group_pdf_words_into_lines(
                        fallback_words_by_table[table_index]
                    )
                    fallback_text = normalize_text(
                        "\n".join(text for _, text in fallback_lines)
                    )
                    if fallback_text:
                        events.append(
                            (
                                table_top,
                                2,
                                "text",
                                fallback_text,
                                table_flags,
                            )
                        )
            for image in getattr(page, "images", []) or []:
                bbox = pdf_image_bbox(image, float(getattr(page, "height", 0) or 0))
                events.append((bbox[1], 3, "picture", (image, bbox), []))

            # 단어 추출이 실패하면 표가 있더라도 본문 전체를 보존한다. 이 경우
            # 표 텍스트가 일부 겹칠 수 있으므로 품질 플래그로 반드시 드러낸다.
            needs_fallback = not words
            if needs_fallback:
                try:
                    fallback_text = normalize_text(page.extract_text() or "")
                except Exception:
                    fallback_text = ""
                    fallback_text_error_count += 1
                    document_quality_flags.add("pdf_fallback_text_failed")
                if fallback_text:
                    flags = []
                    if prepared_tables:
                        flags.append("pdf_fallback_text_may_duplicate_table")
                        document_quality_flags.add(flags[0])
                    events.append((0.0, 0, "text", fallback_text, flags))

            for _, _, event_type, payload, event_flags in sorted(
                events, key=lambda event: (event[0], event[1])
            ):
                block_order = len(blocks) + 1
                block_id = f"{source_id}:B{block_order:06d}"
                table_id: str | None = None
                picture_id: str | None = None
                render_mode: str | None = None
                bbox_value: tuple[float, float, float, float] | None = None
                quality_flags = list(event_flags)

                if event_type == "text":
                    display_content = normalize_text(str(payload))
                    retrieval_text = display_content
                    index_policy = "index" if retrieval_text else "exclude"
                    if "pdf_table_text_fallback" in quality_flags:
                        index_reason = "incomplete_pdf_table_bbox_text"
                    else:
                        index_reason = (
                            "pdf_page_text" if retrieval_text else "empty_pdf_text"
                        )
                    detected_heading = heading_from_text(retrieval_text)
                    if detected_heading:
                        section_path = detected_heading
                    block_type = "text"
                elif event_type == "table":
                    pdf_table, matrix = payload
                    table_number = len(tables) + 1
                    table_id = f"{source_id}:pdf:T{table_number:06d}"
                    display_content = render_pdf_table(matrix)
                    retrieval_text = pdf_matrix_text(matrix)
                    table_class, index_policy, index_reason = classify_pdf_table(matrix)
                    if "pdf_table_text_fallback" in quality_flags:
                        index_policy = "exclude"
                        index_reason = "incomplete_pdf_table_replaced_by_bbox_text"
                    render_mode = "gfm"
                    block_type = "table"
                    if table_class != "content":
                        quality_flags.append(f"{table_class}_table")
                    bbox_value = tuple(
                        round(float(value), 3) for value in pdf_table.bbox
                    )
                    tables.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "source_id": source_id,
                            "document_id": source_id,
                            "table_id": table_id,
                            "source_table_index": table_number - 1,
                            "scope": "body",
                            "is_top_level": True,
                            "nested_depth": 0,
                            "parent_block_id": block_id,
                            "parent_table_id": None,
                            "section_idx": None,
                            "para_idx": None,
                            "page": page_number,
                            "bbox": bbox_value,
                            "rows": len(matrix),
                            "cols": max((len(row) for row in matrix), default=0),
                            "cell_count": sum(len(row) for row in matrix),
                            "merged_cell_count": None,
                            "empty_cell_count": sum(
                                not compact_text("" if value is None else str(value))
                                for row in matrix
                                for value in row
                            ),
                            "header_cell_count": None,
                            "has_caption": False,
                            "caption_direction": None,
                            "caption_section_idx": None,
                            "caption_para_idx": None,
                            "has_nested_table": False,
                            "render_mode": render_mode,
                            "table_class": table_class,
                            "retrieval_chars": len(retrieval_text),
                            "index_policy": index_policy,
                            "index_reason": index_reason,
                        }
                    )
                else:
                    image, image_bbox = payload
                    picture_number = len(images) + 1
                    picture_id = f"{source_id}:pdf:I{picture_number:06d}"
                    alt = f"PDF {page_number}쪽 이미지 {picture_number}"
                    display_content = f"![{alt}](image://{picture_id})"
                    retrieval_text = ""
                    index_policy = "exclude"
                    index_reason = "image_metadata_only"
                    block_type = "picture"
                    bbox_value = tuple(round(float(value), 3) for value in image_bbox)
                    images.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "source_id": source_id,
                            "document_id": source_id,
                            "picture_id": picture_id,
                            "source_picture_index": picture_number - 1,
                            "scope": "body",
                            "is_top_level": True,
                            "nested_depth": 0,
                            "parent_block_id": block_id,
                            "parent_table_id": None,
                            "section_idx": None,
                            "para_idx": None,
                            "page": page_number,
                            "bbox": bbox_value,
                            "original_ir_uri": "",
                            "original_ir_uri_redacted": False,
                            "mime_type": (
                                "image/mask" if image.get("imagemask") else "unknown"
                            ),
                            "width": round(float(image_bbox[2] - image_bbox[0]), 3),
                            "height": round(float(image_bbox[3] - image_bbox[1]), 3),
                            "dpi": None,
                            "alt_text": alt,
                            "has_caption": False,
                            "caption_direction": None,
                            "caption_section_idx": None,
                            "caption_para_idx": None,
                            "has_description": False,
                            "binary_status": "not_read_by_preprocessor",
                            "binary_size_bytes": None,
                            "binary_sha256": "",
                            "index_enabled": False,
                            "index_policy": "exclude",
                            "index_reason": "image_metadata_only",
                            "payload_stored": False,
                        }
                    )

                blocks.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_id": source_id,
                        "document_id": source_id,
                        "block_id": block_id,
                        "block_order": block_order,
                        "parent_block_id": None,
                        "scope": "body",
                        "furniture_type": None,
                        "block_type": block_type,
                        "display_content": display_content,
                        "retrieval_text": retrieval_text,
                        "index_policy": index_policy,
                        "index_reason": index_reason,
                        "section_path": section_path,
                        "section_idx": None,
                        "para_idx": None,
                        "page": page_number,
                        "bbox": bbox_value,
                        "table_id": table_id,
                        "picture_id": picture_id,
                        "nested_depth": 0,
                        "render_mode": render_mode,
                        "field_kind": None,
                        "field_disposition": None,
                        "field_raw_instruction": None,
                        "list_marker": None,
                        "list_enumerated": None,
                        "list_level": None,
                        "note_number": None,
                        "note_marker_section_idx": None,
                        "note_marker_para_idx": None,
                        "formula_script": None,
                        "formula_script_kind": None,
                        "formula_inline": None,
                        "toc_entry_count": None,
                        "toc_entries": [],
                        "caption_direction": None,
                        "quality_flags": quality_flags,
                    }
                )
        page_count = len(pdf.pages)

    stats = {
        "parser": "pdfplumber",
        "page_count": page_count,
        "section_count": None,
        "paragraph_count": None,
        "body_table_count": len(tables),
        "furniture_table_count": 0,
        "body_picture_count": len(images),
        "furniture_picture_count": 0,
        "picture_placeholder_count": len(images),
        "pdf_table_error_count": table_error_count,
        "pdf_word_extraction_error_count": word_error_count,
        "pdf_fallback_text_error_count": fallback_text_error_count,
        "pdf_table_text_fallback_count": table_text_fallback_count,
        "pdf_table_text_fallback_page_count": len(table_text_fallback_pages),
        "quality_flags": sorted(document_quality_flags),
    }
    return blocks, tables, images, stats


# ---------------------------------------------------------------------------
# 공개 API와 보안 검사
# ---------------------------------------------------------------------------


def has_forbidden_image_payload(value: Any) -> bool:
    """결과 어디에도 이미지 바이트·Base64·payload가 없는지 재귀 검사한다."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, str):
        return bool(DATA_URI.search(value))
    if isinstance(value, (list, tuple)):
        return any(has_forbidden_image_payload(item) for item in value)
    if isinstance(value, dict):
        forbidden_keys = {
            "data",
            "base64",
            "payload",
            "image_bytes",
            "binary_payload",
        }
        if forbidden_keys & {str(key).casefold() for key in value}:
            return True
        return any(has_forbidden_image_payload(item) for item in value.values())
    return False


def validate_preprocessing_contract(
    source: SourceDocument,
    blocks: Sequence[dict[str, Any]],
    tables: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]],
) -> None:
    """EDA 파일 없이 확인 가능한 문서 단위 구조 불변식을 검사한다."""
    block_ids = [str(block["block_id"]) for block in blocks]
    table_ids = [str(table["table_id"]) for table in tables]
    picture_ids = [str(image["picture_id"]) for image in images]

    if [block["block_order"] for block in blocks] != list(range(1, len(blocks) + 1)):
        raise ValueError("block_order가 1부터 연속된 문서 순서가 아닙니다")
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_id가 문서 안에서 중복되었습니다")
    if len(table_ids) != len(set(table_ids)):
        raise ValueError("table_id가 문서 안에서 중복되었습니다")
    if len(picture_ids) != len(set(picture_ids)):
        raise ValueError("picture_id가 문서 안에서 중복되었습니다")

    block_id_set = set(block_ids)
    table_id_set = set(table_ids)
    picture_id_set = set(picture_ids)
    records = [*blocks, *tables, *images]
    if any(record.get("source_id") != source.source_id for record in records):
        raise ValueError("전처리 레코드의 source_id가 원본과 다릅니다")
    if any(record.get("document_id") != source.document_id for record in records):
        raise ValueError("전처리 레코드의 document_id가 원본과 다릅니다")
    if any(table.get("render_mode") != "gfm" for table in tables):
        raise ValueError("Naive RAG의 모든 표는 GFM Markdown이어야 합니다")

    for manifest in [*tables, *images]:
        if manifest.get("parent_block_id") not in block_id_set:
            raise ValueError(
                "표·이미지의 parent_block_id가 실제 블록을 가리키지 않습니다"
            )
        parent_table_id = manifest.get("parent_table_id")
        if parent_table_id is not None and parent_table_id not in table_id_set:
            raise ValueError(
                "중첩 객체의 parent_table_id가 실제 표를 가리키지 않습니다"
            )

    for block in blocks:
        if block.get("table_id") is not None and block["table_id"] not in table_id_set:
            raise ValueError("블록의 table_id가 표 manifest에 없습니다")
        if (
            block.get("picture_id") is not None
            and block["picture_id"] not in picture_id_set
        ):
            raise ValueError("블록의 picture_id가 이미지 manifest에 없습니다")
        if block.get("index_policy") not in {"index", "flatten", "exclude"}:
            raise ValueError("알 수 없는 index_policy가 있습니다")
        if block.get("index_policy") in {"index", "flatten"} and not compact_text(
            str(block.get("retrieval_text") or "")
        ):
            raise ValueError("검색 대상 블록의 retrieval_text가 비어 있습니다")

    file_type = source.file_type.casefold()
    if file_type in HWP_FILE_TYPES:
        if any(record.get("page") is not None for record in records):
            raise ValueError("HWP/HWPX 페이지 번호는 추측하지 않고 None이어야 합니다")
    else:
        if any(
            not isinstance(record.get("page"), int) or record["page"] < 1
            for record in records
        ):
            raise ValueError("PDF 페이지 번호는 1 이상의 정수여야 합니다")

    rendered_content = "\n".join(
        str(block.get("display_content") or "") for block in blocks
    )
    table_owner_ids = {table.get("parent_block_id") for table in tables}
    for block in blocks:
        if block.get("block_id") not in table_owner_ids:
            continue
        display_content = str(block.get("display_content") or "")
        if HTML_TABLE_TAG.search(display_content):
            raise ValueError("표 표시 내용에 HTML 태그가 포함되어 있습니다")

    for image in images:
        picture_id = str(image["picture_id"])
        if rendered_content.count(f"image://{picture_id}") != 1:
            raise ValueError("이미지 placeholder가 정확히 한 번 표시되지 않았습니다")
        if image.get("index_policy") != "exclude":
            raise ValueError("이미지는 항상 검색에서 제외해야 합니다")
        uri = str(image.get("original_ir_uri") or "")
        if uri and not uri.casefold().startswith("bin://"):
            raise ValueError("허용되지 않은 이미지 원본 URI가 있습니다")

    if has_forbidden_image_payload([blocks, tables, images]):
        raise ValueError(
            "전처리 결과에서 금지된 이미지 payload 또는 data URI를 발견했습니다"
        )


def _import_required_module(module_name: str, package_name: str) -> Any:
    """형식별 파서를 실제로 사용할 때만 불러온다."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise PreprocessingDependencyError(
            f"{package_name} 패키지가 필요합니다. 프로젝트 폴더에서 uv sync를 실행하세요."
        ) from error


def _validated_analysis_path(
    source: SourceDocument,
    analysis_source: VerifiedAnalysisSource | None,
) -> tuple[Path, str, str]:
    """원본 동일성과 분석 파일의 검증된 연결 관계를 확인한다."""
    source_path = source.source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"loader가 확인한 원본 문서를 찾을 수 없습니다: {source_path}"
        )
    if source.document_id != source.source_id:
        raise ValueError("document_id와 source_id가 서로 다릅니다")
    if source.source_id != source.source_sha256[:16]:
        raise ValueError("source_id가 loader의 SHA-256 축약 규칙과 다릅니다")

    # loader 이후 파일이 바뀌었는데 이전 ID로 처리되는 일을 막는다.
    current_source_sha256 = sha256_file(source_path)
    if current_source_sha256 != source.source_sha256:
        raise ValueError("loader 실행 뒤 원본 파일 내용이 변경되었습니다")

    if analysis_source is None:
        path = source_path
        expected_analysis_sha256 = current_source_sha256
        verification_source = "loader_source_sha256"
    else:
        if analysis_source.original_source_id != source.source_id:
            raise ValueError("검증된 분석본의 original_source_id가 원본과 다릅니다")
        if analysis_source.original_source_sha256 != source.source_sha256:
            raise ValueError("검증된 분석본의 original_source_sha256이 원본과 다릅니다")
        verification_source = analysis_source.verification_source.strip()
        if not verification_source:
            raise ValueError("대체 분석본의 검증 출처가 필요합니다")
        path = Path(analysis_source.path).expanduser().resolve()
        expected_analysis_sha256 = analysis_source.sha256

    if not path.is_file():
        raise FileNotFoundError(f"분석할 원본 문서를 찾을 수 없습니다: {path}")

    analysis_type = path.suffix.casefold().lstrip(".")
    if source.file_type in HWP_FILE_TYPES and analysis_type not in HWP_FILE_TYPES:
        raise ValueError("HWP/HWPX 원본의 분석 파일은 .hwp 또는 .hwpx여야 합니다")
    if source.file_type == PDF_FILE_TYPE and analysis_type != PDF_FILE_TYPE:
        raise ValueError("PDF 원본의 분석 파일은 .pdf여야 합니다")
    analysis_sha256 = (
        current_source_sha256 if path == source_path else sha256_file(path)
    )
    if analysis_sha256 != expected_analysis_sha256:
        raise ValueError("분석 파일의 SHA-256이 검증 이력과 다릅니다")
    return path, analysis_sha256, verification_source


def preprocess_document(
    source: SourceDocument,
    *,
    analysis_source: VerifiedAnalysisSource | None = None,
    rhwp_module: Any | None = None,
    pdfplumber_module: Any | None = None,
) -> PreprocessingResult:
    """loader 결과 한 개를 구조화해 chunking 단계에 넘길 결과를 만든다.

    복구 HWPX처럼 다른 파일을 읽어야 하면 ``VerifiedAnalysisSource``에 원본과
    분석본의 검증 정보를 모두 넣는다. 단순 경로는 받지 않아 오연결을 막는다.
    테스트에서는 ``rhwp_module``/``pdfplumber_module``에 가짜 파서를 넣을 수 있다.
    """
    if not isinstance(source, SourceDocument):
        raise TypeError("source는 loader의 SourceDocument여야 합니다")
    if source.duplicate_group_size > 1 and source.canonical_selection_source not in {
        "loader_default",
        "team_policy",
    }:
        raise ValueError(
            "중복 원본은 loader의 select_canonical_documents로 대표를 먼저 선택해야 합니다"
        )
    file_type = source.file_type.casefold()
    if file_type not in {*HWP_FILE_TYPES, PDF_FILE_TYPE}:
        raise ValueError(f"지원하지 않는 문서 형식입니다: {source.file_type}")

    selected_path, analysis_sha256, analysis_verification_source = (
        _validated_analysis_path(source, analysis_source)
    )
    if file_type in HWP_FILE_TYPES:
        parser = rhwp_module or _import_required_module("rhwp", "rhwp-python==0.8.1")
        blocks, tables, images, stats = process_hwp_document(
            source,
            selected_path,
            parser,
        )
    else:
        parser = pdfplumber_module or _import_required_module(
            "pdfplumber", "pdfplumber==0.11.10"
        )
        blocks, tables, images, stats = process_pdf_document(
            source,
            parser,
            analysis_path=selected_path,
        )

    validate_preprocessing_contract(source, blocks, tables, images)

    source_path = source.source_path.expanduser().resolve()
    analysis_is_original = selected_path == source_path
    quality_flags = set(stats.get("quality_flags", []))
    if not analysis_is_original:
        quality_flags.add("alternate_analysis_source")
    indexable_blocks = [
        block
        for block in blocks
        if block["index_policy"] in {"index", "flatten"}
        and compact_text(str(block.get("retrieval_text") or ""))
    ]
    indexable_retrieval_chars = sum(
        len(str(block["retrieval_text"])) for block in indexable_blocks
    )
    chunking_ready = bool(indexable_blocks)
    if not chunking_ready:
        quality_flags.add("no_indexable_content")

    document = {
        "schema_version": SCHEMA_VERSION,
        **source.as_metadata(),
        "analysis_source_filename": unicodedata.normalize("NFC", selected_path.name),
        "analysis_source_sha256": analysis_sha256,
        "analysis_file_type": selected_path.suffix.casefold().lstrip("."),
        "analysis_source_is_original": analysis_is_original,
        "analysis_source_relationship_verified": True,
        "analysis_source_verification": analysis_verification_source,
        "source_identity_preserved": True,
        "parser": stats["parser"],
        "parse_status": "success",
        "page_count": stats.get("page_count"),
        "section_count": stats.get("section_count"),
        "paragraph_count": stats.get("paragraph_count"),
        "block_count": len(blocks),
        "table_count": len(tables),
        "picture_count": len(images),
        "indexable_block_count": len(indexable_blocks),
        "indexable_retrieval_chars": indexable_retrieval_chars,
        "chunking_ready": chunking_ready,
        "body_table_count": stats.get("body_table_count", 0),
        "furniture_table_count": stats.get("furniture_table_count", 0),
        "body_picture_count": stats.get("body_picture_count", 0),
        "furniture_picture_count": stats.get("furniture_picture_count", 0),
        "picture_placeholder_count": stats.get("picture_placeholder_count", 0),
        "pdf_table_error_count": stats.get("pdf_table_error_count", 0),
        "pdf_word_extraction_error_count": stats.get(
            "pdf_word_extraction_error_count", 0
        ),
        "pdf_fallback_text_error_count": stats.get("pdf_fallback_text_error_count", 0),
        "pdf_table_text_fallback_count": stats.get("pdf_table_text_fallback_count", 0),
        "pdf_table_text_fallback_page_count": stats.get(
            "pdf_table_text_fallback_page_count", 0
        ),
        "text_storage": "in_memory_only",
        "image_storage": "metadata_only_no_payload",
        "quality_flags": sorted(quality_flags),
    }

    complete_result = [document, blocks, tables, images]
    if has_forbidden_image_payload(complete_result):
        raise ValueError(
            "전처리 결과에서 금지된 이미지 payload 또는 Base64를 발견했습니다"
        )

    return PreprocessingResult(
        document=document,
        blocks=tuple(blocks),
        tables=tuple(tables),
        images=tuple(images),
    )


def preprocess_documents(
    sources: Iterable[SourceDocument],
    *,
    analysis_sources: Mapping[str, VerifiedAnalysisSource] | None = None,
    rhwp_module: Any | None = None,
    pdfplumber_module: Any | None = None,
) -> tuple[PreprocessingResult, ...]:
    """대표 문서 목록을 순서대로 전처리한다.

    같은 SHA의 별칭을 두 번 처리하지 않도록 source_id 중복을 즉시 오류로
    알린다. loader의 ``select_default_canonical_documents`` 결과를 넘기면 된다.
    파서의 메모리 사용과 thread safety가 확실하지 않아 내부 병렬화는 하지 않는다.
    """
    selected_analysis_sources = analysis_sources or {}
    source_list = tuple(sources)
    seen_source_ids: set[str] = set()
    duplicate_source_ids: set[str] = set()
    for source in source_list:
        if not isinstance(source, SourceDocument):
            raise TypeError("sources의 모든 항목은 SourceDocument여야 합니다")
        if source.source_id in seen_source_ids:
            duplicate_source_ids.add(source.source_id)
        seen_source_ids.add(source.source_id)

    if duplicate_source_ids:
        raise ValueError(
            "같은 source_id가 두 번 입력되었습니다. "
            "loader에서 기본 대표 문서만 선택하세요."
        )
    unknown_analysis_ids = set(selected_analysis_sources) - seen_source_ids
    if unknown_analysis_ids:
        raise ValueError("입력 문서에 없는 source_id의 analysis_source가 있습니다")

    results: list[PreprocessingResult] = []
    for source in source_list:
        results.append(
            preprocess_document(
                source,
                analysis_source=selected_analysis_sources.get(source.source_id),
                rhwp_module=rhwp_module,
                pdfplumber_module=pdfplumber_module,
            )
        )
    return tuple(results)
