"""Advanced RAG용 표의 HTML·Markdown 표현을 생성한다.

검색과 임베딩에는 Markdown을 사용하고, 병합 셀과 중첩 표 등 원본 구조
보존에는 HTML을 사용한다. 이미지 바이트는 저장하지 않고 ``image://``
참조만 남긴다.

기존 Naive 전처리 코드는 수정하지 않는다.
"""

from __future__ import annotations

import html as html_lib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src.preprocessing.clean_text import (
    block_display_text,
    caption_text,
    child_blocks,
    kind_name,
    list_item_display_text,
    normalize_text,
    note_display_text,
    picture_alt,
    render_pdf_table,
    render_table_gfm,
)

__all__ = [
    "build_hwp_table_formats",
    "build_pdf_table_formats",
    "render_hwp_table_html",
    "render_pdf_table_html",
]


def _escape_html_text(value: str | None) -> str:
    """표 셀의 텍스트를 안전한 HTML 문자열로 변환한다."""
    escaped = html_lib.escape(normalize_text(value or ""), quote=False)
    return escaped.replace("\n", "<br>")


def _render_picture_html(block: Any, picture_id: str) -> str:
    """이미지를 Base64 없이 ``image://`` 참조로 표현한다."""
    uri = f"image://{picture_id}"
    alt = picture_alt(block, picture_id)
    return (
        f'<img src="{html_lib.escape(uri, quote=True)}" '
        f'alt="{html_lib.escape(alt, quote=True)}">'
    )


def _render_hwp_cell_block(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> str:
    """HWP 표 셀의 자식 블록을 compact HTML로 렌더링한다."""
    kind = kind_name(block)

    if kind == "table":
        return render_hwp_table_html(block, table_ids, picture_ids)
    if kind == "picture":
        return _render_picture_html(block, picture_ids[id(block)])

    if kind == "list_item":
        own_text = list_item_display_text(block)
    elif kind in {"footnote", "endnote"}:
        own_text = note_display_text(block)
    else:
        own_text = block_display_text(block)

    parts: list[str] = []
    if own_text:
        parts.append(f"<p>{_escape_html_text(own_text)}</p>")

    # 일반 문단 text에는 자식 텍스트가 이미 합쳐진 경우가 많다.
    # 표와 이미지는 별도 구조이므로 항상 추가하고, own_text가 없을 때만
    # 나머지 자식을 순회해 같은 텍스트가 두 번 저장되는 것을 방지한다.
    for child in child_blocks(block):
        child_kind = kind_name(child)
        if child_kind in {"table", "picture"} or not own_text:
            rendered = _render_hwp_cell_block(
                child,
                table_ids,
                picture_ids,
            )
            if rendered:
                parts.append(rendered)

    return "".join(parts)


def render_hwp_table_html(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> str:
    """HWP 표의 병합 셀·중첩 표·이미지 참조를 HTML로 보존한다."""
    cells = list(getattr(block, "cells", []) or [])
    declared_rows = max(int(getattr(block, "rows", 0) or 0), 0)

    row_count = max(
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

    cells_by_row: dict[int, list[Any]] = defaultdict(list)
    for cell in sorted(
        cells,
        key=lambda item: (
            int(getattr(item, "row", 0) or 0),
            int(getattr(item, "col", 0) or 0),
        ),
    ):
        row = int(getattr(cell, "row", 0) or 0)
        cells_by_row[row].append(cell)

    table_id = table_ids[id(block)]
    parts = [f'<table data-table-id="{html_lib.escape(table_id, quote=True)}">']

    caption = caption_text(block)
    if caption:
        parts.append(f"<caption>{_escape_html_text(caption)}</caption>")

    if not cells:
        parts.append("<tr><td>&nbsp;</td></tr>")
    else:
        for row_index in range(row_count):
            parts.append("<tr>")

            for cell in cells_by_row.get(row_index, []):
                role = str(getattr(cell, "role", "") or "").casefold()
                tag = (
                    "th" if role in {"header", "column_header", "row_header"} else "td"
                )
                row_span = max(
                    int(getattr(cell, "row_span", 1) or 1),
                    1,
                )
                col_span = max(
                    int(getattr(cell, "col_span", 1) or 1),
                    1,
                )

                attributes: list[str] = []
                if row_span > 1:
                    attributes.append(f'rowspan="{row_span}"')
                if col_span > 1:
                    attributes.append(f'colspan="{col_span}"')
                attribute_text = f" {' '.join(attributes)}" if attributes else ""

                content = "".join(
                    _render_hwp_cell_block(
                        child,
                        table_ids,
                        picture_ids,
                    )
                    for child in (getattr(cell, "blocks", []) or [])
                )
                parts.append(f"<{tag}{attribute_text}>{content or '&nbsp;'}</{tag}>")

            parts.append("</tr>")

    parts.append("</table>")
    return "".join(parts)


def render_pdf_table_html(
    matrix: Sequence[Sequence[Any]],
    table_id: str,
) -> str:
    """pdfplumber 표 행렬을 구조 보존용 HTML로 변환한다."""
    normalized_rows = [
        ["" if value is None else str(value) for value in row] for row in matrix
    ]
    width = max((len(row) for row in normalized_rows), default=0)

    if width == 0:
        normalized_rows = [[""]]
        width = 1

    padded_rows = [row + [""] * (width - len(row)) for row in normalized_rows]

    parts = [
        f'<table data-table-id="{html_lib.escape(table_id, quote=True)}">',
        "<thead><tr>",
    ]
    parts.extend(
        f"<th>{_escape_html_text(value) or '&nbsp;'}</th>" for value in padded_rows[0]
    )
    parts.append("</tr></thead>")

    if len(padded_rows) > 1:
        parts.append("<tbody>")
        for row in padded_rows[1:]:
            parts.append("<tr>")
            parts.extend(
                f"<td>{_escape_html_text(value) or '&nbsp;'}</td>" for value in row
            )
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts)


def build_hwp_table_formats(
    block: Any,
    table_ids: dict[int, str],
    picture_ids: dict[int, str],
) -> dict[str, str]:
    """HWP 표의 저장용 HTML과 벡터화용 Markdown을 함께 반환한다."""
    return {
        "table_html": render_hwp_table_html(
            block,
            table_ids,
            picture_ids,
        ),
        "table_markdown": render_table_gfm(
            block,
            table_ids,
            picture_ids,
        ),
        "vectorize_field": "table_markdown",
    }


def build_pdf_table_formats(
    matrix: Sequence[Sequence[Any]],
    table_id: str,
) -> dict[str, str]:
    """PDF 표의 저장용 HTML과 벡터화용 Markdown을 함께 반환한다."""
    return {
        "table_html": render_pdf_table_html(matrix, table_id),
        "table_markdown": render_pdf_table(matrix),
        "vectorize_field": "table_markdown",
    }
