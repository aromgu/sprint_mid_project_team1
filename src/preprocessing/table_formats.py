"""Advanced RAG용 표의 HTML·Markdown 표현을 생성한다.

검색과 임베딩에는 Markdown을 사용하고, 병합 셀과 중첩 표 등 원본 구조
보존에는 HTML을 사용한다. 이미지 바이트는 저장하지 않고 ``image://``
참조만 남긴다.

기존 Naive 전처리 코드는 수정하지 않는다.
"""

from __future__ import annotations

import html as html_lib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.preprocessing.clean_text import (
    REFERENCE_MODE_METADATA,
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
    table_reference_ids,
)

__all__ = [
    "build_hwp_table_formats",
    "build_pdf_table_formats",
    "build_pdf_table_grid",
    "render_hwp_table_html",
    "render_pdf_table_html",
    "render_pdf_table_html_spans",
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
) -> dict[str, Any]:
    """HWP 표의 저장용 HTML과 벡터화용 Markdown을 함께 반환한다.

    Advanced 경로 전용이므로 중첩 표·이미지 참조는 임베딩 본문에 넣지 않고
    ``nested_table_refs``·``image_refs``로만 돌려준다(팀 회의 결정 2026-07-28).
    참조의 셀 위치는 ``table_html``이 그대로 보존한다.
    """
    nested_refs, image_refs = table_reference_ids(block, table_ids, picture_ids)
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
            reference_mode=REFERENCE_MODE_METADATA,
        ),
        "vectorize_field": "table_markdown",
        "nested_table_refs": nested_refs,
        "image_refs": image_refs,
    }


def _cluster_edges(values: Iterable[float], tolerance: float) -> list[float]:
    """tolerance 안에서 가까운 좌표들을 하나의 격자 경계로 묶는다."""
    ordered = sorted(values)
    edges: list[float] = []
    for value in ordered:
        if not edges or value - edges[-1] > tolerance:
            edges.append(value)
    return edges


def _edge_index(value: float, edges: Sequence[float], tolerance: float) -> int | None:
    """좌표가 속한 격자 경계 인덱스를 tolerance 안에서 찾는다."""
    best_index: int | None = None
    best_distance = tolerance
    for index, edge in enumerate(edges):
        distance = abs(value - edge)
        if distance <= best_distance:
            best_distance = distance
            best_index = index
    return best_index


def build_pdf_table_grid(
    cell_bboxes: Sequence[Sequence[float]] | None,
    matrix: Sequence[Sequence[Any]],
    *,
    tolerance: float = 3.0,
) -> dict[str, Any] | None:
    """pdfplumber 셀 좌표와 추출 행렬로 병합(rowspan/colspan) 격자를 복원한다.

    좌표와 행렬이 일관되게 맞지 않으면 병합을 추측하지 않고 ``None``을 반환해
    호출부가 기존 평면 렌더링으로 안전하게 되돌아가게 한다. 좌표는
    ``(x0, top, x1, bottom)`` 순서이며 pdfplumber ``Table.cells``와 같다.
    """
    if not cell_bboxes or not matrix:
        return None
    row_count = len(matrix)
    col_count = max((len(row) for row in matrix), default=0)
    if row_count == 0 or col_count == 0:
        return None
    for bbox in cell_bboxes:
        if len(bbox) < 4 or any(value is None for value in bbox[:4]):
            return None

    x_edges = _cluster_edges(
        (float(coord) for bbox in cell_bboxes for coord in (bbox[0], bbox[2])),
        tolerance,
    )
    y_edges = _cluster_edges(
        (float(coord) for bbox in cell_bboxes for coord in (bbox[1], bbox[3])),
        tolerance,
    )
    # 격자 경계 개수는 행렬 행·열 수와 정확히 일치해야 병합을 신뢰할 수 있다.
    if len(x_edges) - 1 != col_count or len(y_edges) - 1 != row_count:
        return None

    occupancy = [[0] * col_count for _ in range(row_count)]
    spans: list[dict[str, Any]] = []
    for bbox in cell_bboxes:
        left = _edge_index(float(bbox[0]), x_edges, tolerance)
        right = _edge_index(float(bbox[2]), x_edges, tolerance)
        top = _edge_index(float(bbox[1]), y_edges, tolerance)
        bottom = _edge_index(float(bbox[3]), y_edges, tolerance)
        if left is None or right is None or top is None or bottom is None:
            return None
        if not (0 <= left < right <= col_count):
            return None
        if not (0 <= top < bottom <= row_count):
            return None
        source_row = matrix[top]
        raw = source_row[left] if left < len(source_row) else ""
        spans.append(
            {
                "row": top,
                "col": left,
                "rowspan": bottom - top,
                "colspan": right - left,
                "text": "" if raw is None else str(raw),
            }
        )
        for row_index in range(top, bottom):
            for col_index in range(left, right):
                occupancy[row_index][col_index] += 1

    # 모든 격자 칸이 정확히 한 셀에 덮여야 한다. 겹침·빈칸이 있으면 좌표를
    # 신뢰할 수 없으므로 평면 렌더링으로 되돌린다.
    if any(count != 1 for row in occupancy for count in row):
        return None

    filled_matrix = [["" for _ in range(col_count)] for _ in range(row_count)]
    for span in spans:
        for row_index in range(span["row"], span["row"] + span["rowspan"]):
            for col_index in range(span["col"], span["col"] + span["colspan"]):
                filled_matrix[row_index][col_index] = span["text"]

    return {
        "row_count": row_count,
        "col_count": col_count,
        "spans": spans,
        "filled_matrix": filled_matrix,
    }


def render_pdf_table_html_spans(grid: Mapping[str, Any], table_id: str) -> str:
    """복원한 격자를 rowspan·colspan을 보존한 HTML 표로 만든다."""
    spans_by_row: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for span in grid["spans"]:
        spans_by_row[int(span["row"])].append(span)

    parts = [f'<table data-table-id="{html_lib.escape(table_id, quote=True)}">']
    for row_index in range(int(grid["row_count"])):
        parts.append("<tr>")
        tag = "th" if row_index == 0 else "td"
        for span in sorted(
            spans_by_row.get(row_index, []), key=lambda item: int(item["col"])
        ):
            attributes: list[str] = []
            row_span = int(span["rowspan"])
            col_span = int(span["colspan"])
            if row_span > 1:
                attributes.append(f'rowspan="{row_span}"')
            if col_span > 1:
                attributes.append(f'colspan="{col_span}"')
            attribute_text = f" {' '.join(attributes)}" if attributes else ""
            content = _escape_html_text(str(span["text"])) or "&nbsp;"
            parts.append(f"<{tag}{attribute_text}>{content}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def build_pdf_table_formats(
    matrix: Sequence[Sequence[Any]],
    table_id: str,
    *,
    cell_bboxes: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """PDF 표의 저장용 HTML과 벡터화용 Markdown을 함께 반환한다.

    pdfplumber 셀 좌표(``cell_bboxes``)가 주어지고 추출 행렬과 일관되면 병합을
    보존한다. HTML은 rowspan·colspan을 유지하고, 임베딩용 Markdown은 병합 값을
    각 행에 반복해 채워 행 단위 검색을 보존한다. 좌표가 없거나 맞지 않으면 기존
    평면 표현으로 되돌아간다.
    """
    grid = build_pdf_table_grid(cell_bboxes, matrix)
    if grid is not None:
        return {
            "table_html": render_pdf_table_html_spans(grid, table_id),
            "table_markdown": render_pdf_table(grid["filled_matrix"]),
            "vectorize_field": "table_markdown",
            # pdfplumber 행렬에는 중첩 표·이미지가 없다. HWP와 계약을 맞춘다.
            "nested_table_refs": [],
            "image_refs": [],
        }
    return {
        "table_html": render_pdf_table_html(matrix, table_id),
        "table_markdown": render_pdf_table(matrix),
        "nested_table_refs": [],
        "image_refs": [],
        "vectorize_field": "table_markdown",
    }
