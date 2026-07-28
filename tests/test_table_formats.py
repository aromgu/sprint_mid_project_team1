"""PDF 표의 병합(rowspan/colspan) 보존 표현 계약을 검증한다.

pdfplumber ``Table.cells`` 좌표와 ``Table.extract()`` 행렬을 함께 넘기면 HTML은
rowspan·colspan을 유지하고, 임베딩용 Markdown은 병합 값을 각 행에 반복해 채운다.
좌표가 없거나 행렬과 맞지 않으면 병합을 추측하지 않고 기존 평면 표현으로
되돌아간다.
"""

from __future__ import annotations

from src.preprocessing.table_formats import (
    build_pdf_table_formats,
    build_pdf_table_grid,
    render_pdf_table_html_spans,
)


def test_grid_recovers_rowspan_from_cell_geometry() -> None:
    """세로 병합 셀은 좌표에서 rowspan으로 복원된다."""
    matrix = [["A", "B", "C"], [None, "E", "F"], ["G", "H", "I"]]
    cell_bboxes = [
        (0, 0, 10, 20),
        (10, 0, 20, 10),
        (20, 0, 30, 10),
        (10, 10, 20, 20),
        (20, 10, 30, 20),
        (0, 20, 10, 30),
        (10, 20, 20, 30),
        (20, 20, 30, 30),
    ]

    grid = build_pdf_table_grid(cell_bboxes, matrix)

    assert grid is not None
    assert grid["row_count"] == 3
    assert grid["col_count"] == 3
    merged = [span for span in grid["spans"] if span["rowspan"] > 1]
    assert len(merged) == 1
    assert merged[0]["text"] == "A"
    assert merged[0]["rowspan"] == 2
    assert merged[0]["colspan"] == 1


def test_grid_recovers_colspan_from_cell_geometry() -> None:
    """가로 병합 셀은 좌표에서 colspan으로 복원된다."""
    matrix = [["TITLE", None, "X"], ["a", "b", "c"]]
    cell_bboxes = [
        (0, 0, 20, 10),
        (20, 0, 30, 10),
        (0, 10, 10, 20),
        (10, 10, 20, 20),
        (20, 10, 30, 20),
    ]

    grid = build_pdf_table_grid(cell_bboxes, matrix)

    assert grid is not None
    merged = [span for span in grid["spans"] if span["colspan"] > 1]
    assert len(merged) == 1
    assert merged[0]["text"] == "TITLE"
    assert merged[0]["colspan"] == 2
    assert merged[0]["rowspan"] == 1


def test_html_preserves_rowspan_and_markdown_fills_merged_value() -> None:
    """HTML은 rowspan을 유지하고 Markdown은 병합 값을 각 행에 반복한다."""
    matrix = [["A", "B", "C"], [None, "E", "F"], ["G", "H", "I"]]
    cell_bboxes = [
        (0, 0, 10, 20),
        (10, 0, 20, 10),
        (20, 0, 30, 10),
        (10, 10, 20, 20),
        (20, 10, 30, 20),
        (0, 20, 10, 30),
        (10, 20, 20, 30),
        (20, 20, 30, 30),
    ]

    formats = build_pdf_table_formats(matrix, "T1", cell_bboxes=cell_bboxes)

    # 병합 셀 "A"는 헤더 행에서 시작하므로 rowspan을 가진 <th>로 렌더링된다.
    assert '<th rowspan="2">A</th>' in formats["table_html"]
    assert formats["table_html"].startswith('<table data-table-id="T1">')
    lines = formats["table_markdown"].splitlines()
    # 헤더(row0) + 구분선 다음 행에서 병합된 "A"가 반복 채워진다.
    assert lines[0] == "| A | B | C |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| A | E | F |"
    assert lines[3] == "| G | H | I |"
    assert formats["vectorize_field"] == "table_markdown"


def test_html_uses_colspan_for_horizontal_merge() -> None:
    """가로 병합 셀은 HTML에서 colspan으로 렌더링된다."""
    matrix = [["TITLE", None, "X"], ["a", "b", "c"]]
    cell_bboxes = [
        (0, 0, 20, 10),
        (20, 0, 30, 10),
        (0, 10, 10, 20),
        (10, 10, 20, 20),
        (20, 10, 30, 20),
    ]

    grid = build_pdf_table_grid(cell_bboxes, matrix)
    html = render_pdf_table_html_spans(grid, "T2")

    assert '<th colspan="2">TITLE</th>' in html


def test_grid_returns_none_when_geometry_does_not_reconcile() -> None:
    """좌표 격자가 행렬과 맞지 않으면 병합을 추측하지 않는다."""
    matrix = [["A", "B", "C"], ["D", "E", "F"]]
    # 열 경계가 2개뿐이라 3열 행렬과 맞지 않는다.
    cell_bboxes = [
        (0, 0, 10, 10),
        (10, 0, 20, 10),
        (0, 10, 10, 20),
        (10, 10, 20, 20),
    ]

    assert build_pdf_table_grid(cell_bboxes, matrix) is None


def test_build_formats_falls_back_to_flat_without_geometry() -> None:
    """좌표가 없으면 기존 평면 HTML(thead/tbody)로 되돌아간다."""
    matrix = [["구분", "내용"], ["기간", "90일"]]

    formats = build_pdf_table_formats(matrix, "T3")

    assert "rowspan" not in formats["table_html"]
    assert "colspan" not in formats["table_html"]
    assert "<thead>" in formats["table_html"]
    assert formats["table_markdown"] == (
        "| 구분 | 내용 |\n| --- | --- |\n| 기간 | 90일 |"
    )


def test_overlapping_geometry_falls_back_to_flat() -> None:
    """셀이 겹치면 좌표를 신뢰하지 않고 평면 표현으로 되돌린다."""
    matrix = [["A", "B"], ["C", "D"]]
    cell_bboxes = [
        (0, 0, 20, 10),  # 첫 행 전체를 덮어 아래 두 셀과 겹친다.
        (0, 0, 10, 10),
        (10, 0, 20, 10),
        (0, 10, 10, 20),
        (10, 10, 20, 20),
    ]

    assert build_pdf_table_grid(cell_bboxes, matrix) is None
