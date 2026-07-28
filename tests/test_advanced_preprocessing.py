"""Advanced RAG 표 이중 표현(HTML·Markdown) 테스트."""

from types import SimpleNamespace

from src.preprocessing.table_formats import (
    build_hwp_table_formats,
    build_pdf_table_formats,
)


def _paragraph(text: str) -> SimpleNamespace:
    """rhwp 문단 블록을 대신하는 최소 테스트 객체를 만든다."""
    return SimpleNamespace(kind="paragraph", text=text, blocks=[])


def _cell(
    row: int,
    col: int,
    text: str = "",
    *,
    row_span: int = 1,
    col_span: int = 1,
    role: str = "body",
    blocks: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """병합 정보와 자식 블록을 가진 HWP 표 셀을 만든다."""
    return SimpleNamespace(
        row=row,
        col=col,
        row_span=row_span,
        col_span=col_span,
        role=role,
        blocks=blocks if blocks is not None else [_paragraph(text)],
    )


def _table(
    cells: list[SimpleNamespace],
    *,
    rows: int,
    cols: int,
    caption: str = "",
) -> SimpleNamespace:
    """rhwp 표 블록을 대신하는 최소 테스트 객체를 만든다."""
    return SimpleNamespace(
        kind="table",
        rows=rows,
        cols=cols,
        cells=cells,
        caption=caption,
    )


def test_hwp_table_keeps_markdown_and_merged_cell_html() -> None:
    """Markdown은 검색에, HTML은 병합 셀 구조 보존에 사용한다."""
    table = _table(
        [
            _cell(
                0,
                0,
                "공통 과업",
                row_span=2,
                col_span=2,
                role="header",
            )
        ],
        rows=2,
        cols=2,
        caption="단계별 일정",
    )

    formats = build_hwp_table_formats(
        table,
        table_ids={id(table): "source:body:T000001"},
        picture_ids={},
    )

    assert formats["vectorize_field"] == "table_markdown"
    assert "|" in formats["table_markdown"]
    assert "[병합 2행×2열]" in formats["table_markdown"]
    assert "<table" not in formats["table_markdown"]
    assert 'data-table-id="source:body:T000001"' in formats["table_html"]
    assert 'rowspan="2"' in formats["table_html"]
    assert 'colspan="2"' in formats["table_html"]
    assert "<caption>단계별 일정</caption>" in formats["table_html"]


def test_hwp_table_uses_image_reference_without_binary_payload() -> None:
    """표 안 이미지는 Base64가 아니라 안정적인 image URI만 저장한다."""
    picture = SimpleNamespace(
        kind="picture",
        description="업무 흐름도",
        caption="",
        blocks=[],
    )
    table = _table(
        [_cell(0, 0, blocks=[picture])],
        rows=1,
        cols=1,
    )
    picture_id = "source:body:I000001"

    formats = build_hwp_table_formats(
        table,
        table_ids={id(table): "source:body:T000002"},
        picture_ids={id(picture): picture_id},
    )

    for value in (formats["table_html"], formats["table_markdown"]):
        assert f"image://{picture_id}" in value
        assert "data:" not in value
        assert "base64" not in value.casefold()


def test_hwp_table_escapes_untrusted_text_in_html() -> None:
    """원문 기호를 보존하되 실행 가능한 HTML 태그로 해석하지 않는다."""
    table = _table(
        [_cell(0, 0, '<script>alert("x")</script> & 안내')],
        rows=1,
        cols=1,
    )

    formats = build_hwp_table_formats(
        table,
        table_ids={id(table): "source:body:T000003"},
        picture_ids={},
    )

    assert "<script>" not in formats["table_html"]
    assert "&lt;script&gt;" in formats["table_html"]
    assert "&amp; 안내" in formats["table_html"]


def test_pdf_table_builds_dual_formats_and_vectorizes_markdown() -> None:
    """PDF 표도 HTML과 Markdown을 함께 저장하고 Markdown만 벡터화한다."""
    formats = build_pdf_table_formats(
        [["구분", "기간"], ["분석", "1개월"], ["승인 <완료>"]],
        table_id="pdf-source:p0003:T000001",
    )

    assert formats["vectorize_field"] == "table_markdown"
    assert formats["table_markdown"].startswith("| 구분 | 기간 |")
    assert "<table" not in formats["table_markdown"]
    assert '<table data-table-id="pdf-source:p0003:T000001">' in formats["table_html"]
    assert "<thead>" in formats["table_html"]
    assert "<tbody>" in formats["table_html"]
    assert "승인 &lt;완료&gt;" in formats["table_html"]
    assert "data:" not in formats["table_html"]
