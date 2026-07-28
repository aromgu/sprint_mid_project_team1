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
    # Markdown에는 병합 주석을 넣지 않고 값을 반복해 채운다. 2행×2열 병합이므로
    # 헤더 행과 데이터 행 모두에서 같은 값이 각 열에 채워진다.
    assert "[병합" not in formats["table_markdown"]
    assert formats["table_markdown"].count("공통 과업") == 4
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
        assert "data:" not in value
        assert "base64" not in value.casefold()

    # 팀 회의 결정(2026-07-28): 이미지 참조는 임베딩 본문에서 빼고 table_html과
    # image_refs에만 남긴다. 셀 위치는 HTML의 <img>가 보존한다.
    assert f"image://{picture_id}" in formats["table_html"]
    assert "image://" not in formats["table_markdown"]
    assert formats["image_refs"] == [picture_id]
    assert formats["nested_table_refs"] == []


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


def test_pdf_table_markdown_keeps_literal_characters() -> None:
    """table_markdown은 Dense 임베딩 본문이므로 원문 기호를 그대로 담는다.

    회귀 방지: ``공통 > 기준정보관리``가 ``공통 &gt; 기준정보관리``로 저장되면
    원문에 없는 HTML 엔티티가 임베딩된다. HTML 이스케이프는 table_html 몫이다.
    """
    formats = build_pdf_table_formats(
        [
            ["구분", "내용"],
            ["메뉴", "공통 > 기준정보관리 기능 정의"],
            ["조건", "처리시간 < 3초 & 동시접속 200명"],
        ],
        table_id="pdf-source:p0007:T000001",
    )

    markdown = formats["table_markdown"]
    assert "공통 > 기준정보관리 기능 정의" in markdown
    assert "처리시간 < 3초 & 동시접속 200명" in markdown
    assert "&gt;" not in markdown
    assert "&lt;" not in markdown
    assert "&amp;" not in markdown

    # 같은 원문을 HTML로 저장할 때는 여전히 HTML 이스케이프가 필요하다.
    html = formats["table_html"]
    assert "공통 &gt; 기준정보관리 기능 정의" in html
    assert "처리시간 &lt; 3초 &amp; 동시접속 200명" in html


def test_hwp_table_markdown_keeps_literal_characters() -> None:
    """HWP 표 셀과 캡션도 원문 기호를 엔티티로 바꾸지 않는다."""
    table = _table(
        [_cell(0, 0, "공통 > 기준정보관리 & 배점 < 30")],
        rows=1,
        cols=1,
        caption="<사후변경에 따른 복수의결권주식의 보통주 전환>",
    )

    formats = build_hwp_table_formats(
        table,
        table_ids={id(table): "source:body:T000004"},
        picture_ids={},
    )

    markdown = formats["table_markdown"]
    assert "공통 > 기준정보관리 & 배점 < 30" in markdown
    assert "<사후변경에 따른 복수의결권주식의 보통주 전환>" in markdown
    assert "&gt;" not in markdown
    assert "&lt;" not in markdown

    html = formats["table_html"]
    assert "공통 &gt; 기준정보관리 &amp; 배점 &lt; 30" in html


def test_table_markdown_still_neutralizes_html_table_markup() -> None:
    """원문이 표 HTML 문자열이면 Markdown 계약을 지키도록 중화한다.

    ``table_markdown``에 표 태그가 남으면 청킹 단계의 HTML_TABLE_TAG 가드가
    ValueError를 낸다. 원문 기호 보존과 달리 이 경우는 이스케이프가 필요하다.
    """
    formats = build_pdf_table_formats(
        [
            ["구분", "<table>문자열</table>"],
            ["스크립트", '<script>alert("x")</script>'],
        ],
        table_id="pdf-source:p0009:T000001",
    )

    markdown = formats["table_markdown"]
    assert "&lt;table&gt;문자열&lt;/table&gt;" in markdown
    assert "<table" not in markdown
    assert "</table" not in markdown
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
