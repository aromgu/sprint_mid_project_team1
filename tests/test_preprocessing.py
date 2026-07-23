from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.loader.load_documents import SourceDocument
from src.preprocessing.clean_text import (
    VerifiedAnalysisSource,
    classify_table,
    field_parts,
    has_forbidden_image_payload,
    normalize_text,
    preprocess_document,
    preprocess_documents,
    render_pdf_table,
    render_table,
)


class PreprocessingTests(unittest.TestCase):
    """loader 다음 단계인 구조 전처리의 공통 계약을 검증한다."""

    def setUp(self) -> None:
        """실제 RFP 대신 작은 임시 파일을 만든다."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def source(self, filename: str, content: bytes = b"source") -> SourceDocument:
        """전처리 테스트에 사용할 loader 결과 한 개를 만든다."""
        path = self.root / filename
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return SourceDocument(
            source_id=digest[:16],
            document_id=digest[:16],
            source_path=path,
            source_relative_path=filename,
            source_filename=filename,
            source_sha256=digest,
            file_type=path.suffix.casefold().lstrip("."),
            source_file_size_bytes=len(content),
            duplicate_group_size=1,
            is_default_canonical=True,
            default_canonical_filename=filename,
        )

    @staticmethod
    def paragraph(
        text: str,
        *,
        section_idx: int | None = None,
        para_idx: int | None = None,
    ) -> SimpleNamespace:
        """원문 위치를 선택적으로 가진 가짜 rhwp 문단을 만든다."""
        provenance = (
            SimpleNamespace(section_idx=section_idx, para_idx=para_idx)
            if section_idx is not None or para_idx is not None
            else None
        )
        return SimpleNamespace(
            kind="paragraph",
            text=text,
            blocks=[],
            prov=provenance,
        )

    @staticmethod
    def picture(
        description: str = "구조도",
        *,
        uri: str = "bin://1",
    ) -> SimpleNamespace:
        """이미지 바이트 없이 참조 메타데이터만 가진 가짜 그림을 만든다."""
        return SimpleNamespace(
            kind="picture",
            image=SimpleNamespace(
                uri=uri,
                mime_type="image/png",
                width=640,
                height=480,
                dpi=96,
            ),
            description=description,
            caption=None,
            text="",
            blocks=[],
            prov=SimpleNamespace(section_idx=0, para_idx=2),
        )

    @staticmethod
    def cell(
        row: int,
        col: int,
        blocks: list[SimpleNamespace],
        *,
        row_span: int = 1,
        col_span: int = 1,
        role: str = "data",
    ) -> SimpleNamespace:
        """행·열·병합 정보를 가진 가짜 표 셀을 만든다."""
        return SimpleNamespace(
            row=row,
            col=col,
            row_span=row_span,
            col_span=col_span,
            role=role,
            blocks=blocks,
        )

    @staticmethod
    def table(
        rows: int,
        cols: int,
        cells: list[SimpleNamespace],
        *,
        caption: str | None = None,
    ) -> SimpleNamespace:
        """가짜 rhwp 표 블록을 만든다."""
        return SimpleNamespace(
            kind="table",
            rows=rows,
            cols=cols,
            cells=cells,
            text="",
            caption=caption,
            caption_block=None,
            prov=SimpleNamespace(section_idx=0, para_idx=1),
        )

    @staticmethod
    def rhwp_module(ir: Any) -> tuple[SimpleNamespace, SimpleNamespace]:
        """parse와 to_ir 호출 횟수를 셀 수 있는 가짜 rhwp 모듈을 만든다."""
        state = SimpleNamespace(parse_calls=[], to_ir_calls=0)

        class ParsedDocument:
            page_count = None
            section_count = 1
            paragraph_count = len(getattr(ir, "body", []) or [])

            def to_ir(self) -> Any:
                state.to_ir_calls += 1
                return ir

            def bytes_for_image(self, *_: Any) -> bytes:
                raise AssertionError("전처리에서 이미지 바이트를 읽으면 안 됩니다")

        def parse(path: str) -> ParsedDocument:
            state.parse_calls.append(path)
            return ParsedDocument()

        return SimpleNamespace(parse=parse), state

    def test_normalize_text_is_idempotent_and_keeps_indentation(self) -> None:
        """최소 정규화를 반복해도 결과와 첫 줄 들여쓰기가 유지된다."""
        original = "  제목  \r\n\r\n\r\n내용   \r마지막  "

        once = normalize_text(original)
        twice = normalize_text(once)

        self.assertEqual(once, twice)
        self.assertEqual(once, "  제목\n\n내용\n마지막")

    def test_hwp_and_hwpx_both_use_rhwp_to_ir_once(self) -> None:
        """HWP와 HWPX는 모두 rhwp로 파싱하고 IR 변환은 한 번만 한다."""
        for extension in ("hwp", "hwpx"):
            with self.subTest(extension=extension):
                source = self.source(f"sample.{extension}", extension.encode())
                ir = SimpleNamespace(
                    body=[self.paragraph("본문", section_idx=0, para_idx=0)],
                    furniture=None,
                )
                fake_rhwp, state = self.rhwp_module(ir)

                result = preprocess_document(source, rhwp_module=fake_rhwp)

                self.assertEqual(len(state.parse_calls), 1)
                self.assertEqual(state.to_ir_calls, 1)
                self.assertEqual(result.document["parser"], "rhwp.to_ir")
                self.assertEqual(result.blocks[0]["page"], None)
                self.assertEqual(result.blocks[0]["section_idx"], 0)
                self.assertEqual(result.blocks[0]["para_idx"], 0)

    def test_preprocessing_preserves_loader_source_identity(self) -> None:
        """전처리 후에도 loader의 ID·SHA·파일명은 바뀌지 않는다."""
        source = self.source("identity.hwp", b"identity")
        ir = SimpleNamespace(body=[self.paragraph("본문")], furniture=None)
        fake_rhwp, _ = self.rhwp_module(ir)

        result = preprocess_document(source, rhwp_module=fake_rhwp)

        self.assertEqual(result.document["source_id"], source.source_id)
        self.assertEqual(result.document["document_id"], source.document_id)
        self.assertEqual(result.document["source_sha256"], source.source_sha256)
        self.assertEqual(result.document["source_filename"], source.source_filename)
        self.assertTrue(result.document["source_identity_preserved"])
        self.assertNotIn("source_path", result.document)

    def test_preprocessing_rejects_source_changed_after_loading(self) -> None:
        """loader가 해시를 만든 뒤 원본이 바뀌면 이전 ID로 처리하지 않는다."""
        source = self.source("changed.hwp", b"before")
        source.source_path.write_bytes(b"after")
        fake_rhwp, state = self.rhwp_module(
            SimpleNamespace(body=[self.paragraph("본문")], furniture=None)
        )

        with self.assertRaisesRegex(ValueError, "원본 파일 내용이 변경"):
            preprocess_document(source, rhwp_module=fake_rhwp)

        self.assertEqual(state.parse_calls, [])

    def test_table_classification_covers_all_policies(self) -> None:
        """내용 표·배치 표·빈 표·이미지 표를 서로 구분한다."""
        content = self.table(
            2,
            2,
            [
                self.cell(0, 0, [self.paragraph("요구사항 ID")], role="header"),
                self.cell(0, 1, [self.paragraph("세부내용")], role="header"),
                self.cell(1, 0, [self.paragraph("REQ-001")]),
                self.cell(1, 1, [self.paragraph("로그인 기능")]),
            ],
        )
        layout = self.table(1, 1, [self.cell(0, 0, [self.paragraph("문서 장식")])])
        empty = self.table(1, 1, [self.cell(0, 0, [])])
        picture_only = self.table(
            1,
            1,
            [self.cell(0, 0, [self.picture()])],
        )

        self.assertEqual(classify_table(content)[:2], ("content", "index"))
        self.assertEqual(classify_table(layout)[:2], ("layout", "flatten"))
        self.assertEqual(classify_table(empty)[:2], ("empty", "exclude"))
        self.assertEqual(classify_table(picture_only)[:2], ("layout", "exclude"))

    def test_empty_pdf_table_still_has_markdown_structure(self) -> None:
        """PDF 표 내용 추출이 비어도 Markdown 표 골격은 유지한다."""
        self.assertEqual(render_pdf_table([]), "|  |\n| --- |")

    def test_all_tables_use_markdown_even_when_cells_are_merged(self) -> None:
        """단순 표와 병합 표 모두 HTML 없이 GFM Markdown으로 만든다."""
        simple = self.table(
            1,
            2,
            [
                self.cell(0, 0, [self.paragraph("요구사항")]),
                self.cell(0, 1, [self.paragraph("보안")]),
            ],
        )
        merged = self.table(
            1,
            2,
            [
                self.cell(
                    0,
                    0,
                    [self.paragraph("요구사항 보안\n<table>문자열</table>")],
                    col_span=2,
                )
            ],
        )
        table_ids = {
            id(simple): "doc:body:T000001",
            id(merged): "doc:body:T000002",
        }

        simple_content, simple_mode = render_table(simple, table_ids, {})
        merged_content, merged_mode = render_table(merged, table_ids, {})

        self.assertEqual(classify_table(simple)[0], "content")
        self.assertEqual(classify_table(merged)[0], "content")
        self.assertEqual(simple_mode, "gfm")
        self.assertTrue(simple_content.startswith("| "))
        self.assertEqual(merged_mode, "gfm")
        self.assertTrue(merged_content.startswith("| "))
        self.assertIn("[병합 1행×2열]", merged_content)
        self.assertIn("1행 1열 참조", merged_content)
        self.assertIn("&lt;table&gt;문자열&lt;/table&gt;", merged_content)
        self.assertNotIn("<table", merged_content)
        self.assertNotIn("<br>", merged_content)

    def test_nested_tables_and_images_are_emitted_only_once(self) -> None:
        """중첩 표·이미지는 부모 표시 내용과 manifest에만 한 번 남는다."""
        picture = self.picture("업무 흐름도")
        inner = self.table(
            1,
            1,
            [self.cell(0, 0, [self.paragraph("안쪽 표"), picture])],
        )
        outer = self.table(
            1,
            1,
            [self.cell(0, 0, [self.paragraph("바깥 요구사항"), inner])],
        )
        source = self.source("nested.hwp")
        fake_rhwp, _ = self.rhwp_module(SimpleNamespace(body=[outer], furniture=None))

        result = preprocess_document(source, rhwp_module=fake_rhwp)

        self.assertEqual(len(result.blocks), 1)
        self.assertEqual(len(result.tables), 2)
        self.assertEqual(len(result.images), 1)
        display_content = result.blocks[0]["display_content"]
        self.assertNotIn("<table", display_content)
        self.assertEqual(display_content.count("| --- |"), 2)
        self.assertIn("[중첩 표:", display_content)
        self.assertIn("**중첩 표 `", display_content)
        self.assertEqual(display_content.count("image://"), 1)
        outer_record, inner_record = result.tables
        image_record = result.images[0]
        self.assertEqual(outer_record["render_mode"], "gfm")
        self.assertEqual(inner_record["render_mode"], "gfm")
        self.assertEqual(inner_record["parent_block_id"], result.blocks[0]["block_id"])
        self.assertEqual(inner_record["parent_table_id"], outer_record["table_id"])
        self.assertEqual(image_record["parent_table_id"], inner_record["table_id"])
        self.assertFalse(image_record["payload_stored"])
        self.assertEqual(image_record["index_policy"], "exclude")
        self.assertFalse(has_forbidden_image_payload(result.images))

    def test_merged_cell_does_not_duplicate_image_placeholder(self) -> None:
        """병합 범위를 펼쳐도 셀 안 이미지 참조는 원점에 한 번만 남긴다."""
        picture = self.picture("병합 셀 이미지")
        merged = self.table(
            1,
            2,
            [
                self.cell(
                    0,
                    0,
                    [self.paragraph("업무 구성"), picture],
                    col_span=2,
                )
            ],
        )
        source = self.source("merged-picture.hwp")
        fake_rhwp, _ = self.rhwp_module(SimpleNamespace(body=[merged], furniture=None))

        result = preprocess_document(source, rhwp_module=fake_rhwp)

        display_content = result.blocks[0]["display_content"]
        self.assertEqual(display_content.count("image://"), 1)
        self.assertIn("병합 1행×2열 계속", display_content)
        self.assertNotIn("<img", display_content)
        self.assertEqual(result.tables[0]["render_mode"], "gfm")

    def test_hwp_preserves_list_note_formula_and_toc_metadata(self) -> None:
        """목록·각주·수식·목차의 검색과 추적에 필요한 값을 보존한다."""
        list_item = SimpleNamespace(
            kind="list_item",
            text="첫 번째 조건",
            blocks=[],
            marker="1.",
            enumerated=True,
            level=2,
            prov=SimpleNamespace(section_idx=0, para_idx=0),
        )
        formula = SimpleNamespace(
            kind="formula",
            text_alt="1 / 2",
            script="1 over 2",
            script_kind="hwp_eq",
            inline=True,
            blocks=[],
            prov=SimpleNamespace(section_idx=0, para_idx=1),
        )
        toc_entry = SimpleNamespace(
            text="제 1 장 사업 개요",
            level=1,
            target_bookmark_name="chapter-1",
            target_section_idx=0,
            cached_page=3,
            is_stale=False,
            prov=SimpleNamespace(section_idx=0, para_idx=2),
        )
        toc = SimpleNamespace(
            kind="toc",
            entries=[toc_entry],
            blocks=[],
            prov=SimpleNamespace(section_idx=0, para_idx=2),
        )
        footnote = SimpleNamespace(
            kind="footnote",
            number=3,
            blocks=[self.paragraph("각주 설명")],
            marker_prov=SimpleNamespace(section_idx=0, para_idx=5),
            prov=SimpleNamespace(section_idx=0, para_idx=5),
        )
        furniture = SimpleNamespace(
            page_headers=[],
            page_footers=[],
            footnotes=[footnote],
            endnotes=[],
        )
        source = self.source("special-blocks.hwp")
        fake_rhwp, _ = self.rhwp_module(
            SimpleNamespace(body=[list_item, formula, toc], furniture=furniture)
        )

        result = preprocess_document(source, rhwp_module=fake_rhwp)
        list_block, formula_block, toc_block, note_block = result.blocks

        self.assertTrue(list_block["display_content"].startswith("    1. "))
        self.assertIn("1. 첫 번째 조건", list_block["retrieval_text"])
        self.assertEqual(list_block["list_marker"], "1.")
        self.assertTrue(list_block["list_enumerated"])
        self.assertEqual(list_block["list_level"], 2)
        self.assertEqual(formula_block["formula_script"], "1 over 2")
        self.assertEqual(formula_block["formula_script_kind"], "hwp_eq")
        self.assertTrue(formula_block["formula_inline"])
        self.assertIn("제 1 장 사업 개요", toc_block["retrieval_text"])
        self.assertEqual(toc_block["toc_entry_count"], 1)
        self.assertEqual(toc_block["toc_entries"][0]["cached_page"], 3)
        self.assertIn("[각주 3]", note_block["display_content"])
        self.assertEqual(note_block["note_number"], 3)
        self.assertEqual(note_block["note_marker_section_idx"], 0)
        self.assertEqual(note_block["note_marker_para_idx"], 5)

    def test_hwp_redacts_external_image_uri_and_preserves_caption_position(
        self,
    ) -> None:
        """로컬 이미지 경로는 숨기고 구조화 캡션 위치만 남긴다."""
        picture = self.picture("", uri="file:///Users/example/private.png")
        picture.caption = SimpleNamespace(
            kind="caption",
            text="",
            blocks=[self.paragraph("시스템 구성도")],
            direction="top",
            prov=SimpleNamespace(section_idx=0, para_idx=7),
        )
        source = self.source("external-image.hwp")
        fake_rhwp, _ = self.rhwp_module(SimpleNamespace(body=[picture], furniture=None))

        result = preprocess_document(source, rhwp_module=fake_rhwp)
        image = result.images[0]

        self.assertEqual(image["original_ir_uri"], "")
        self.assertTrue(image["original_ir_uri_redacted"])
        self.assertEqual(image["caption_direction"], "top")
        self.assertEqual(image["caption_section_idx"], 0)
        self.assertEqual(image["caption_para_idx"], 7)
        self.assertIn("시스템 구성도", result.blocks[0]["display_content"])
        self.assertFalse(result.document["chunking_ready"])
        self.assertIn("no_indexable_content", result.document["quality_flags"])

    def test_hwp_and_pdf_results_share_the_same_record_schema(self) -> None:
        """형식별 값은 달라도 document/block/table/image 필드명은 같아야 한다."""
        hwp_picture = self.picture()
        hwp_table = self.table(
            1,
            1,
            [self.cell(0, 0, [self.paragraph("요구사항"), hwp_picture])],
        )
        hwp_source = self.source("schema.hwp", b"hwp-schema")
        fake_rhwp, _ = self.rhwp_module(
            SimpleNamespace(body=[hwp_table], furniture=None)
        )
        hwp_result = preprocess_document(hwp_source, rhwp_module=fake_rhwp)

        class FakeTable:
            bbox = (0.0, 10.0, 100.0, 30.0)

            def extract(self) -> list[list[str]]:
                return [["요구사항"]]

        class FakePage:
            height = 100.0
            images = [
                {
                    "x0": 0.0,
                    "x1": 20.0,
                    "top": 40.0,
                    "bottom": 60.0,
                    "imagemask": False,
                }
            ]

            def find_tables(self) -> list[FakeTable]:
                return [FakeTable()]

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                return []

            def extract_text(self) -> str:
                return ""

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        pdf_source = self.source("schema.pdf", b"pdf-schema")
        pdf_result = preprocess_document(
            pdf_source,
            pdfplumber_module=SimpleNamespace(open=lambda _: FakePdf()),
        )

        self.assertEqual(set(hwp_result.document), set(pdf_result.document))
        self.assertEqual(set(hwp_result.blocks[0]), set(pdf_result.blocks[0]))
        self.assertEqual(set(hwp_result.tables[0]), set(pdf_result.tables[0]))
        self.assertEqual(set(hwp_result.images[0]), set(pdf_result.images[0]))

    def test_field_instructions_keep_links_but_exclude_controls(self) -> None:
        """HTTP 링크는 검색에 남고 계산식과 입력 제어문은 제외된다."""
        hyperlink = SimpleNamespace(
            kind="field",
            field_kind="hyperlink",
            cached_value=None,
            raw_instruction=r"https\://example.com/manual;1;5;-1;",
            blocks=[],
            prov=None,
        )
        calculation = SimpleNamespace(
            kind="field",
            field_kind="calc",
            cached_value=None,
            raw_instruction="=SUM(BELOW)",
            blocks=[],
            prov=None,
        )

        link_parts = field_parts(hyperlink)
        calculation_parts = field_parts(calculation)

        self.assertEqual(link_parts["retrieval_text"], "https://example.com/manual")
        self.assertEqual(link_parts["field_disposition"], "safe_hyperlink_target")
        self.assertEqual(calculation_parts["retrieval_text"], "")
        self.assertEqual(
            calculation_parts["field_disposition"],
            "calculation_not_indexed",
        )

    def test_pdf_preserves_order_and_removes_table_words_from_body(self) -> None:
        """PDF 본문·표·이미지 순서를 지키고 표 안 단어의 중복을 막는다."""

        class FakeTable:
            bbox = (0.0, 20.0, 100.0, 40.0)

            def extract(self) -> list[list[str]]:
                return [["요구사항", "내용"], ["REQ-1", "로그인"]]

        class FakePage:
            height = 100.0
            images = [
                {
                    "x0": 10.0,
                    "x1": 30.0,
                    "top": 60.0,
                    "bottom": 80.0,
                    "imagemask": False,
                }
            ]

            def find_tables(self) -> list[FakeTable]:
                return [FakeTable()]

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                return [
                    {"text": "표위본문", "x0": 0, "x1": 20, "top": 10, "bottom": 15},
                    {"text": "요구사항", "x0": 5, "x1": 30, "top": 25, "bottom": 30},
                    {"text": "표아래본문", "x0": 0, "x1": 30, "top": 50, "bottom": 55},
                ]

            def extract_text(self) -> str:
                return "사용되지 않는 fallback"

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        source = self.source("ordered.pdf")
        fake_pdfplumber = SimpleNamespace(open=lambda _: FakePdf())

        result = preprocess_document(
            source,
            pdfplumber_module=fake_pdfplumber,
        )

        self.assertEqual(
            [block["block_type"] for block in result.blocks],
            ["text", "table", "text", "picture"],
        )
        self.assertEqual({block["page"] for block in result.blocks}, {1})
        body_text = " ".join(
            block["retrieval_text"]
            for block in result.blocks
            if block["block_type"] == "text"
        )
        self.assertIn("표위본문", body_text)
        self.assertIn("표아래본문", body_text)
        self.assertNotIn("요구사항", body_text)
        self.assertEqual(result.tables[0]["page"], 1)
        self.assertEqual(result.images[0]["page"], 1)
        self.assertEqual(result.blocks[-1]["retrieval_text"], "")

    def test_pdf_incomplete_table_matrix_preserves_bbox_text(self) -> None:
        """표 셀 추출량이 너무 적으면 bbox 안 원문을 본문에서 삭제하지 않는다."""

        class FakeTable:
            bbox = (0.0, 20.0, 100.0, 80.0)

            def extract(self) -> list[list[str | None]]:
                return [
                    ["요구사항번호", None, "PMR-003"],
                    ["세부내용", None, None],
                ]

        preserved_words = [
            "하도급계약",
            "사전승인",
            "소프트웨어진흥법",
            "공정거래위원회",
            "하도급계획서",
            "공동수급체",
            "계약상대자",
            "대금지급",
            "발주기관",
            "하수급인",
            "적정성판단",
            "평가점수",
            "계약조건",
            "사업수행",
            "지급내역",
            "증빙자료",
            "관련법률",
            "승인절차",
        ]

        class FakePage:
            height = 100.0
            images: list[dict[str, Any]] = []

            def find_tables(self) -> list[FakeTable]:
                return [FakeTable()]

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                return [
                    {
                        "text": text,
                        "x0": 10.0,
                        "x1": 90.0,
                        "top": 25.0 + index * 3,
                        "bottom": 27.0 + index * 3,
                    }
                    for index, text in enumerate(preserved_words)
                ]

            def extract_text(self) -> str:
                return " ".join(preserved_words)

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        source = self.source("incomplete-table.pdf")
        result = preprocess_document(
            source,
            pdfplumber_module=SimpleNamespace(open=lambda _: FakePdf()),
        )

        body_text = " ".join(
            block["retrieval_text"]
            for block in result.blocks
            if block["block_type"] == "text"
        )
        self.assertIn("하도급계약", body_text)
        self.assertIn("사업수행", body_text)
        self.assertIn("pdf_table_text_fallback", result.document["quality_flags"])
        self.assertEqual(result.document["pdf_table_text_fallback_count"], 1)
        self.assertEqual(result.document["pdf_table_text_fallback_page_count"], 1)
        table_block = next(
            block for block in result.blocks if block["block_type"] == "table"
        )
        self.assertIn("pdf_table_text_fallback", table_block["quality_flags"])
        self.assertEqual(table_block["index_policy"], "exclude")
        self.assertEqual(
            table_block["index_reason"],
            "incomplete_pdf_table_replaced_by_bbox_text",
        )
        fallback_block = next(
            block
            for block in result.blocks
            if block["block_type"] == "text"
            and "pdf_table_text_fallback" in block["quality_flags"]
        )
        self.assertEqual(
            fallback_block["index_reason"],
            "incomplete_pdf_table_bbox_text",
        )

    def test_pdf_page_numbers_start_at_one(self) -> None:
        """PDF 첫 페이지와 둘째 페이지는 각각 1과 2로 기록된다."""

        class FakePage:
            height = 100.0
            images: list[dict[str, Any]] = []

            def __init__(self, text: str) -> None:
                self.text = text

            def find_tables(self) -> list[Any]:
                return []

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                return [
                    {
                        "text": self.text,
                        "x0": 10.0,
                        "x1": 50.0,
                        "top": 10.0,
                        "bottom": 20.0,
                    }
                ]

            def extract_text(self) -> str:
                return self.text

        class FakePdf:
            pages = [FakePage("첫 페이지"), FakePage("둘째 페이지")]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        source = self.source("pages.pdf")
        result = preprocess_document(
            source,
            pdfplumber_module=SimpleNamespace(open=lambda _: FakePdf()),
        )

        self.assertEqual([block["page"] for block in result.blocks], [1, 2])
        self.assertEqual(result.document["page_count"], 2)

    def test_pdf_word_failure_uses_visible_fallback_and_flag(self) -> None:
        """PDF 단어 좌표 추출 실패를 숨기지 않고 전체 텍스트와 플래그를 남긴다."""

        class FakePage:
            height = 100.0
            images: list[dict[str, Any]] = []

            def find_tables(self) -> list[Any]:
                return []

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                raise ValueError("coordinate extraction failed")

            def extract_text(self) -> str:
                return "보존해야 하는 본문"

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        source = self.source("fallback.pdf")
        result = preprocess_document(
            source,
            pdfplumber_module=SimpleNamespace(open=lambda _: FakePdf()),
        )

        self.assertEqual(result.blocks[0]["retrieval_text"], "보존해야 하는 본문")
        self.assertIn("pdf_word_extraction_failed", result.document["quality_flags"])
        self.assertEqual(result.document["pdf_word_extraction_error_count"], 1)

    def test_pdf_empty_words_with_table_keeps_fallback_text(self) -> None:
        """표가 있어도 단어 결과가 비면 전체 본문을 버리지 않고 경고한다."""

        class FakeTable:
            bbox = (0.0, 20.0, 100.0, 40.0)

            def extract(self) -> list[list[str]]:
                return [["요구사항", "내용"]]

        class FakePage:
            height = 100.0
            images: list[dict[str, Any]] = []

            def find_tables(self) -> list[FakeTable]:
                return [FakeTable()]

            def extract_words(self, **_: Any) -> list[dict[str, Any]]:
                return []

            def extract_text(self) -> str:
                return "표 밖 본문과 요구사항 내용"

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> FakePdf:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

        source = self.source("empty-words.pdf")
        result = preprocess_document(
            source,
            pdfplumber_module=SimpleNamespace(open=lambda _: FakePdf()),
        )

        self.assertEqual(result.blocks[0]["block_type"], "text")
        self.assertIn("표 밖 본문", result.blocks[0]["retrieval_text"])
        self.assertIn(
            "pdf_fallback_text_may_duplicate_table",
            result.document["quality_flags"],
        )

    def test_alternate_hwpx_keeps_original_identity(self) -> None:
        """검증된 HWPX 분석본을 써도 결과 ID와 SHA는 원본 것을 유지한다."""
        source = self.source("original.hwp", b"original")
        recovered = self.root / "recovered.hwpx"
        recovered.write_bytes(b"recovered")
        recovered_sha256 = hashlib.sha256(b"recovered").hexdigest()
        ir = SimpleNamespace(body=[self.paragraph("복구 본문")], furniture=None)
        fake_rhwp, state = self.rhwp_module(ir)

        result = preprocess_document(
            source,
            analysis_source=VerifiedAnalysisSource(
                path=recovered,
                sha256=recovered_sha256,
                original_source_id=source.source_id,
                original_source_sha256=source.source_sha256,
                verification_source="test_recovery_manifest",
            ),
            rhwp_module=fake_rhwp,
        )

        self.assertEqual(result.document["source_id"], source.source_id)
        self.assertEqual(result.document["source_sha256"], source.source_sha256)
        self.assertEqual(result.document["analysis_file_type"], "hwpx")
        self.assertFalse(result.document["analysis_source_is_original"])
        self.assertTrue(result.document["analysis_source_relationship_verified"])
        self.assertEqual(
            result.document["analysis_source_verification"],
            "test_recovery_manifest",
        )
        self.assertIn("alternate_analysis_source", result.document["quality_flags"])
        self.assertEqual(state.parse_calls, [str(recovered.resolve())])

    def test_alternate_analysis_rejects_wrong_source_relationship(self) -> None:
        """다른 원본에 속한 복구본을 현재 source_id 아래 연결하지 않는다."""
        source = self.source("original-rejected.hwp", b"original")
        recovered = self.root / "unrelated.hwpx"
        recovered.write_bytes(b"unrelated")
        fake_rhwp, state = self.rhwp_module(
            SimpleNamespace(body=[self.paragraph("다른 본문")], furniture=None)
        )
        analysis_source = VerifiedAnalysisSource(
            path=recovered,
            sha256=hashlib.sha256(b"unrelated").hexdigest(),
            original_source_id="wrong-source-id",
            original_source_sha256=source.source_sha256,
            verification_source="test_recovery_manifest",
        )

        with self.assertRaisesRegex(ValueError, "original_source_id"):
            preprocess_document(
                source,
                analysis_source=analysis_source,
                rhwp_module=fake_rhwp,
            )

        wrong_analysis_sha = replace(
            analysis_source,
            original_source_id=source.source_id,
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocess_document(
                source,
                analysis_source=wrong_analysis_sha,
                rhwp_module=fake_rhwp,
            )

        self.assertEqual(state.parse_calls, [])

    def test_batch_rejects_duplicate_source_id(self) -> None:
        """같은 원문의 파일명 별칭을 배치에서 두 번 처리하지 않는다."""
        first = self.source("first.hwp", b"same")
        second = self.source("second.hwp", b"same")
        fake_rhwp, state = self.rhwp_module(
            SimpleNamespace(body=[self.paragraph("본문")], furniture=None)
        )

        with self.assertRaisesRegex(ValueError, "같은 source_id"):
            preprocess_documents(
                [first, second],
                rhwp_module=fake_rhwp,
            )
        self.assertEqual(state.parse_calls, [])

    def test_duplicate_source_requires_explicit_canonical_selection(self) -> None:
        """중복 그룹의 임의 파일 한 개를 대표 검토 없이 처리하지 않는다."""
        source = replace(self.source("unselected.hwp"), duplicate_group_size=2)
        fake_rhwp, state = self.rhwp_module(
            SimpleNamespace(body=[self.paragraph("본문")], furniture=None)
        )

        with self.assertRaisesRegex(ValueError, "대표를 먼저 선택"):
            preprocess_document(source, rhwp_module=fake_rhwp)

        self.assertEqual(state.parse_calls, [])

    def test_payload_guard_rejects_bytes_and_data_uri(self) -> None:
        """이미지 바이트와 Base64 data URI는 결과 계약에서 거부한다."""
        self.assertTrue(has_forbidden_image_payload({"payload": b"raw"}))
        self.assertTrue(
            has_forbidden_image_payload(
                {"display_content": "data:image/png;base64,AAAA"}
            )
        )
        self.assertTrue(
            has_forbidden_image_payload(
                {"display_content": "data:image/png;charset=utf-8;base64,AAAA"}
            )
        )
        self.assertTrue(
            has_forbidden_image_payload({"display_content": "data:;base64,AAAA"})
        )
        self.assertFalse(
            has_forbidden_image_payload(
                {"picture_id": "doc:I1", "payload_stored": False}
            )
        )


if __name__ == "__main__":
    unittest.main()
