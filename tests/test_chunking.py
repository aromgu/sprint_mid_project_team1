"""구조 전처리 결과를 512/102 토큰 청크로 나누는 계약을 검증한다."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.split_text import (
    ChunkConfig,
    TiktokenCodec,
    block_chunk_text,
    boundary_key,
    build_chunk_corpus,
    build_context_prefix,
    build_recursive_character_splitter,
    build_streams,
    chunk_document,
    chunk_preprocessing_result,
    make_retrieval_text,
    parse_markdown_table_segments,
    select_indexable_blocks,
    split_text_token_ranges,
    validate_chunk_corpus,
)
from src.preprocessing.clean_text import PreprocessingResult


class CharacterCodec:
    """문자 하나를 토큰 하나로 보는 테스트 전용 결정적 코덱이다."""

    model_name = "character-test-model"
    encoding_name = "unicode-codepoint"
    version = "test-v1"

    def encode(self, text: str) -> list[int]:
        """각 유니코드 문자를 코드 포인트 토큰으로 바꾼다."""
        return [ord(char) for char in text]

    def token_bytes(self, token_id: int) -> bytes:
        """원문과 토큰의 UTF-8 경계를 검증할 바이트를 반환한다."""
        return chr(token_id).encode("utf-8")


CODEC = CharacterCodec()
CONFIG = ChunkConfig(
    max_tokens=512,
    overlap_tokens=102,
    model_name=CODEC.model_name,
    encoding_name=CODEC.encoding_name,
    strategy_id="naive_character_512_102_test",
)


def make_document(
    source_id: str = "source-001", file_type: str = "hwp"
) -> dict[str, Any]:
    """출처 메타데이터를 가진 최소 전처리 문서를 만든다."""
    extension = "pdf" if file_type == "pdf" else "hwp"
    return {
        "source_id": source_id,
        "document_id": source_id,
        "source_sha256": (source_id.encode().hex() + "0" * 64)[:64],
        "source_filename": f"{source_id}.{extension}",
        "source_relative_path": f"원본/{source_id}.{extension}",
        "filename_aliases": [f"{source_id}-별칭.{extension}"],
        "file_type": file_type,
        "chunking_ready": True,
    }


def make_block(
    order: int,
    text: str,
    *,
    source_id: str = "source-001",
    policy: str = "index",
    block_type: str = "paragraph",
    section_path: str = "Ⅰ. 사업 개요",
    section_idx: int | None = 0,
    para_idx: int | None = None,
    page: int | None = None,
    table_id: str | None = None,
    display_content: str | None = None,
    scope: str = "body",
) -> dict[str, Any]:
    """위치·검색 정책·표시 내용을 조절할 수 있는 최소 블록을 만든다."""
    return {
        "source_id": source_id,
        "document_id": source_id,
        "block_id": f"{source_id}:B{order:06d}",
        "block_order": order,
        "scope": scope,
        "furniture_type": None,
        "block_type": block_type,
        "display_content": display_content if display_content is not None else text,
        "retrieval_text": text,
        "index_policy": policy,
        "index_reason": "test_fixture",
        "section_path": section_path,
        "section_idx": section_idx,
        "para_idx": order if para_idx is None and page is None else para_idx,
        "page": page,
        "table_id": table_id,
        "picture_id": (
            f"{source_id}:body:I{order:06d}" if block_type == "picture" else None
        ),
        "render_mode": "gfm" if block_type == "table" else None,
        "quality_flags": [],
    }


class TokenRangeTests(unittest.TestCase):
    """LangChain 재귀 분할과 기존 512/102 품질 계약을 검증한다."""

    def test_uses_langchain_recursive_character_text_splitter(self) -> None:
        """운영 경계 선택기가 요청된 LangChain 클래스를 실제로 만든다."""
        splitter = build_recursive_character_splitter(
            CODEC,
            chunk_size=410,
            overlap_tokens=102,
        )

        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)
        self.assertGreater(len(splitter.split_text("가" * 900)), 1)

    def test_context_included_limit_and_exact_overlap(self) -> None:
        """문서 문맥을 포함해 512 이하이고 인접 원문은 102토큰 겹친다."""
        text = "가" * 1_600
        prefix = "[문서] 테스트.hwp\n[위치] Ⅰ. 사업 개요\n[유형] 본문"

        token_map, ranges = split_text_token_ranges(text, prefix, CODEC, CONFIG)

        self.assertGreater(len(ranges), 2)
        for start, end, _ in ranges:
            retrieval = make_retrieval_text(prefix, token_map.slice(start, end))
            self.assertLessEqual(len(CODEC.encode(retrieval)), 512)
        for previous, current in zip(ranges, ranges[1:]):
            self.assertEqual(current[2], 102)
            self.assertEqual(current[0], previous[1] - 102)

    def test_recursive_split_keeps_source_whitespace(self) -> None:
        """문단 경계의 공백을 지우지 않아 저장된 원문 좌표가 유지된다."""
        document = make_document()
        text = "\n\n".join(f"문단{index} " + "가" * 180 for index in range(10))
        chunks = chunk_document(document, [make_block(1, text)], CODEC, CONFIG)

        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            start = chunk["stream_token_start"]
            end = chunk["stream_token_end"]
            self.assertEqual(chunk["raw_text"], text[start:end])


class BlockSelectionAndBoundaryTests(unittest.TestCase):
    """검색 정책과 HWP/PDF 위치 경계를 검증한다."""

    def test_exclude_and_image_placeholder_never_enter_chunks(self) -> None:
        """짧은 본문은 보존하고 이미지 블록과 placeholder는 제외한다."""
        document = make_document()
        blocks = [
            make_block(1, "앞 본문 ![구조도](image://source-001:I1) 뒤 본문"),
            make_block(2, "★"),
            make_block(
                3,
                "image://source-001:I2",
                policy="exclude",
                block_type="picture",
            ),
        ]

        selected = select_indexable_blocks(blocks)
        chunks = chunk_document(document, blocks, CODEC, CONFIG)
        covered = {
            block_id for chunk in chunks for block_id in chunk["source_block_ids"]
        }

        self.assertEqual(selected, blocks[:2])
        self.assertIn("★", "\n".join(chunk["raw_text"] for chunk in chunks))
        self.assertNotIn(blocks[2]["block_id"], covered)
        self.assertTrue(
            all("image://" not in chunk["retrieval_text"] for chunk in chunks)
        )

    def test_hwp_section_change_starts_new_stream(self) -> None:
        """HWP 섹션이 바뀌면 overlap 없는 별도 청크가 된다."""
        document = make_document(file_type="hwp")
        first = make_block(1, "첫 섹션", section_path="Ⅰ. 개요", section_idx=0)
        second = make_block(2, "둘째 섹션", section_path="Ⅱ. 요구", section_idx=0)

        self.assertNotEqual(
            boundary_key(document, first), boundary_key(document, second)
        )
        self.assertEqual(len(build_streams(document, [first, second])), 2)
        chunks = chunk_document(document, [first, second], CODEC, CONFIG)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(
            all(chunk["overlap_from_previous_tokens"] == 0 for chunk in chunks)
        )

    def test_pdf_page_change_starts_new_stream(self) -> None:
        """PDF 페이지가 바뀌면 이전 페이지의 overlap을 넘기지 않는다."""
        document = make_document(file_type="pdf")
        first = make_block(
            1,
            "첫 페이지",
            page=1,
            section_idx=None,
            para_idx=None,
        )
        second = make_block(
            2,
            "둘째 페이지",
            page=2,
            section_idx=None,
            para_idx=None,
        )

        chunks = chunk_document(document, [first, second], CODEC, CONFIG)

        self.assertEqual(
            [(row["page_start"], row["page_end"]) for row in chunks], [(1, 1), (2, 2)]
        )
        self.assertTrue(all(row["overlap_from_previous_tokens"] == 0 for row in chunks))


class MarkdownTableTests(unittest.TestCase):
    """표가 HTML 없이 독립 Markdown 청크로 유지되는지 검증한다."""

    @staticmethod
    def table_block(
        markdown: str, *, order: int = 1, policy: str = "index"
    ) -> dict[str, Any]:
        """표시 Markdown과 검색 평문을 함께 가진 표 블록을 만든다."""
        return make_block(
            order,
            "요구사항 ID | 설명\nREQ-001 | 로그인",
            policy=policy,
            block_type="table",
            table_id=f"source-001:body:T{order:06d}",
            display_content=markdown,
        )

    def test_small_table_is_one_standalone_markdown_chunk(self) -> None:
        """작은 표는 주변 본문과 섞이지 않고 Markdown 표 한 개가 된다."""
        markdown = "| 요구사항 ID | 설명 |\n| --- | --- |\n| REQ-001 | 로그인 |"
        table = self.table_block(markdown, order=2)
        blocks = [make_block(1, "표 앞"), table, make_block(3, "표 뒤")]

        chunks = chunk_document(make_document(), blocks, CODEC, CONFIG)
        table_chunks = [row for row in chunks if row["content_type"] == "table"]

        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0]["raw_text"], markdown)
        self.assertEqual(table_chunks[0]["table_id"], table["table_id"])
        self.assertEqual(table_chunks[0]["render_mode"], "gfm")
        self.assertNotIn("<table", table_chunks[0]["raw_text"])

    def test_large_table_splits_on_rows_and_repeats_markdown_header(self) -> None:
        """큰 표의 각 part는 GFM 헤더·구분 행을 반복하고 행을 보존한다."""
        header = "| 요구사항 ID | 설명 |"
        separator = "| --- | --- |"
        rows = [f"| REQ-{index:02d} | {'가' * 90} |" for index in range(14)]
        table = self.table_block("\n".join([header, separator, *rows]))

        chunks = chunk_document(make_document(), [table], CODEC, CONFIG)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["token_count"] <= 512 for row in chunks))
        self.assertTrue(
            all(
                row["raw_text"].splitlines()[:2] == [header, separator]
                for row in chunks
            )
        )
        self.assertFalse(chunks[0]["table_header_repeated"])
        self.assertTrue(all(row["table_header_repeated"] for row in chunks[1:]))
        emitted_rows = {
            line for chunk in chunks for line in chunk["raw_text"].splitlines()[2:]
        }
        self.assertEqual(emitted_rows, set(rows))

    def test_nested_tables_become_separate_markdown_segments(self) -> None:
        """중첩 표는 같은 table_id 아래 별도 Markdown segment로 한 번씩 나온다."""
        markdown = (
            "| 바깥 | 값 |\n"
            "| --- | --- |\n"
            "| A | B |\n\n"
            "**중첩 표 `inner`**\n\n"
            "| 안쪽 | 값 |\n"
            "| --- | --- |\n"
            "| X | Y ![그림](image://inner:I1) |"
        )
        table = self.table_block(markdown)

        segments = parse_markdown_table_segments(block_chunk_text(table))
        chunks = chunk_document(make_document(), [table], CODEC, CONFIG)

        self.assertEqual(len(segments), 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([row["table_segment_index"] for row in chunks], [1, 2])
        self.assertTrue(all(row["table_segment_count"] == 2 for row in chunks))
        self.assertTrue(all("image://" not in row["retrieval_text"] for row in chunks))
        self.assertIn("**중첩 표 `inner`**", chunks[1]["raw_text"])

    def test_html_table_is_rejected(self) -> None:
        """Naive RAG 입력 표에 HTML이 다시 들어오면 즉시 실패한다."""
        table = self.table_block("<table><tr><td>값</td></tr></table>")

        with self.assertRaisesRegex(ValueError, "HTML"):
            chunk_document(make_document(), [table], CODEC, CONFIG)

    def test_oversized_single_header_falls_back_to_markdown_cells(self) -> None:
        """긴 병합 헤더도 버리지 않고 유효한 한 셀 Markdown 표로 나눈다."""
        table = self.table_block(f"| [병합 1행×1열] {'가' * 1_100} |\n| --- |")

        chunks = chunk_document(make_document(), [table], CODEC, CONFIG)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["token_count"] <= 512 for row in chunks))
        self.assertTrue(
            all(row["raw_text"].startswith("| 표 내용 |\n| --- |") for row in chunks)
        )
        self.assertTrue(
            all(
                "oversized_table_header_split" in row["quality_flags"] for row in chunks
            )
        )
        self.assertTrue(all("<table" not in row["raw_text"] for row in chunks))

    def test_nearly_full_header_and_row_fall_back_without_data_loss(self) -> None:
        """헤더가 예산을 소진해도 다음 행을 버리지 않고 한 셀 표로 나눈다."""
        document = make_document()
        probe_table = self.table_block("|  |\n| --- |\n| 데이터 |")
        prefix = build_context_prefix(document, probe_table, "table")
        base_header = "|  |\n| --- |"
        fill_length = (
            CONFIG.max_tokens
            - 1
            - len(CODEC.encode(make_retrieval_text(prefix, base_header)))
        )
        markdown = f"| {'가' * fill_length} |\n| --- |\n| 데이터 |"
        table = self.table_block(markdown)

        chunks = chunk_document(document, [table], CODEC, CONFIG)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["token_count"] <= 512 for row in chunks))
        self.assertTrue(
            all(
                "table_header_budget_exhausted_fallback" in row["quality_flags"]
                for row in chunks
            )
        )
        self.assertEqual(
            sum(row["raw_text"].count("가") for row in chunks), fill_length
        )
        self.assertIn("데이터", "\n".join(row["raw_text"] for row in chunks))


class CorpusContractTests(unittest.TestCase):
    """결정적 ID·공개 API·완료 게이트를 검증한다."""

    def test_chunk_ids_do_not_depend_on_input_order(self) -> None:
        """문서와 블록 입력 순서를 바꿔도 청크와 ID가 동일하다."""
        documents = [make_document("source-b"), make_document("source-a")]
        blocks = [
            make_block(2, "두 번째", source_id="source-a"),
            make_block(1, "첫 번째", source_id="source-a"),
            make_block(1, "다른 문서", source_id="source-b"),
        ]

        first = build_chunk_corpus(documents, blocks, CODEC, CONFIG)
        second = build_chunk_corpus(
            list(reversed(documents)),
            list(reversed(blocks)),
            CODEC,
            CONFIG,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [row["chunk_id"] for row in first],
            [
                "source-a:N512O102:C000001",
                "source-b:N512O102:C000001",
            ],
        )

    def test_public_api_returns_chunks_and_summary_without_mutation(self) -> None:
        """전처리 결과를 바꾸지 않고 청크와 품질 요약을 반환한다."""
        document = make_document()
        block = make_block(1, "검색 가능한 본문")
        preprocessing = PreprocessingResult(
            document=document,
            blocks=(block,),
            tables=(),
            images=(),
        )
        before = copy.deepcopy(preprocessing)

        result = chunk_preprocessing_result(
            preprocessing,
            codec=CODEC,
            config=CONFIG,
        )

        self.assertEqual(preprocessing, before)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.summary["chunk_count"], 1)
        self.assertTrue(result.summary["validation"]["overall_pass"])
        self.assertNotIn("source_path", result.chunks[0])

    def test_validator_detects_tampered_token_count(self) -> None:
        """저장된 토큰 수가 실제 문자열과 다르면 완료 게이트가 실패한다."""
        document = make_document()
        block = make_block(1, "검증할 본문")
        chunks = chunk_document(document, [block], CODEC, CONFIG)
        tampered = copy.deepcopy(chunks)
        tampered[0]["token_count"] = 999

        validation = validate_chunk_corpus(
            [document],
            [block],
            tampered,
            CODEC,
            CONFIG,
        )

        self.assertFalse(validation["overall_pass"])
        self.assertFalse(validation["gates"]["token_counts_and_limit_are_valid"])

    def test_invalid_config_is_rejected(self) -> None:
        """0토큰 또는 크기 이상의 overlap 설정은 거부한다."""
        invalid = [
            ChunkConfig(max_tokens=0, overlap_tokens=0),
            ChunkConfig(max_tokens=512, overlap_tokens=512),
            ChunkConfig(max_tokens=512, overlap_tokens=-1),
        ]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_chunk_corpus([], [], CODEC, config)

    def test_tiktoken_codec_class_has_expected_defaults(self) -> None:
        """운영 코덱의 기본 모델과 인코딩 계약을 문서화한다."""
        signature = SimpleNamespace(
            model=ChunkConfig().model_name,
            encoding=ChunkConfig().encoding_name,
        )
        self.assertEqual(signature.model, "text-embedding-3-small")
        self.assertEqual(signature.encoding, "cl100k_base")
        self.assertTrue(callable(TiktokenCodec))


if __name__ == "__main__":
    unittest.main()
