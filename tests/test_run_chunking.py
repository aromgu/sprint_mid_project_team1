"""전체 청킹 CLI의 결정적 파일 입출력을 검증한다."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_chunking import (
    adapt_legacy_html_tables,
    read_jsonl,
    sha256_file,
    write_deterministic_jsonl_gzip,
)


class ChunkingCliIoTests(unittest.TestCase):
    """같은 청크 입력은 실행 시각과 무관하게 같은 gzip을 만들어야 한다."""

    def test_deterministic_gzip_round_trip(self) -> None:
        """gzip 헤더를 고정하고 한글 JSONL을 손실 없이 복원한다."""
        rows = [
            {"chunk_id": "source:C000001", "raw_text": "첫 번째 청크"},
            {"chunk_id": "source:C000002", "raw_text": "두 번째 청크"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jsonl.gz"
            second = root / "second.jsonl.gz"

            write_deterministic_jsonl_gzip(first, rows)
            write_deterministic_jsonl_gzip(second, rows)

            self.assertEqual(sha256_file(first), sha256_file(second))
            with gzip.open(first, "rt", encoding="utf-8") as stream:
                restored = [json.loads(line) for line in stream]
            self.assertEqual(restored, rows)

    def test_read_jsonl_reports_invalid_line_number(self) -> None:
        """깨진 입력은 해당 줄 번호를 포함한 오류로 중단한다."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.jsonl"
            path.write_text('{"valid":true}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"broken\.jsonl:2"):
                read_jsonl(path)

    def test_legacy_html_table_uses_retrieval_text_as_gfm(self) -> None:
        """복원용 HTML 대신 검색용 평문만 Markdown 표로 변환한다."""
        original = {
            "block_id": "source:B000001",
            "block_type": "table",
            "index_policy": "index",
            "display_content": "<table><tr><td>복원용</td></tr></table>",
            "retrieval_text": "구분 | 내용\n기간 | 30일",
            "render_mode": "html",
            "quality_flags": [],
        }

        adapted, converted_count, sanitized_break_count = adapt_legacy_html_tables(
            [original]
        )

        self.assertEqual(converted_count, 1)
        self.assertEqual(sanitized_break_count, 0)
        self.assertEqual(original["render_mode"], "html")
        self.assertNotIn("<table", adapted[0]["display_content"])
        self.assertTrue(adapted[0]["display_content"].startswith("| 내용 |"))
        self.assertEqual(adapted[0]["render_mode"], "gfm")
        self.assertIn(
            "legacy_html_table_flattened_to_gfm",
            adapted[0]["quality_flags"],
        )

    def test_gfm_inline_break_is_sanitized_without_flattening(self) -> None:
        """GFM 셀의 br 태그만 바꾸고 원래 열 구조는 유지한다."""
        original = {
            "block_id": "source:B000002",
            "block_type": "table",
            "index_policy": "index",
            "display_content": (
                "| 구분 | 내용 |\n| --- | --- |\n| 기간 | 착수일<br>종료일 |"
            ),
            "retrieval_text": "구분 | 내용\n기간 | 착수일 종료일",
            "render_mode": "gfm",
            "quality_flags": [],
        }

        adapted, converted_count, sanitized_break_count = adapt_legacy_html_tables(
            [original]
        )

        self.assertEqual(converted_count, 0)
        self.assertEqual(sanitized_break_count, 1)
        self.assertIn("| 기간 | 착수일 / 종료일 |", adapted[0]["display_content"])
        self.assertNotIn("<br>", adapted[0]["display_content"])
        self.assertIn("gfm_inline_break_sanitized", adapted[0]["quality_flags"])


if __name__ == "__main__":
    unittest.main()
