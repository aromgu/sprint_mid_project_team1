from __future__ import annotations

import tempfile
import unicodedata
import unittest
from pathlib import Path

from src.loader.load_documents import (
    discover_source_files,
    load_documents,
    select_canonical_documents,
    select_default_canonical_documents,
    sha256_file,
)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        """실제 RFP 대신 작은 임시 원본 파일을 만든다."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_source(self, relative_path: str, content: bytes) -> Path:
        """테스트용 HWP/PDF 모양의 파일을 원하는 하위 경로에 만든다."""
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_discover_source_files_keeps_only_supported_visible_files(self) -> None:
        """HWP/HWPX/PDF만 찾고 숨김·임시 파일은 제외한다."""
        expected = [
            self.write_source("a.hwp", b"hwp"),
            self.write_source("nested/b.HWPX", b"hwpx"),
            self.write_source("nested/c.pdf", b"pdf"),
        ]
        self.write_source("notes.txt", b"not a source document")
        self.write_source(".hidden/secret.pdf", b"hidden")

        discovered = discover_source_files(self.root)

        self.assertEqual(discovered, [path.resolve() for path in expected])

    def test_sha256_file_matches_known_value(self) -> None:
        """파일 해시는 같은 내용이면 항상 같은 값이어야 한다."""
        source = self.write_source("sample.hwp", b"abc")

        digest = sha256_file(source)

        self.assertEqual(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_load_documents_marks_exact_duplicates_without_removing_them(self) -> None:
        """같은 내용의 파일도 목록에는 남기고 같은 source_id로 묶는다."""
        self.write_source("a.hwp", b"same")
        self.write_source("b.hwp", b"same")
        self.write_source("c.pdf", b"different")

        documents = load_documents(self.root)

        self.assertEqual(len(documents), 3)
        duplicate_documents = [
            document for document in documents if document.duplicate_group_size == 2
        ]
        self.assertEqual(len(duplicate_documents), 2)
        self.assertEqual(
            len({document.source_id for document in duplicate_documents}),
            1,
        )
        self.assertEqual(
            sum(document.is_default_canonical for document in duplicate_documents),
            1,
        )
        self.assertTrue(
            all(len(document.filename_aliases) == 1 for document in duplicate_documents)
        )
        self.assertTrue(
            all(
                len(document.all_source_filenames) == 2
                for document in duplicate_documents
            )
        )

    def test_select_default_canonical_documents_returns_one_per_hash(self) -> None:
        """기본 대표만 고르면 동일 내용은 한 문서로 줄어야 한다."""
        self.write_source("a.hwp", b"same")
        self.write_source("b.hwp", b"same")
        self.write_source("c.pdf", b"different")

        documents = load_documents(self.root)
        canonical = select_default_canonical_documents(documents)

        self.assertEqual(len(canonical), 2)
        self.assertEqual(len({document.source_sha256 for document in canonical}), 2)
        duplicate_canonical = next(
            document for document in canonical if document.duplicate_group_size == 2
        )
        self.assertEqual(
            duplicate_canonical.canonical_selection_source, "loader_default"
        )
        self.assertEqual(len(duplicate_canonical.filename_aliases), 1)

    def test_select_canonical_documents_applies_team_policy(self) -> None:
        """팀이 고른 상대경로를 대표로 쓰고 다른 파일은 별칭으로 보존한다."""
        self.write_source("a.hwp", b"same")
        self.write_source("reviewed/b.hwp", b"same")
        documents = load_documents(self.root)
        source_id = documents[0].source_id

        canonical = select_canonical_documents(
            documents,
            preferred_relative_path_by_source_id={source_id: "reviewed/b.hwp"},
        )

        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0].source_relative_path, "reviewed/b.hwp")
        self.assertEqual(canonical[0].canonical_selection_source, "team_policy")
        self.assertEqual(canonical[0].filename_aliases, ("a.hwp",))

    def test_load_documents_normalizes_filename_to_nfc(self) -> None:
        """macOS의 분해형 한글 파일명도 팀이 비교하기 쉬운 NFC로 통일한다."""
        decomposed_name = unicodedata.normalize("NFD", "제안서.hwp")
        self.write_source(decomposed_name, b"content")

        document = load_documents(self.root)[0]

        self.assertEqual(document.source_filename, "제안서.hwp")
        self.assertEqual(document.source_relative_path, "제안서.hwp")

    def test_load_documents_rejects_missing_source_directory(self) -> None:
        """원본 폴더 경로가 틀리면 빈 결과 대신 이해하기 쉬운 오류를 낸다."""
        missing = self.root / "does-not-exist"

        with self.assertRaisesRegex(NotADirectoryError, "원본 문서 폴더"):
            load_documents(missing)


if __name__ == "__main__":
    unittest.main()
