"""외부 임베딩 전 개인정보 마스킹 계약을 검증한다."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.embeddings.redact_sensitive_text import (
    PRIVACY_SCHEMA_VERSION,
    redact_embedding_corpus,
    redact_text,
    sha256_file,
)


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """테스트용 gzip JSONL을 저장한다."""

    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_redact_text_masks_sensitive_values_and_keeps_business_number() -> None:
    """개인정보·보안값은 가리고 공고번호 같은 업무 식별자는 보존한다."""

    source = (
        "담당자: 홍길동, email@example.com, 010-1234-5678, "
        "주민등록번호 900101-1234567, 서버 192.168.0.10, "
        "공고번호 20240637286"
    )

    redacted, counts = redact_text(source)

    assert "홍길동" not in redacted
    assert "email@example.com" not in redacted
    assert "010-1234-5678" not in redacted
    assert "900101-1234567" not in redacted
    assert "192.168.0.10" not in redacted
    assert "20240637286" in redacted
    assert counts["person_name_labeled"] == 1
    assert counts["email"] == 1
    assert counts["phone"] == 1
    assert counts["resident_or_foreign_id"] == 1
    assert counts["ipv4"] == 1


def test_redaction_adds_embedding_text_without_changing_source_text(
    tmp_path: Path,
) -> None:
    """원문 retrieval_text는 유지하고 외부 전송용 필드만 추가한다."""

    input_path = tmp_path / "input.jsonl.gz"
    output_path = tmp_path / "output.jsonl.gz"
    report_path = tmp_path / "report.json"
    source_text = "[문서] sample.hwp\n\n담당자: 홍길동 02-1234-5678"
    write_gzip_jsonl(
        input_path,
        [
            {
                "chunk_id": "source-1:C000001",
                "source_id": "source-1",
                "document_id": "source-1",
                "retrieval_text": source_text,
                "project_summary": "- 문의: sample@example.com",
            }
        ],
    )

    report = redact_embedding_corpus(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
    )

    with gzip.open(output_path, "rt", encoding="utf-8") as file_obj:
        row = json.loads(next(file_obj))

    assert row["retrieval_text"] == source_text
    assert "홍길동" not in row["embedding_text"]
    assert "02-1234-5678" not in row["embedding_text"]
    assert "sample@example.com" not in row["project_summary"]
    assert row["privacy_schema_version"] == PRIVACY_SCHEMA_VERSION
    assert row["embedding_token_count"] > 0
    assert report.changed_chunk_count == 1
    assert report.redaction_occurrence_count == 3


def test_redaction_output_is_deterministic(tmp_path: Path) -> None:
    """같은 입력은 실행 시각과 무관하게 같은 gzip SHA를 만든다."""

    input_path = tmp_path / "input.jsonl.gz"
    first_output = tmp_path / "first.jsonl.gz"
    second_output = tmp_path / "second.jsonl.gz"
    write_gzip_jsonl(
        input_path,
        [
            {
                "chunk_id": "source-1:C000001",
                "source_id": "source-1",
                "document_id": "source-1",
                "retrieval_text": "문의 sample@example.com",
            }
        ],
    )

    redact_embedding_corpus(
        input_path=input_path,
        output_path=first_output,
        report_path=tmp_path / "first.json",
    )
    redact_embedding_corpus(
        input_path=input_path,
        output_path=second_output,
        report_path=tmp_path / "second.json",
    )

    assert sha256_file(first_output) == sha256_file(second_output)
