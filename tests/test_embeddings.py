"""OpenAI 호출 없이 임베딩 입력 검증과 metadata 변환을 확인한다."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import src.embeddings.build_embeddings as embedding_module


def make_row(chunk_id: str = "source-1:C000001") -> dict:
    """테스트에 필요한 최소 청크 레코드를 만든다."""

    return {
        "chunk_id": chunk_id,
        "source_id": "source-1",
        "source_filename": "sample.hwp",
        "retrieval_text": "[문서] sample.hwp\n\n검색 본문",
        "token_count": 10,
        "schema_version": "rfp_naive_chunk_v1",
        "strategy_id": "naive_recursive_tiktoken_cl100k_base_512_102_v1",
        "page_start": None,
        "quality_flags": [],
    }


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """임시 gzip JSONL 테스트 입력을 저장한다."""

    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_normalize_metadata_keeps_only_safe_scalars() -> None:
    """None·목록은 제외하고 인계용 필드명은 보존한다."""

    metadata = embedding_module.normalize_metadata(
        make_row(),
        embedding_model="text-embedding-3-small",
        create_date="2026-07-20T00:00:00+00:00",
    )

    assert metadata["chunk_id"] == "source-1:C000001"
    assert metadata["file_nm"] == "sample.hwp"
    assert metadata["embedding_model"] == "text-embedding-3-small"
    assert "page_start" not in metadata
    assert "quality_flags" not in metadata


def test_audit_input_accepts_valid_smoke_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """소량 모드에서는 고유 ID·본문·토큰 범위를 검증한다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    write_gzip_jsonl(input_path, [make_row(), make_row("source-1:C000002")])
    monkeypatch.setattr(
        embedding_module,
        "EXPECTED_INPUT_SHA256",
        embedding_module.sha256_file(input_path),
    )

    audit = embedding_module.audit_input(input_path, max_records=2)

    assert audit.chunk_count == 2
    assert audit.document_count == 1
    assert audit.total_tokens == 20


def test_audit_input_rejects_duplicate_chunk_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """같은 ID를 두 번 저장해 벡터가 덮이는 사고를 차단한다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    write_gzip_jsonl(input_path, [make_row(), make_row()])
    monkeypatch.setattr(
        embedding_module,
        "EXPECTED_INPUT_SHA256",
        embedding_module.sha256_file(input_path),
    )

    with pytest.raises(ValueError, match="중복 chunk_id"):
        embedding_module.audit_input(input_path, max_records=2)
