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


def register_test_contract(
    input_path: Path,
    rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> embedding_module.InputContract:
    """임시 입력 SHA를 운영과 같은 계약 검사 경로에 등록한다."""

    contract = embedding_module.InputContract(
        name="test_contract",
        input_sha256=embedding_module.sha256_file(input_path),
        chunk_count=len(rows),
        document_count=len({row["source_id"] for row in rows}),
        total_tokens=sum(row["token_count"] for row in rows),
        schema_version="rfp_naive_chunk_v1",
        strategy_id="naive_recursive_tiktoken_cl100k_base_512_102_v1",
    )
    monkeypatch.setattr(
        embedding_module,
        "INPUT_CONTRACTS_BY_SHA256",
        {contract.input_sha256: contract},
    )
    return contract


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


def test_normalize_metadata_keeps_enriched_filters_and_aliases() -> None:
    """사업 필터와 중복 파일 별칭을 Chroma 호환 scalar로 바꾼다."""

    row = make_row()
    row.update(
        {
            "project_name": "통합정보시스템 고도화",
            "project_summary": "- 사업의 주요 범위와 목표",
            "issuer": "발주기관",
            "project_amount_status": "usable",
            "project_summary_review_status": "ai_source_review_pass",
            "metadata_schema_version": "business_metadata_v1",
            "filename_aliases": ["과거 파일명.hwp"],
        }
    )

    metadata = embedding_module.normalize_metadata(
        row,
        embedding_model="text-embedding-3-small",
        create_date="2026-07-23T00:00:00+00:00",
    )

    assert metadata["project_name"] == "통합정보시스템 고도화"
    assert metadata["project_summary"] == "- 사업의 주요 범위와 목표"
    assert metadata["issuer"] == "발주기관"
    assert metadata["metadata_schema_version"] == "business_metadata_v1"
    assert metadata["filename_aliases"] == '["과거 파일명.hwp"]'
    assert metadata["filename_alias_count"] == 1


def test_audit_input_accepts_valid_smoke_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """소량 모드에서는 고유 ID·본문·토큰 범위를 검증한다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    rows = [make_row(), make_row("source-1:C000002")]
    write_gzip_jsonl(input_path, rows)
    register_test_contract(input_path, rows, monkeypatch)

    audit = embedding_module.audit_input(input_path, max_records=2)

    assert audit.chunk_count == 2
    assert audit.document_count == 1
    assert audit.total_tokens == 20


def test_audit_input_rejects_duplicate_chunk_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """같은 ID를 두 번 저장해 벡터가 덮이는 사고를 차단한다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    rows = [make_row(), make_row()]
    write_gzip_jsonl(input_path, rows)
    register_test_contract(input_path, rows, monkeypatch)

    with pytest.raises(ValueError, match="중복 chunk_id"):
        embedding_module.audit_input(input_path, max_records=2)


def test_audit_input_rejects_unknown_file_sha(tmp_path: Path) -> None:
    """등록하지 않은 청크 파일은 소량 실행도 API 호출 전에 거부한다."""

    input_path = tmp_path / "unknown.jsonl.gz"
    write_gzip_jsonl(input_path, [make_row()])

    with pytest.raises(ValueError, match="승인되지 않은"):
        embedding_module.audit_input(input_path, max_records=1)


def test_full_audit_uses_selected_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """전체 검증은 선택된 파일 계약의 건수·문서·토큰을 사용한다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    rows = [make_row(), make_row("source-1:C000002")]
    write_gzip_jsonl(input_path, rows)
    contract = register_test_contract(input_path, rows, monkeypatch)

    audit = embedding_module.audit_input(input_path)

    assert audit.contract_name == contract.name
    assert audit.chunk_count == contract.chunk_count
    assert audit.total_tokens == contract.total_tokens


def test_rcts_v2_contract_matches_generated_corpus() -> None:
    """운영 RCTS v2 청크의 승인된 고정값이 실수로 바뀌지 않게 한다."""

    contract = embedding_module.RCTS_V2_INPUT_CONTRACT

    assert contract.chunk_count == 31_451
    assert contract.document_count == 98
    assert contract.total_tokens == 10_316_103
    assert contract.strategy_id == "naive_langchain_recursive_cl100k_base_512_102_v2"


def test_rcts_v3_contract_matches_generated_corpus() -> None:
    """병합표 보정과 PDF 원문 복구를 반영한 RCTS v3 계약을 고정한다."""

    contract = embedding_module.RCTS_V3_INPUT_CONTRACT

    assert contract.input_sha256 == (
        "8d5107140ff20c5f78fa3b3a88c06a2149a1a31397a22e8fb1ca6cd32f3f7c09"
    )
    assert contract.chunk_count == 31_627
    assert contract.document_count == 98
    assert contract.total_tokens == 10_414_025
    assert contract.strategy_id == "naive_langchain_recursive_cl100k_base_512_102_v3"


def test_rcts_v3_metadata_v1_contract_matches_enriched_corpus() -> None:
    """사업 메타데이터 보강이 끝난 최종 GCP 임베딩 입력을 고정한다."""

    contract = embedding_module.RCTS_V3_METADATA_V1_INPUT_CONTRACT

    assert contract.input_sha256 == (
        "4c77826f4705f8df70dfa15d180312ec133d624ab25ace85cfd32f0c9f8f9194"
    )
    assert contract.chunk_count == 31_627
    assert contract.document_count == 98
    assert contract.total_tokens == 10_414_025
    assert contract.strategy_id == "naive_langchain_recursive_cl100k_base_512_102_v3"
    assert contract.metadata_schema_version == "business_metadata_v1"


def test_audit_rejects_wrong_metadata_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """보강 계약 파일에 다른 metadata schema가 섞이면 API 호출 전에 막는다."""

    input_path = tmp_path / "chunks.jsonl.gz"
    rows = [make_row()]
    rows[0]["metadata_schema_version"] = "wrong_metadata_v1"
    write_gzip_jsonl(input_path, rows)
    contract = embedding_module.InputContract(
        name="enriched_test_contract",
        input_sha256=embedding_module.sha256_file(input_path),
        chunk_count=1,
        document_count=1,
        total_tokens=10,
        schema_version="rfp_naive_chunk_v1",
        strategy_id="naive_recursive_tiktoken_cl100k_base_512_102_v1",
        metadata_schema_version="business_metadata_v1",
    )
    monkeypatch.setattr(
        embedding_module,
        "INPUT_CONTRACTS_BY_SHA256",
        {contract.input_sha256: contract},
    )

    with pytest.raises(ValueError, match="metadata_schema_version"):
        embedding_module.audit_input(input_path)


def test_collection_contract_rejects_different_input_sha() -> None:
    """같은 컬렉션 이름에 다른 청크 벡터를 섞지 못하게 한다."""

    audit = embedding_module.InputAudit(
        input_sha256="new-sha",
        contract_name="test_contract",
        chunk_count=1,
        document_count=1,
        total_tokens=10,
        chunk_ids=frozenset({"source-1:C000001"}),
        schema_versions=("rfp_naive_chunk_v1",),
        strategy_ids=("naive_strategy_v2",),
        metadata_schema_versions=("business_metadata_v1",),
    )
    metadata = {
        "embedding_model": "text-embedding-3-small",
        "schema_version": "rfp_naive_chunk_v1",
        "strategy_id": "naive_strategy_v2",
        "metadata_schema_version": "business_metadata_v1",
        "input_sha256": "old-sha",
    }

    with pytest.raises(RuntimeError, match="Collection 계약"):
        embedding_module.validate_collection_contract(
            metadata,
            audit=audit,
            embedding_model="text-embedding-3-small",
        )
