"""Advanced Dense·BM25 인덱싱 계약을 외부 API 호출 없이 검증한다."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import src.embeddings.build_advanced_index as advanced_module


def make_text_row(
    chunk_id: str = "source-1:ADV2:A512O51:C000001",
    *,
    source_id: str = "source-1",
) -> dict[str, Any]:
    """KSS·Kiwi 처리가 끝난 일반 텍스트 테스트 청크를 만든다."""

    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "document_id": source_id,
        "source_filename": "sample.hwp",
        "source_relative_path": "files/sample.hwp",
        "file_type": "hwp",
        "project_name": "테스트 사업",
        "issuer": "테스트 기관",
        "notice_number": "2026-01",
        "notice_round": "1",
        "published_at": "2026-07-01",
        "bid_start_at": "2026-07-02",
        "bid_end_at": "2026-07-10",
        "project_amount_won": 100_000_000,
        "content_type": "text",
        "raw_text": "사업 수행 기간과 주요 과업입니다.",
        "embedding_text": "사업 수행 기간과 주요 과업입니다.",
        "token_count": 9,
        "token_count_basis": "embedding_text",
        "vectorize_field": "embedding_text",
        "embedding_prefix_included": False,
        "schema_version": "test_advanced_schema",
        "strategy_id": "test_advanced_strategy",
        "corpus_id": "test_advanced_corpus",
        "source_sha256": "source-sha",
        "page_start": None,
        "page_end": None,
        "section_path": "사업 개요",
        "section_idx_start": 1,
        "section_idx_end": 1,
        "para_idx_start": 10,
        "para_idx_end": 11,
        "chunk_order": 1,
        "table_id": None,
        "kss_applied": True,
        "bm25_eligible": True,
        "bm25_tokens": ["사업", "수행", "기간", "주요", "과업"],
        "bm25_token_count": 5,
    }


def make_table_row(
    chunk_id: str = "source-2:ADV2:A512O51:C000001",
    *,
    source_id: str = "source-2",
) -> dict[str, Any]:
    """Markdown만 Dense 벡터화하는 표 테스트 청크를 만든다."""

    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "document_id": source_id,
        "source_filename": "sample.pdf",
        "source_relative_path": "files/sample.pdf",
        "file_type": "pdf",
        "project_name": "표 테스트 사업",
        "issuer": "표 테스트 기관",
        "content_type": "table",
        "raw_text": "| 구분 | 설명 |\n| --- | --- |\n| 기간 | 3개월 |",
        "embedding_text": "| 구분 | 설명 |\n| --- | --- |\n| 기간 | 3개월 |",
        "token_count": 20,
        "token_count_basis": "embedding_text",
        "vectorize_field": "embedding_text",
        "embedding_prefix_included": False,
        "schema_version": "test_advanced_schema",
        "strategy_id": "test_advanced_strategy",
        "corpus_id": "test_advanced_corpus",
        "source_sha256": "source-2-sha",
        "page_start": 3,
        "page_end": 3,
        "section_path": "사업 범위",
        "section_idx_start": 2,
        "section_idx_end": 2,
        "para_idx_start": None,
        "para_idx_end": None,
        "chunk_order": 1,
        "table_id": "source-2:T000001",
        "table_part_index": 1,
        "table_part_count": 1,
        "table_segment_index": 1,
        "table_segment_count": 1,
        "render_mode": "markdown",
        "kss_applied": False,
        "bm25_eligible": False,
        "bm25_tokens": [],
        "bm25_token_count": 0,
    }


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """운영 파일과 같은 gzip JSONL 테스트 입력을 저장한다."""

    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_test_contract(
    input_path: Path,
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> advanced_module.AdvancedInputContract:
    """임시 입력 SHA와 실제 통계로 운영과 같은 승인 계약을 등록한다."""

    text_rows = [row for row in rows if row["content_type"] == "text"]
    table_rows = [row for row in rows if row["content_type"] == "table"]
    contract = advanced_module.AdvancedInputContract(
        name="test_advanced_contract",
        input_sha256=advanced_module.sha256_file(input_path),
        chunk_count=len(rows),
        document_count=len({row["source_id"] for row in rows}),
        total_tokens=sum(row["token_count"] for row in rows),
        text_chunk_count=len(text_rows),
        table_chunk_count=len(table_rows),
        bm25_chunk_count=len(text_rows),
        bm25_token_total=sum(row["bm25_token_count"] for row in text_rows),
        schema_version="test_advanced_schema",
        strategy_id="test_advanced_strategy",
        corpus_id="test_advanced_corpus",
    )
    monkeypatch.setattr(
        advanced_module,
        "INPUT_CONTRACTS_BY_SHA256",
        {contract.input_sha256: contract},
    )
    return contract


def make_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, list[dict[str, Any]], advanced_module.AdvancedInputAudit]:
    """테스트 입력을 만들고 전체 계약 검증 결과까지 반환한다."""

    selected_rows = rows or [make_text_row(), make_table_row()]
    input_path = tmp_path / "chunks_advanced.jsonl.gz"
    write_gzip_jsonl(input_path, selected_rows)
    register_test_contract(input_path, selected_rows, monkeypatch)
    return (
        input_path,
        selected_rows,
        advanced_module.audit_advanced_input(input_path),
    )


def test_final_advanced_v2_contract_is_fixed() -> None:
    """팀이 승인한 v4 청크의 SHA·건수·전략이 실수로 바뀌지 않게 한다."""

    contract = advanced_module.ADVANCED_V2_INPUT_CONTRACT

    assert contract.chunk_count == 82_442
    assert contract.document_count == 98
    assert contract.text_chunk_count == 41_830
    assert contract.table_chunk_count == 40_612
    assert contract.bm25_token_total == 797_788
    assert contract.corpus_id == "advanced_v2"


def test_audit_accepts_text_and_table_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense 전체와 텍스트 전용 BM25 통계를 각각 계산한다."""

    _, _, audit = make_input(tmp_path, monkeypatch)

    assert audit.chunk_count == 2
    assert audit.document_count == 2
    assert audit.text_chunk_count == 1
    assert audit.table_chunk_count == 1
    assert audit.bm25_chunk_count == 1
    assert audit.bm25_token_total == 5


def test_audit_rejects_unknown_sha(tmp_path: Path) -> None:
    """승인되지 않은 파일은 API 호출 전에 차단한다."""

    input_path = tmp_path / "unknown.jsonl.gz"
    write_gzip_jsonl(input_path, [make_text_row()])

    with pytest.raises(ValueError, match="승인되지 않은"):
        advanced_module.audit_advanced_input(input_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("embedding_prefix_included", True, "임베딩 접두사"),
        (
            "embedding_text",
            "첫 문장입니다.\n둘째 문장입니다.",
            "줄바꿈",
        ),
    ],
)
def test_audit_rejects_invalid_text_embedding_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    """메타 접두사·줄바꿈이 일반 텍스트 벡터 본문에 들어가지 않게 한다."""

    row = make_text_row()
    row[field] = value
    input_path = tmp_path / "invalid-text.jsonl.gz"
    write_gzip_jsonl(input_path, [row])
    register_test_contract(input_path, [row], monkeypatch)

    with pytest.raises(ValueError, match=message):
        advanced_module.audit_advanced_input(input_path)


def test_audit_rejects_table_bm25_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """표가 KSS·Kiwi/BM25 경로에 잘못 섞이면 검증에 실패한다."""

    row = make_table_row()
    row["bm25_eligible"] = True
    row["bm25_tokens"] = ["표"]
    row["bm25_token_count"] = 1
    input_path = tmp_path / "invalid-table.jsonl.gz"
    write_gzip_jsonl(input_path, [row])
    register_test_contract(input_path, [row], monkeypatch)

    with pytest.raises(ValueError, match="표는 BM25"):
        advanced_module.audit_advanced_input(input_path)


def test_dense_batches_use_embedding_text_for_text_and_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """일반 텍스트와 Markdown 표 모두 embedding_text로 Dense 입력을 만든다."""

    input_path, rows, _ = make_input(tmp_path, monkeypatch)

    batches = list(
        advanced_module.iter_dense_batches(
            input_path,
            batch_size=2,
            embedding_model="text-embedding-3-small",
            create_date="2026-07-23T00:00:00+00:00",
        )
    )

    assert [record.embedding_text for record in batches[0]] == [
        row["embedding_text"] for row in rows
    ]
    assert batches[0][0].metadata["para_idx_start"] == 10
    assert batches[0][1].metadata["page_start"] == 3
    assert batches[0][1].metadata["table_id"] == "source-2:T000001"
    assert batches[0][0].metadata["file_nm"] == "sample.hwp"


def test_dense_collection_rejects_different_input() -> None:
    """같은 이름의 컬렉션에 다른 Advanced 청크가 섞이지 않게 한다."""

    audit = advanced_module.AdvancedInputAudit(
        input_sha256="new-sha",
        contract_name="test",
        chunk_count=1,
        document_count=1,
        total_tokens=9,
        text_chunk_count=1,
        table_chunk_count=0,
        bm25_chunk_count=1,
        bm25_token_total=5,
        chunk_ids=frozenset({"chunk-1"}),
        schema_versions=("schema",),
        strategy_ids=("strategy",),
        corpus_ids=("corpus",),
    )
    metadata = {
        "index_kind": "advanced_dense_v2",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "schema_version": "schema",
        "strategy_id": "strategy",
        "corpus_id": "corpus",
        "input_sha256": "old-sha",
        "document_field": "embedding_text",
    }

    with pytest.raises(RuntimeError, match="Advanced Collection 계약"):
        advanced_module.validate_dense_collection_contract(
            metadata,
            audit=audit,
            embedding_model="text-embedding-3-small",
        )


class FakeCollection:
    """OpenAI·Chroma 없이 resume와 저장 내용을 확인하는 가짜 컬렉션이다."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """ID별 Dense 저장 내용을 메모리에 보관한다."""

        for chunk_id, vector, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            self.records[chunk_id] = {
                "embedding": vector,
                "document": document,
                "metadata": metadata,
            }

    def count(self) -> int:
        """현재 저장된 가짜 벡터 개수를 반환한다."""

        return len(self.records)


class FakeVectorstore:
    """build_dense_index가 쓰는 Chroma 최소 인터페이스를 흉내 낸다."""

    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def get(self, *, include: list[str]) -> dict[str, list[str]]:
        """기존 ID를 반환해 재실행 시 API 호출을 건너뛰게 한다."""

        assert include == []
        return {"ids": list(self._collection.records)}


class FakeEmbeddings:
    """호출된 본문을 기록하고 1536차원 가짜 벡터를 반환한다."""

    calls: list[list[str]] = []

    def __init__(self, **_: Any) -> None:
        self.__class__.calls = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """OpenAI 호출 대신 입력별 고정 차원 벡터를 만든다."""

        self.__class__.calls.append(list(texts))
        return [[float(index)] * 1536 for index, _ in enumerate(texts, start=1)]


def test_dense_index_embeds_both_types_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """첫 실행은 두 유형을 저장하고 재실행은 기존 ID를 모두 건너뛴다."""

    input_path, rows, audit = make_input(tmp_path, monkeypatch)
    expected_metadata = {
        "index_kind": "advanced_dense_v2",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "corpus_id": audit.corpus_ids[0],
        "input_sha256": audit.input_sha256,
        "document_field": "embedding_text",
    }
    collection = FakeCollection(expected_metadata)
    vectorstore = FakeVectorstore(collection)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(advanced_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(advanced_module, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(
        advanced_module,
        "create_advanced_vectorstore",
        lambda **_: vectorstore,
    )

    first = advanced_module.build_dense_index(
        input_path=input_path,
        audit=audit,
        persist_directory=tmp_path / "chroma",
        collection_name="advanced-test",
        embedding_model="text-embedding-3-small",
        batch_size=2,
        max_records=None,
    )
    second = advanced_module.build_dense_index(
        input_path=input_path,
        audit=audit,
        persist_directory=tmp_path / "chroma",
        collection_name="advanced-test",
        embedding_model="text-embedding-3-small",
        batch_size=2,
        max_records=None,
    )

    assert first.embedded_count == 2
    assert first.skipped_existing_count == 0
    assert first.final_collection_count == 2
    assert second.embedded_count == 0
    assert second.skipped_existing_count == 2
    assert list(collection.records) == [row["chunk_id"] for row in rows]
    assert [collection.records[row["chunk_id"]]["document"] for row in rows] == [
        row["embedding_text"] for row in rows
    ]


def test_dense_index_persists_real_chroma_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """로컬 결정형 벡터로 실제 Chroma 생성·저장·재열기까지 확인한다."""

    input_path, rows, audit = make_input(tmp_path, monkeypatch)
    persist_directory = tmp_path / "chroma"
    local_embeddings = DeterministicFakeEmbedding(size=1536)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(advanced_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        advanced_module,
        "OpenAIEmbeddings",
        lambda **_: local_embeddings,
    )

    report = advanced_module.build_dense_index(
        input_path=input_path,
        audit=audit,
        persist_directory=persist_directory,
        collection_name="advanced-real-chroma-test",
        embedding_model="text-embedding-3-small",
        batch_size=2,
        max_records=None,
    )
    reopened = advanced_module.Chroma(
        collection_name="advanced-real-chroma-test",
        embedding_function=local_embeddings,
        persist_directory=str(persist_directory),
    )
    stored = reopened.get(
        ids=[row["chunk_id"] for row in rows],
        include=["documents", "metadatas"],
    )

    assert report.embedded_count == 2
    assert report.final_collection_count == 2
    assert stored["ids"] == [row["chunk_id"] for row in rows]
    assert stored["documents"] == [row["embedding_text"] for row in rows]
    assert stored["metadatas"][0]["content_type"] == "text"
    assert stored["metadatas"][1]["content_type"] == "table"


def test_bm25_indexes_text_only_and_reuses_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kiwi 토큰이 있는 일반 텍스트만 저장하고 동일 실행은 재사용한다."""

    rows = [
        make_text_row(),
        make_text_row(
            "source-1:ADV2:A512O51:C000002",
            source_id="source-1",
        ),
        make_table_row(),
    ]
    rows[1]["bm25_tokens"] = ["계약", "기간", "납품"]
    rows[1]["bm25_token_count"] = 3
    input_path, _, audit = make_input(tmp_path, monkeypatch, rows)
    bm25_directory = tmp_path / "bm25"

    first = advanced_module.build_bm25_index(
        input_path=input_path,
        audit=audit,
        bm25_directory=bm25_directory,
        max_records=None,
    )
    payload = advanced_module.load_bm25_artifact(
        Path(first.artifact_path),
        expected_input_sha256=audit.input_sha256,
    )
    second = advanced_module.build_bm25_index(
        input_path=input_path,
        audit=audit,
        bm25_directory=bm25_directory,
        max_records=None,
    )

    assert first.indexed_text_chunk_count == 2
    assert first.indexed_token_total == 8
    assert first.reused_existing is False
    assert payload["chunk_ids"] == [rows[0]["chunk_id"], rows[1]["chunk_id"]]
    assert payload["tokenized_corpus"] == [
        rows[0]["bm25_tokens"],
        rows[1]["bm25_tokens"],
    ]
    assert second.reused_existing is True
    assert second.artifact_sha256 == first.artifact_sha256


def test_bm25_artifact_rejects_different_input_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """검색 코드가 다른 청크용 BM25 파일을 잘못 열지 않게 한다."""

    input_path, _, audit = make_input(tmp_path, monkeypatch)
    report = advanced_module.build_bm25_index(
        input_path=input_path,
        audit=audit,
        bm25_directory=tmp_path / "bm25",
        max_records=None,
    )

    with pytest.raises(ValueError, match="입력 SHA-256"):
        advanced_module.load_bm25_artifact(
            Path(report.artifact_path),
            expected_input_sha256="different-sha",
        )
