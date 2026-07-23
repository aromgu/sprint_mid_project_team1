"""Naive 청크를 OpenAI로 임베딩하고 Chroma에 영속 저장하는 모듈."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DEFAULT_INPUT_PATH = Path(
    "/home/data/advanced/chunks/chunks_naive_rcts_v3_metadata_v1.jsonl.gz"
)
DEFAULT_PERSIST_DIRECTORY = Path("/home/data/chroma_naive_v3_metadata_v1")
DEFAULT_REPORT_PATH = Path(
    "/home/data/reports/ai11_policy_naive_v3_metadata_v1_indexing_report.json"
)
DEFAULT_COLLECTION_NAME = "ai11_policy_naive_v3_metadata_v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class InputContract:
    """API 비용을 쓰기 전에 확인할 승인된 청크 파일의 고정 계약이다."""

    name: str
    input_sha256: str
    chunk_count: int
    document_count: int
    total_tokens: int
    schema_version: str
    strategy_id: str
    metadata_schema_version: str | None = None


LEGACY_INPUT_CONTRACT = InputContract(
    name="naive_v1",
    input_sha256=("c94567760dee6352248b23f71f64e2e88db0f7bcb1a705c1c17c4bef6d030b99"),
    chunk_count=31_339,
    document_count=98,
    total_tokens=10_052_480,
    schema_version="rfp_naive_chunk_v1",
    strategy_id="naive_recursive_tiktoken_cl100k_base_512_102_v1",
)
RCTS_V2_INPUT_CONTRACT = InputContract(
    name="naive_rcts_v2",
    input_sha256=("b7e6293c17db71d4f887bfcdd268411b0d30255a499aac76392bcf1f36f79ec8"),
    chunk_count=31_451,
    document_count=98,
    total_tokens=10_316_103,
    schema_version="rfp_naive_chunk_v1",
    strategy_id="naive_langchain_recursive_cl100k_base_512_102_v2",
)
RCTS_V3_INPUT_CONTRACT = InputContract(
    name="naive_rcts_v3",
    input_sha256=("8d5107140ff20c5f78fa3b3a88c06a2149a1a31397a22e8fb1ca6cd32f3f7c09"),
    chunk_count=31_627,
    document_count=98,
    total_tokens=10_414_025,
    schema_version="rfp_naive_chunk_v1",
    strategy_id="naive_langchain_recursive_cl100k_base_512_102_v3",
)
RCTS_V3_METADATA_V1_INPUT_CONTRACT = InputContract(
    name="naive_rcts_v3_business_metadata_v1",
    input_sha256=("4c77826f4705f8df70dfa15d180312ec133d624ab25ace85cfd32f0c9f8f9194"),
    chunk_count=31_627,
    document_count=98,
    total_tokens=10_414_025,
    schema_version="rfp_naive_chunk_v1",
    strategy_id="naive_langchain_recursive_cl100k_base_512_102_v3",
    metadata_schema_version="business_metadata_v1",
)
INPUT_CONTRACTS_BY_SHA256 = {
    contract.input_sha256: contract
    for contract in (
        LEGACY_INPUT_CONTRACT,
        RCTS_V2_INPUT_CONTRACT,
        RCTS_V3_INPUT_CONTRACT,
        RCTS_V3_METADATA_V1_INPUT_CONTRACT,
    )
}

EXPECTED_EMBEDDING_DIMENSION = 1_536

# 검색 필터와 답변 근거 표시에 필요한 값만 Chroma metadata에 저장한다.
# raw_text/retrieval_text처럼 긴 문자열은 metadata에 중복 저장하지 않는다.
METADATA_FIELDS = (
    "chunk_id",
    "source_id",
    "document_id",
    "source_filename",
    "file_type",
    "source_row",
    "project_name",
    "project_summary",
    "issuer",
    "notice_number",
    "notice_round",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "project_amount_won",
    "project_amount_status",
    "bid_period_status",
    "validation_status",
    "project_summary_review_status",
    "business_metadata_match_rule",
    "metadata_schema_version",
    "content_type",
    "page_start",
    "page_end",
    "section_path",
    "section_idx_start",
    "section_idx_end",
    "chunk_order",
    "token_count",
    "strategy_id",
    "schema_version",
    "source_sha256",
    "table_id",
    "render_mode",
)


@dataclass(frozen=True)
class ChunkRecord:
    """Chroma 한 레코드에 필요한 최소 청크 데이터."""

    chunk_id: str
    retrieval_text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class InputAudit:
    """API 호출 전 입력 파일의 무결성 검사 결과."""

    input_sha256: str
    contract_name: str
    chunk_count: int
    document_count: int
    total_tokens: int
    chunk_ids: frozenset[str]
    schema_versions: tuple[str, ...]
    strategy_ids: tuple[str, ...]
    metadata_schema_versions: tuple[str, ...]


@dataclass(frozen=True)
class IndexingReport:
    """팀 공유용 임베딩·Chroma 저장 실행 결과."""

    started_at_utc: str
    finished_at_utc: str
    input_path: str
    input_sha256: str
    input_contract_name: str
    schema_version: str
    strategy_id: str
    metadata_schema_version: str | None
    source_document_count: int
    input_chunk_count: int
    embedding_model: str
    embedding_dimension: int
    collection_name: str
    persist_directory: str
    batch_size: int
    skipped_existing_count: int
    embedded_count: int
    final_collection_count: int
    embedding_api_seconds: float
    chroma_write_seconds: float
    total_seconds: float
    report_path: str


def utc_now_iso() -> str:
    """재현 가능한 실행 기록을 위해 UTC 현재 시각을 ISO 문자열로 반환한다."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """대용량 파일을 메모리에 한꺼번에 올리지 않고 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_jsonl(path: Path) -> TextIO:
    """일반 JSONL과 gzip JSONL을 동일한 텍스트 스트림으로 연다."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def normalize_metadata(
    row: dict[str, Any], *, embedding_model: str, create_date: str
) -> dict[str, str | int | float | bool]:
    """None·목록을 제외하고 Chroma 필터에 안전한 스칼라 metadata를 만든다."""

    metadata: dict[str, str | int | float | bool] = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value

    # XLSX의 기존 필드명도 유지해 다음 담당자가 별도 변환 없이 사용할 수 있게 한다.
    source_filename = row.get("source_filename")
    if isinstance(source_filename, str) and source_filename:
        metadata["file_nm"] = source_filename

    # Chroma metadata는 문자열 목록을 직접 받을 수 없으므로 JSON 문자열로 보존한다.
    # 검색 계층은 이 값을 파싱해 중복 파일명을 대표 문서로 연결할 수 있다.
    filename_aliases = row.get("filename_aliases")
    if isinstance(filename_aliases, list) and all(
        isinstance(value, str) for value in filename_aliases
    ):
        metadata["filename_aliases"] = json.dumps(
            filename_aliases,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metadata["filename_alias_count"] = len(filename_aliases)

    metadata["embedding_model"] = embedding_model
    metadata["create_date"] = create_date
    return metadata


def iter_rows(
    path: Path, *, max_records: int | None = None
) -> Iterator[dict[str, Any]]:
    """JSONL을 줄 단위로 읽어 메모리 사용량을 제한한다."""

    with open_jsonl(path) as file_obj:
        for index, line in enumerate(file_obj):
            if max_records is not None and index >= max_records:
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 실패: {path}:{index + 1}") from exc


def resolve_input_contract(input_sha256: str) -> InputContract:
    """파일 SHA로 승인된 Naive 청킹 버전의 계약을 찾는다."""

    contract = INPUT_CONTRACTS_BY_SHA256.get(input_sha256)
    if contract is None:
        raise ValueError(
            f"승인되지 않은 임베딩 입력 SHA-256입니다: actual={input_sha256}"
        )
    return contract


def audit_input(path: Path, *, max_records: int | None = None) -> InputAudit:
    """청크 수·ID·본문·스키마를 검사해 잘못된 입력의 API 비용 발생을 막는다."""

    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    input_sha256 = sha256_file(path)
    contract = resolve_input_contract(input_sha256)

    chunk_ids: set[str] = set()
    source_ids: set[str] = set()
    schema_versions: set[str] = set()
    strategy_ids: set[str] = set()
    metadata_schema_versions: set[str] = set()
    total_tokens = 0
    chunk_count = 0

    for row in iter_rows(path, max_records=max_records):
        chunk_id = row.get("chunk_id")
        retrieval_text = row.get("retrieval_text")
        source_id = row.get("source_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"chunk_id가 비어 있습니다: row={chunk_count + 1}")
        if chunk_id in chunk_ids:
            raise ValueError(f"중복 chunk_id가 있습니다: {chunk_id}")
        if not isinstance(retrieval_text, str) or not retrieval_text.strip():
            raise ValueError(f"retrieval_text가 비어 있습니다: {chunk_id}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"source_id가 비어 있습니다: {chunk_id}")

        token_count = row.get("token_count")
        if not isinstance(token_count, int) or not 1 <= token_count <= 512:
            raise ValueError(f"token_count 범위 오류: {chunk_id}={token_count}")

        chunk_ids.add(chunk_id)
        source_ids.add(source_id)
        schema_versions.add(str(row.get("schema_version")))
        strategy_ids.add(str(row.get("strategy_id")))
        metadata_schema_version = row.get("metadata_schema_version")
        if metadata_schema_version not in (None, ""):
            metadata_schema_versions.add(str(metadata_schema_version))
        total_tokens += token_count
        chunk_count += 1

    # smoke test도 잘못된 스키마·청킹 전략에는 API 비용을 쓰지 않는다.
    if schema_versions != {contract.schema_version}:
        raise ValueError(f"schema_version 오류: {sorted(schema_versions)}")
    if strategy_ids != {contract.strategy_id}:
        raise ValueError(f"strategy_id 오류: {sorted(strategy_ids)}")
    if contract.metadata_schema_version is not None and metadata_schema_versions != {
        contract.metadata_schema_version
    }:
        raise ValueError(
            "metadata_schema_version 오류: "
            f"{sorted(metadata_schema_versions)} != "
            f"{contract.metadata_schema_version}"
        )

    # 소량 smoke test가 아닌 전체 실행에서만 전체 건수 조건을 강제한다.
    if max_records is None:
        if chunk_count != contract.chunk_count:
            raise ValueError(f"청크 수 오류: {chunk_count} != {contract.chunk_count}")
        if len(source_ids) != contract.document_count:
            raise ValueError(
                f"고유 문서 수 오류: {len(source_ids)} != {contract.document_count}"
            )
        if total_tokens != contract.total_tokens:
            raise ValueError(
                f"전체 토큰 수 오류: {total_tokens} != {contract.total_tokens}"
            )

    return InputAudit(
        input_sha256=input_sha256,
        contract_name=contract.name,
        chunk_count=chunk_count,
        document_count=len(source_ids),
        total_tokens=total_tokens,
        chunk_ids=frozenset(chunk_ids),
        schema_versions=tuple(sorted(schema_versions)),
        strategy_ids=tuple(sorted(strategy_ids)),
        metadata_schema_versions=tuple(sorted(metadata_schema_versions)),
    )


def iter_chunk_batches(
    path: Path,
    *,
    batch_size: int,
    embedding_model: str,
    create_date: str,
    max_records: int | None = None,
) -> Iterator[list[ChunkRecord]]:
    """검증된 JSONL을 OpenAI·Chroma 처리 크기에 맞춘 배치로 변환한다."""

    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    batch: list[ChunkRecord] = []
    for row in iter_rows(path, max_records=max_records):
        batch.append(
            ChunkRecord(
                chunk_id=row["chunk_id"],
                retrieval_text=row["retrieval_text"],
                metadata=normalize_metadata(
                    row,
                    embedding_model=embedding_model,
                    create_date=create_date,
                ),
            )
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_vectorstore(
    *,
    embeddings: OpenAIEmbeddings,
    embedding_model: str,
    input_sha256: str,
    schema_version: str,
    strategy_id: str,
    metadata_schema_version: str | None,
    collection_name: str,
    persist_directory: Path,
) -> Chroma:
    """팀 합의 경로에 cosine 기반 영속 Chroma Collection을 연다."""

    persist_directory.mkdir(parents=True, exist_ok=True)
    collection_metadata = {
        "embedding_model": embedding_model,
        "schema_version": schema_version,
        "strategy_id": strategy_id,
        "input_sha256": input_sha256,
    }
    if metadata_schema_version is not None:
        collection_metadata["metadata_schema_version"] = metadata_schema_version

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata=collection_metadata,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def validate_collection_contract(
    metadata: dict[str, Any],
    *,
    audit: InputAudit,
    embedding_model: str,
) -> None:
    """다른 청크나 모델로 만든 Collection에 섞어 쓰는 사고를 막는다."""

    expected = {
        "embedding_model": embedding_model,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "input_sha256": audit.input_sha256,
    }
    if audit.metadata_schema_versions:
        expected["metadata_schema_version"] = audit.metadata_schema_versions[0]
    mismatches = {
        key: {"saved": metadata.get(key), "requested": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Collection 계약이 현재 입력과 다릅니다: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def save_report(report: IndexingReport, path: Path) -> None:
    """실행 결과를 JSON으로 원자적 저장해 팀원이 시간과 건수를 재확인하게 한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_embeddings(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    report_path: Path = DEFAULT_REPORT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_records: int | None = None,
) -> IndexingReport:
    """임베딩 API와 Chroma 저장 시간을 분리 측정하며 인덱스를 구축한다."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    started_at = utc_now_iso()
    total_started = perf_counter()
    audit = audit_input(input_path, max_records=max_records)

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        chunk_size=batch_size,
        max_retries=5,
    )
    vectorstore = create_vectorstore(
        embeddings=embeddings,
        embedding_model=embedding_model,
        input_sha256=audit.input_sha256,
        schema_version=audit.schema_versions[0],
        strategy_id=audit.strategy_ids[0],
        metadata_schema_version=(
            audit.metadata_schema_versions[0]
            if audit.metadata_schema_versions
            else None
        ),
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    collection_metadata = vectorstore._collection.metadata or {}
    validate_collection_contract(
        collection_metadata,
        audit=audit,
        embedding_model=embedding_model,
    )

    existing_ids = set(vectorstore.get(include=[]).get("ids", []))
    unexpected_ids = existing_ids - set(audit.chunk_ids)
    if unexpected_ids:
        sample = sorted(unexpected_ids)[:3]
        raise RuntimeError(f"현재 입력에 없는 ID가 Collection에 있습니다: {sample}")

    embedding_api_seconds = 0.0
    chroma_write_seconds = 0.0
    embedded_count = 0
    skipped_existing_count = 0
    embedding_dimension = 0
    create_date = started_at

    for batch in iter_chunk_batches(
        input_path,
        batch_size=batch_size,
        embedding_model=embedding_model,
        create_date=create_date,
        max_records=max_records,
    ):
        pending = [record for record in batch if record.chunk_id not in existing_ids]
        skipped_existing_count += len(batch) - len(pending)
        if not pending:
            continue

        texts = [record.retrieval_text for record in pending]
        ids = [record.chunk_id for record in pending]
        metadatas = [record.metadata for record in pending]

        # ① OpenAI 호출만 감싸 텍스트→벡터 변환 시간을 누적한다.
        embedding_started = perf_counter()
        vectors = embeddings.embed_documents(texts)
        embedding_api_seconds += perf_counter() - embedding_started

        if not vectors or len(vectors) != len(pending):
            raise RuntimeError("OpenAI 임베딩 응답 개수가 입력과 다릅니다.")
        embedding_dimension = len(vectors[0])
        if embedding_dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                "임베딩 차원이 예상값과 다릅니다: "
                f"{embedding_dimension} != {EXPECTED_EMBEDDING_DIMENSION}"
            )

        # ② 이미 계산된 벡터의 Chroma 기록 시간만 별도로 누적한다.
        chroma_started = perf_counter()
        vectorstore._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        chroma_write_seconds += perf_counter() - chroma_started

        existing_ids.update(ids)
        embedded_count += len(ids)
        print(
            f"진행: {len(existing_ids)}/{audit.chunk_count} "
            f"(이번 실행 임베딩 {embedded_count}개)"
        )

    final_collection_count = vectorstore._collection.count()
    if max_records is None and final_collection_count != audit.chunk_count:
        raise RuntimeError(
            "최종 Chroma 개수가 다릅니다: "
            f"{final_collection_count} != {audit.chunk_count}"
        )

    total_seconds = perf_counter() - total_started
    report = IndexingReport(
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        input_path=str(input_path),
        input_sha256=audit.input_sha256,
        input_contract_name=audit.contract_name,
        schema_version=audit.schema_versions[0],
        strategy_id=audit.strategy_ids[0],
        metadata_schema_version=(
            audit.metadata_schema_versions[0]
            if audit.metadata_schema_versions
            else None
        ),
        source_document_count=audit.document_count,
        input_chunk_count=audit.chunk_count,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension or EXPECTED_EMBEDDING_DIMENSION,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        batch_size=batch_size,
        skipped_existing_count=skipped_existing_count,
        embedded_count=embedded_count,
        final_collection_count=final_collection_count,
        embedding_api_seconds=round(embedding_api_seconds, 6),
        chroma_write_seconds=round(chroma_write_seconds, 6),
        total_seconds=round(total_seconds, 6),
        report_path=str(report_path),
    )
    save_report(report, report_path)
    return report
