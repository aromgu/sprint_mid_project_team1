"""Advanced v2 청크를 Dense Chroma와 BM25 인덱스로 저장한다.

Naive 인덱싱과 저장 경로·컬렉션을 완전히 분리하고, API 호출 전에 승인된
Advanced v4 청크의 SHA-256과 스키마 계약을 검사한다. Dense 검색에는
텍스트와 Markdown 표를 모두 포함하며, BM25에는 Kiwi 처리가 끝난 일반
텍스트 청크만 포함한다.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

DEFAULT_INPUT_PATH = Path(
    "/home/data/advanced/chunks_kss_512_51_v4/chunks_advanced_v4.jsonl.gz"
)
DEFAULT_PERSIST_DIRECTORY = Path("/home/data/chroma_advanced_v2")
DEFAULT_BM25_DIRECTORY = Path("/home/data/bm25_advanced_v2")
DEFAULT_REPORT_PATH = Path(
    "/home/data/reports/ai11_policy_advanced_v2_indexing_report.json"
)
DEFAULT_COLLECTION_NAME = "ai11_policy_advanced_v2"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
EXPECTED_EMBEDDING_DIMENSION = 1_536
BM25_ARTIFACT_FILENAME = "bm25_index.pkl"
BM25_MANIFEST_FILENAME = "bm25_manifest.json"


@dataclass(frozen=True)
class AdvancedInputContract:
    """API 비용을 허용할 Advanced 청크 파일의 고정 계약이다."""

    name: str
    input_sha256: str
    chunk_count: int
    document_count: int
    total_tokens: int
    text_chunk_count: int
    table_chunk_count: int
    bm25_chunk_count: int
    bm25_token_total: int
    schema_version: str
    strategy_id: str
    corpus_id: str


ADVANCED_V2_INPUT_CONTRACT = AdvancedInputContract(
    name="advanced_v2_final",
    input_sha256="c1b5cf4f6c0e346265850fd96320e74fed2fd50a6c673e84d0dd8640e5f45ed3",
    chunk_count=82_442,
    document_count=98,
    total_tokens=17_959_212,
    text_chunk_count=41_830,
    table_chunk_count=40_612,
    bm25_chunk_count=41_830,
    bm25_token_total=797_788,
    schema_version="rfp_advanced_chunk_v2",
    strategy_id=(
        "advanced_kss_kiwi_exclude_je_semantic_tail_page_marker_"
        "no_text_newline_cl100k_base_512_51_v2"
    ),
    corpus_id="advanced_v2",
)
INPUT_CONTRACTS_BY_SHA256 = {
    ADVANCED_V2_INPUT_CONTRACT.input_sha256: ADVANCED_V2_INPUT_CONTRACT
}


# 검색 필터와 답변 출처 표시에 필요한 스칼라 값만 Chroma에 저장한다.
# raw_text, embedding_text, bm25_tokens 같은 큰 값은 metadata에 중복 저장하지 않는다.
METADATA_FIELDS = (
    "chunk_id",
    "source_id",
    "document_id",
    "source_filename",
    "source_relative_path",
    "file_type",
    "project_name",
    "issuer",
    "notice_number",
    "notice_round",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "project_amount_won",
    "content_type",
    "page_start",
    "page_end",
    "section_path",
    "section_idx_start",
    "section_idx_end",
    "para_idx_start",
    "para_idx_end",
    "chunk_order",
    "token_count",
    "strategy_id",
    "schema_version",
    "corpus_id",
    "source_sha256",
    "table_id",
    "table_part_index",
    "table_part_count",
    "table_segment_index",
    "table_segment_count",
    "render_mode",
    "kss_applied",
    "bm25_eligible",
)


@dataclass(frozen=True)
class AdvancedChunkRecord:
    """Dense Chroma 한 레코드에 필요한 최소 데이터다."""

    chunk_id: str
    embedding_text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class AdvancedInputAudit:
    """API 호출 전에 확인한 Advanced 입력 무결성 결과다."""

    input_sha256: str
    contract_name: str
    chunk_count: int
    document_count: int
    total_tokens: int
    text_chunk_count: int
    table_chunk_count: int
    bm25_chunk_count: int
    bm25_token_total: int
    chunk_ids: frozenset[str]
    schema_versions: tuple[str, ...]
    strategy_ids: tuple[str, ...]
    corpus_ids: tuple[str, ...]


@dataclass(frozen=True)
class DenseIndexReport:
    """OpenAI 임베딩과 Chroma 저장 결과 및 분리 측정 시간이다."""

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


@dataclass(frozen=True)
class BM25IndexReport:
    """Kiwi 토큰 기반 BM25 인덱스 저장 결과다."""

    artifact_path: str
    artifact_sha256: str
    manifest_path: str
    indexed_text_chunk_count: int
    indexed_token_total: int
    rank_bm25_version: str
    reused_existing: bool
    build_seconds: float


@dataclass(frozen=True)
class AdvancedIndexingReport:
    """팀 공유용 Advanced Dense·BM25 전체 실행 결과다."""

    started_at_utc: str
    finished_at_utc: str
    input_path: str
    input_sha256: str
    input_contract_name: str
    schema_version: str
    strategy_id: str
    corpus_id: str
    source_document_count: int
    input_chunk_count: int
    text_chunk_count: int
    table_chunk_count: int
    mode: str
    dense: DenseIndexReport | None
    bm25: BM25IndexReport | None
    total_seconds: float
    report_path: str


def utc_now_iso() -> str:
    """실행 기록에 사용할 UTC 현재 시각을 ISO 문자열로 반환한다."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """대용량 파일을 한꺼번에 읽지 않고 SHA-256을 계산한다."""

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


def iter_rows(
    path: Path, *, max_records: int | None = None
) -> Iterator[dict[str, Any]]:
    """입력 JSONL을 줄 단위로 읽어 메모리 사용량을 제한한다."""

    with open_jsonl(path) as file_obj:
        yielded = 0
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            if max_records is not None and yielded >= max_records:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 실패: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSON 객체가 아닌 레코드입니다: {path}:{line_number}")
            yielded += 1
            yield row


def resolve_input_contract(input_sha256: str) -> AdvancedInputContract:
    """파일 SHA로 승인된 Advanced 입력 계약을 찾는다."""

    contract = INPUT_CONTRACTS_BY_SHA256.get(input_sha256)
    if contract is None:
        raise ValueError(
            f"승인되지 않은 Advanced 입력 SHA-256입니다: actual={input_sha256}"
        )
    return contract


def _require_nonempty_string(
    row: Mapping[str, Any],
    field: str,
    *,
    row_number: int,
) -> str:
    """필수 문자열 필드를 검사하고 정리된 값을 반환한다."""

    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}가 비어 있습니다: row={row_number}")
    return value


def _validate_bm25_contract(
    row: Mapping[str, Any],
    *,
    chunk_id: str,
    content_type: str,
) -> tuple[int, int]:
    """일반 텍스트에만 Kiwi BM25 토큰이 존재하는지 검사한다."""

    tokens = row.get("bm25_tokens")
    if not isinstance(tokens, list) or any(
        not isinstance(token, str) or not token for token in tokens
    ):
        raise ValueError(f"bm25_tokens 형식 오류: {chunk_id}")

    declared_count = row.get("bm25_token_count")
    if declared_count != len(tokens):
        raise ValueError(
            f"bm25_token_count 불일치: {chunk_id}={declared_count}/{len(tokens)}"
        )

    bm25_eligible = row.get("bm25_eligible")
    kss_applied = row.get("kss_applied")
    if content_type == "text":
        if bm25_eligible is not True or kss_applied is not True or not tokens:
            raise ValueError(f"일반 텍스트 BM25·KSS 계약 오류: {chunk_id}")
        return 1, len(tokens)

    if bm25_eligible is not False or kss_applied is not False or tokens:
        raise ValueError(f"표는 BM25·KSS 대상이 될 수 없습니다: {chunk_id}")
    return 0, 0


def audit_advanced_input(
    path: Path,
    *,
    max_records: int | None = None,
) -> AdvancedInputAudit:
    """스키마·본문·Dense·BM25 계약을 검사해 잘못된 API 호출을 막는다."""

    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records는 1 이상이어야 합니다.")

    input_sha256 = sha256_file(path)
    contract = resolve_input_contract(input_sha256)

    chunk_ids: set[str] = set()
    source_ids: set[str] = set()
    schema_versions: set[str] = set()
    strategy_ids: set[str] = set()
    corpus_ids: set[str] = set()
    content_counts: Counter[str] = Counter()
    total_tokens = 0
    bm25_chunk_count = 0
    bm25_token_total = 0

    for row_number, row in enumerate(
        iter_rows(path, max_records=max_records),
        start=1,
    ):
        chunk_id = _require_nonempty_string(row, "chunk_id", row_number=row_number)
        source_id = _require_nonempty_string(row, "source_id", row_number=row_number)
        embedding_text = _require_nonempty_string(
            row,
            "embedding_text",
            row_number=row_number,
        )
        if chunk_id in chunk_ids:
            raise ValueError(f"중복 chunk_id가 있습니다: {chunk_id}")

        content_type = row.get("content_type")
        if content_type not in {"text", "table"}:
            raise ValueError(f"content_type 오류: {chunk_id}={content_type}")

        token_count = row.get("token_count")
        if not isinstance(token_count, int) or not 1 <= token_count <= 512:
            raise ValueError(f"token_count 범위 오류: {chunk_id}={token_count}")
        if row.get("token_count_basis") != "embedding_text":
            raise ValueError(f"token_count_basis 오류: {chunk_id}")
        if row.get("vectorize_field") != "embedding_text":
            raise ValueError(f"vectorize_field 오류: {chunk_id}")
        if row.get("embedding_prefix_included") is not False:
            raise ValueError(f"임베딩 접두사가 포함된 청크입니다: {chunk_id}")
        if content_type == "text" and (
            "\n" in embedding_text or "\r" in embedding_text
        ):
            raise ValueError(
                f"일반 텍스트 embedding_text에 줄바꿈이 있습니다: {chunk_id}"
            )

        bm25_count, bm25_tokens = _validate_bm25_contract(
            row,
            chunk_id=chunk_id,
            content_type=content_type,
        )

        chunk_ids.add(chunk_id)
        source_ids.add(source_id)
        schema_versions.add(str(row.get("schema_version")))
        strategy_ids.add(str(row.get("strategy_id")))
        corpus_ids.add(str(row.get("corpus_id")))
        content_counts[content_type] += 1
        total_tokens += token_count
        bm25_chunk_count += bm25_count
        bm25_token_total += bm25_tokens

    expected_sets = (
        ("schema_version", schema_versions, {contract.schema_version}),
        ("strategy_id", strategy_ids, {contract.strategy_id}),
        ("corpus_id", corpus_ids, {contract.corpus_id}),
    )
    for label, actual, expected in expected_sets:
        if actual != expected:
            raise ValueError(f"{label} 오류: {sorted(actual)}")

    if max_records is None:
        expected_values = {
            "청크 수": (len(chunk_ids), contract.chunk_count),
            "고유 문서 수": (len(source_ids), contract.document_count),
            "전체 토큰 수": (total_tokens, contract.total_tokens),
            "텍스트 청크 수": (
                content_counts["text"],
                contract.text_chunk_count,
            ),
            "표 청크 수": (content_counts["table"], contract.table_chunk_count),
            "BM25 청크 수": (bm25_chunk_count, contract.bm25_chunk_count),
            "BM25 토큰 수": (bm25_token_total, contract.bm25_token_total),
        }
        for label, (actual, expected) in expected_values.items():
            if actual != expected:
                raise ValueError(f"{label} 오류: {actual} != {expected}")

    return AdvancedInputAudit(
        input_sha256=input_sha256,
        contract_name=contract.name,
        chunk_count=len(chunk_ids),
        document_count=len(source_ids),
        total_tokens=total_tokens,
        text_chunk_count=content_counts["text"],
        table_chunk_count=content_counts["table"],
        bm25_chunk_count=bm25_chunk_count,
        bm25_token_total=bm25_token_total,
        chunk_ids=frozenset(chunk_ids),
        schema_versions=tuple(sorted(schema_versions)),
        strategy_ids=tuple(sorted(strategy_ids)),
        corpus_ids=tuple(sorted(corpus_ids)),
    )


def normalize_advanced_metadata(
    row: Mapping[str, Any],
    *,
    embedding_model: str,
    create_date: str,
) -> dict[str, str | int | float | bool]:
    """Advanced 청크를 Chroma 필터에 안전한 스칼라 metadata로 바꾼다."""

    metadata: dict[str, str | int | float | bool] = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value

    source_filename = row.get("source_filename")
    if isinstance(source_filename, str) and source_filename:
        # 기존 검색 담당자의 필드명과 호환되도록 별칭을 유지한다.
        metadata["file_nm"] = source_filename
    metadata["embedding_model"] = embedding_model
    metadata["create_date"] = create_date
    return metadata


def iter_dense_batches(
    path: Path,
    *,
    batch_size: int,
    embedding_model: str,
    create_date: str,
    max_records: int | None = None,
) -> Iterator[list[AdvancedChunkRecord]]:
    """검증된 청크를 OpenAI·Chroma 처리 크기에 맞춘 배치로 변환한다."""

    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    batch: list[AdvancedChunkRecord] = []
    for row in iter_rows(path, max_records=max_records):
        batch.append(
            AdvancedChunkRecord(
                chunk_id=row["chunk_id"],
                embedding_text=row["embedding_text"],
                metadata=normalize_advanced_metadata(
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


def create_advanced_vectorstore(
    *,
    embeddings: OpenAIEmbeddings,
    embedding_model: str,
    audit: AdvancedInputAudit,
    collection_name: str,
    persist_directory: Path,
) -> Chroma:
    """Naive와 분리된 cosine 기반 Advanced Chroma Collection을 연다."""

    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={
            "index_kind": "advanced_dense_v2",
            "embedding_model": embedding_model,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "schema_version": audit.schema_versions[0],
            "strategy_id": audit.strategy_ids[0],
            "corpus_id": audit.corpus_ids[0],
            "input_sha256": audit.input_sha256,
            "document_field": "embedding_text",
        },
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def validate_dense_collection_contract(
    metadata: Mapping[str, Any],
    *,
    audit: AdvancedInputAudit,
    embedding_model: str,
) -> None:
    """다른 입력·모델·Naive 벡터가 Advanced Collection에 섞이지 않게 한다."""

    expected = {
        "index_kind": "advanced_dense_v2",
        "embedding_model": embedding_model,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "corpus_id": audit.corpus_ids[0],
        "input_sha256": audit.input_sha256,
        "document_field": "embedding_text",
    }
    mismatches = {
        key: {"saved": metadata.get(key), "requested": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Advanced Collection 계약이 현재 입력과 다릅니다: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def build_dense_index(
    *,
    input_path: Path,
    audit: AdvancedInputAudit,
    persist_directory: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    max_records: int | None,
) -> DenseIndexReport:
    """OpenAI 임베딩과 Chroma 저장 시간을 분리 측정해 Dense 인덱스를 만든다."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        dimensions=EXPECTED_EMBEDDING_DIMENSION,
        chunk_size=batch_size,
        max_retries=5,
    )
    vectorstore = create_advanced_vectorstore(
        embeddings=embeddings,
        embedding_model=embedding_model,
        audit=audit,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    validate_dense_collection_contract(
        vectorstore._collection.metadata or {},
        audit=audit,
        embedding_model=embedding_model,
    )

    existing_ids = set(vectorstore.get(include=[]).get("ids", []))
    if max_records is None:
        unexpected_ids = existing_ids - set(audit.chunk_ids)
        if unexpected_ids:
            raise RuntimeError(
                "현재 입력에 없는 ID가 Advanced Collection에 있습니다: "
                f"{sorted(unexpected_ids)[:3]}"
            )

    embedding_api_seconds = 0.0
    chroma_write_seconds = 0.0
    embedded_count = 0
    skipped_existing_count = 0
    embedding_dimension = 0
    create_date = utc_now_iso()

    for batch in iter_dense_batches(
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

        texts = [record.embedding_text for record in pending]
        ids = [record.chunk_id for record in pending]
        metadatas = [record.metadata for record in pending]

        embedding_started = perf_counter()
        vectors = embeddings.embed_documents(texts)
        embedding_api_seconds += perf_counter() - embedding_started
        if not vectors or len(vectors) != len(pending):
            raise RuntimeError("OpenAI 임베딩 응답 개수가 입력과 다릅니다.")

        dimensions = {len(vector) for vector in vectors}
        if dimensions != {EXPECTED_EMBEDDING_DIMENSION}:
            raise RuntimeError(
                "임베딩 차원이 예상값과 다릅니다: "
                f"{sorted(dimensions)} != [{EXPECTED_EMBEDDING_DIMENSION}]"
            )
        embedding_dimension = EXPECTED_EMBEDDING_DIMENSION

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
        completed = skipped_existing_count + embedded_count
        print(
            f"Dense 진행: {completed}/{audit.chunk_count} "
            f"(이번 실행 임베딩 {embedded_count}개)"
        )

    final_collection_count = vectorstore._collection.count()
    if max_records is None and final_collection_count != audit.chunk_count:
        raise RuntimeError(
            "최종 Advanced Chroma 개수가 다릅니다: "
            f"{final_collection_count} != {audit.chunk_count}"
        )

    return DenseIndexReport(
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
    )


def _bm25_manifest_matches(
    manifest: Mapping[str, Any],
    *,
    artifact_path: Path,
    audit: AdvancedInputAudit,
    max_records: int | None,
) -> bool:
    """기존 BM25 결과가 현재 입력 선택과 완전히 같은지 확인한다."""

    expected = {
        "format": "advanced_bm25_pickle_v1",
        "input_sha256": audit.input_sha256,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "corpus_id": audit.corpus_ids[0],
        "selected_input_chunk_count": audit.chunk_count,
        "indexed_text_chunk_count": audit.bm25_chunk_count,
        "indexed_token_total": audit.bm25_token_total,
        "max_records": max_records,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    saved_sha = manifest.get("artifact_sha256")
    return (
        artifact_path.is_file()
        and isinstance(saved_sha, str)
        and sha256_file(artifact_path) == saved_sha
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """손상되거나 형식이 다른 기존 manifest는 재구축 대상으로 처리한다."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def build_bm25_index(
    *,
    input_path: Path,
    audit: AdvancedInputAudit,
    bm25_directory: Path,
    max_records: int | None,
) -> BM25IndexReport:
    """일반 텍스트의 사전 계산 Kiwi 토큰만 사용해 BM25를 원자적으로 저장한다."""

    started = perf_counter()
    bm25_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = bm25_directory / BM25_ARTIFACT_FILENAME
    manifest_path = bm25_directory / BM25_MANIFEST_FILENAME
    existing_manifest = _read_json_object(manifest_path)

    if existing_manifest and _bm25_manifest_matches(
        existing_manifest,
        artifact_path=artifact_path,
        audit=audit,
        max_records=max_records,
    ):
        return BM25IndexReport(
            artifact_path=str(artifact_path),
            artifact_sha256=str(existing_manifest["artifact_sha256"]),
            manifest_path=str(manifest_path),
            indexed_text_chunk_count=audit.bm25_chunk_count,
            indexed_token_total=audit.bm25_token_total,
            rank_bm25_version=str(existing_manifest["rank_bm25_version"]),
            reused_existing=True,
            build_seconds=round(perf_counter() - started, 6),
        )

    chunk_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []
    for row in iter_rows(input_path, max_records=max_records):
        if row.get("bm25_eligible") is not True:
            continue
        chunk_ids.append(row["chunk_id"])
        tokenized_corpus.append(list(row["bm25_tokens"]))

    if not tokenized_corpus:
        raise ValueError("BM25 대상 일반 텍스트 청크가 없습니다.")
    if len(chunk_ids) != audit.bm25_chunk_count:
        raise RuntimeError(
            "BM25 대상 청크 수가 audit 결과와 다릅니다: "
            f"{len(chunk_ids)} != {audit.bm25_chunk_count}"
        )

    bm25 = BM25Okapi(tokenized_corpus)
    payload = {
        "format": "advanced_bm25_pickle_v1",
        "input_sha256": audit.input_sha256,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "corpus_id": audit.corpus_ids[0],
        "chunk_ids": chunk_ids,
        "tokenized_corpus": tokenized_corpus,
        "index": bm25,
    }
    temporary_artifact = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with temporary_artifact.open("wb") as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_artifact.replace(artifact_path)

    artifact_sha256 = sha256_file(artifact_path)
    rank_bm25_version = version("rank-bm25")
    manifest = {
        "format": "advanced_bm25_pickle_v1",
        "created_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "input_sha256": audit.input_sha256,
        "schema_version": audit.schema_versions[0],
        "strategy_id": audit.strategy_ids[0],
        "corpus_id": audit.corpus_ids[0],
        "selected_input_chunk_count": audit.chunk_count,
        "indexed_text_chunk_count": len(chunk_ids),
        "indexed_token_total": sum(map(len, tokenized_corpus)),
        "max_records": max_records,
        "rank_bm25_version": rank_bm25_version,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    return BM25IndexReport(
        artifact_path=str(artifact_path),
        artifact_sha256=artifact_sha256,
        manifest_path=str(manifest_path),
        indexed_text_chunk_count=len(chunk_ids),
        indexed_token_total=sum(map(len, tokenized_corpus)),
        rank_bm25_version=rank_bm25_version,
        reused_existing=False,
        build_seconds=round(perf_counter() - started, 6),
    )


def load_bm25_artifact(
    path: Path,
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    """팀 검색 코드에서 BM25 pickle을 열고 최소 저장 계약을 검사한다.

    pickle은 파일을 만든 Python 코드를 실행할 수 있으므로, 팀 GCP의
    ``build_bm25_index``가 생성한 신뢰 가능한 파일에만 사용해야 한다.
    """

    if not path.is_file():
        raise FileNotFoundError(f"BM25 artifact가 없습니다: {path}")
    with path.open("rb") as file_obj:
        payload = pickle.load(file_obj)  # noqa: S301 - 팀 내부 생성 파일만 허용한다.

    if not isinstance(payload, dict):
        raise ValueError("BM25 artifact 최상위 값이 객체가 아닙니다.")
    if payload.get("format") != "advanced_bm25_pickle_v1":
        raise ValueError("지원하지 않는 BM25 artifact 형식입니다.")

    input_sha256 = payload.get("input_sha256")
    if expected_input_sha256 and input_sha256 != expected_input_sha256:
        raise ValueError(
            "BM25 artifact 입력 SHA-256이 다릅니다: "
            f"{input_sha256} != {expected_input_sha256}"
        )

    chunk_ids = payload.get("chunk_ids")
    tokenized_corpus = payload.get("tokenized_corpus")
    index = payload.get("index")
    if (
        not isinstance(chunk_ids, list)
        or not isinstance(tokenized_corpus, list)
        or len(chunk_ids) != len(tokenized_corpus)
        or not callable(getattr(index, "get_scores", None))
    ):
        raise ValueError("BM25 artifact 내부 인덱스 계약이 손상되었습니다.")
    return payload


def save_report(report: AdvancedIndexingReport, path: Path) -> None:
    """전체 실행 보고서를 원자적으로 JSON 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_advanced_indexes(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    bm25_directory: Path = DEFAULT_BM25_DIRECTORY,
    report_path: Path = DEFAULT_REPORT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_records: int | None = None,
    mode: str = "all",
) -> AdvancedIndexingReport:
    """입력을 한 번 검증한 뒤 요청한 Dense·BM25 인덱스를 구축한다."""

    if mode not in {"all", "dense", "bm25"}:
        raise ValueError(f"지원하지 않는 인덱싱 mode입니다: {mode}")

    started_at = utc_now_iso()
    total_started = perf_counter()
    audit = audit_advanced_input(input_path, max_records=max_records)

    bm25_report = None
    if mode in {"all", "bm25"}:
        bm25_report = build_bm25_index(
            input_path=input_path,
            audit=audit,
            bm25_directory=bm25_directory,
            max_records=max_records,
        )

    dense_report = None
    if mode in {"all", "dense"}:
        dense_report = build_dense_index(
            input_path=input_path,
            audit=audit,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            batch_size=batch_size,
            max_records=max_records,
        )

    report = AdvancedIndexingReport(
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        input_path=str(input_path),
        input_sha256=audit.input_sha256,
        input_contract_name=audit.contract_name,
        schema_version=audit.schema_versions[0],
        strategy_id=audit.strategy_ids[0],
        corpus_id=audit.corpus_ids[0],
        source_document_count=audit.document_count,
        input_chunk_count=audit.chunk_count,
        text_chunk_count=audit.text_chunk_count,
        table_chunk_count=audit.table_chunk_count,
        mode=mode,
        dense=dense_report,
        bm25=bm25_report,
        total_seconds=round(perf_counter() - total_started, 6),
        report_path=str(report_path),
    )
    save_report(report, report_path)
    return report
