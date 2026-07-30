"""저장된 문장 거리로 Advanced p83 시멘틱 청크를 만드는 GCP CLI.

1단계 문장 덤프와 2단계 거리 파일을 현재 전처리 stream과 먼저 대조한다.
선택한 거리 임계값을 넘는 문장 경계만 텍스트 청킹에 전달하며, 표는 고정
512·1024 실험과 같은 Markdown/Dense·평문/BM25 정책을 그대로 사용한다.

문서 단위 프로세스 병렬화를 지원하지만 결과는 source_id 순서로 다시 합쳐
워커 수와 관계없이 같은 JSONL을 만든다. 모든 품질 게이트가 통과한 경우에만
최종 gzip JSONL과 보고서를 원자적으로 저장한다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.run_advanced_chunking import (
    RUN_REPORT_SCHEMA_VERSION,
    _failed_validation_gates,
    ensure_output_policy,
    read_jsonl,
    select_documents,
    sha256_file,
    validate_advanced_inputs,
    validate_no_embedding_prefix,
    write_deterministic_jsonl_gzip,
    write_report_temporary,
)
from src.chunking.advanced_chunking import (
    CORPUS_ID,
    INPUT_SCHEMA_VERSION,
    PAGE_MARKER_DETECTOR_ID,
    SCHEMA_VERSION,
    TEXT_EMBEDDING_NORMALIZATION_ID,
    AdvancedChunkConfig,
    AdvancedTextStream,
    KiwiBm25Tokenizer,
    KssSentenceSplitter,
    TiktokenCodec,
    build_advanced_chunk_corpus,
    build_advanced_streams,
    build_advanced_summary,
    semantic_strategy_id_for,
    validate_advanced_chunks,
)

DEFAULT_DOCUMENTS = Path(
    "/home/data/advanced/preprocessed_v4/documents_advanced_v1.jsonl"
)
DEFAULT_BLOCKS = Path("/home/data/advanced/preprocessed_v4/blocks_advanced_v1.jsonl")
DEFAULT_SENTENCES = Path("/home/data/advanced/semantic_v1/sentences.jsonl.gz")
DEFAULT_DISTANCES = Path("/home/data/advanced/semantic_v1/distances.jsonl.gz")
DEFAULT_OUTPUT_DIR = Path("/home/data/advanced/chunks_v7_semantic_p83")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "chunks_advanced_semantic_p83.jsonl.gz"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "semantic_chunking_report_p83.json"
DEFAULT_THRESHOLD = 0.706
DEFAULT_THRESHOLD_LABEL = "p83"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MIN_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 0
SENTENCE_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True, slots=True)
class SemanticBoundaryData:
    """검증된 stream별 절단 후보와 보고서용 통계를 함께 보관한다."""

    cut_after_by_stream: dict[str, frozenset[int]]
    sentence_count: int
    stream_count: int
    candidate_boundary_count: int
    selected_boundary_count: int
    streams_with_selected_boundary: int
    distance_min: float
    distance_max: float


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    """gzip JSONL을 줄 번호가 포함된 오류로 읽는다."""
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"gzip JSONL 파싱 실패: {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"JSON 객체가 아닌 행입니다: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"gzip JSONL이 비어 있습니다: {path}")
    return rows


def build_selected_text_streams(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
) -> dict[str, AdvancedTextStream]:
    """현재 전처리 입력에서 시멘틱 대상 텍스트 stream을 다시 만든다."""
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)

    streams: dict[str, AdvancedTextStream] = {}
    for document in documents:
        source_id = str(document["source_id"])
        for stream in build_advanced_streams(
            document,
            blocks_by_source.get(source_id, []),
        ):
            if not isinstance(stream, AdvancedTextStream):
                continue
            if stream.stream_id in streams:
                raise ValueError(f"중복 텍스트 stream_id입니다: {stream.stream_id}")
            streams[stream.stream_id] = stream
    return streams


def load_semantic_boundaries(
    sentences_path: Path,
    distances_path: Path,
    threshold: float,
    selected_streams: Mapping[str, AdvancedTextStream],
) -> SemanticBoundaryData:
    """문장·거리 파일이 현재 stream과 일치하는지 검증하고 절단 후보를 만든다.

    문장 덤프가 다른 전처리 결과에서 만들어졌다면 같은 ``stream_id``가 우연히
    남아 있을 수 있다. 그래서 문장 인덱스뿐 아니라 각 문자 span이 0부터 stream
    끝까지 빈틈없이 덮는지도 확인한다.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("코사인 거리 임계값은 0 이상 2 이하의 유한한 수여야 합니다")
    selected_ids = set(selected_streams)
    if not selected_ids:
        raise ValueError("선택된 입력에 텍스트 stream이 없습니다")

    sentence_rows = [
        row
        for row in _read_gzip_jsonl(sentences_path)
        if str(row.get("stream_id") or "") in selected_ids
    ]
    rows_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sentence_rows:
        rows_by_stream[str(row["stream_id"])].append(row)
    if set(rows_by_stream) != selected_ids:
        missing = sorted(selected_ids - set(rows_by_stream))
        extra = sorted(set(rows_by_stream) - selected_ids)
        raise ValueError(
            "문장 덤프와 현재 텍스트 stream 목록이 다릅니다: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    sentence_index_by_stream: dict[str, dict[int, dict[str, Any]]] = {}
    for stream_id, rows in rows_by_stream.items():
        rows.sort(key=lambda row: int(row["sentence_index"]))
        expected_indices = list(range(1, len(rows) + 1))
        actual_indices = [int(row["sentence_index"]) for row in rows]
        if actual_indices != expected_indices:
            raise ValueError(f"문장 인덱스가 연속되지 않습니다: {stream_id}")
        if any(int(row["stream_sentence_count"]) != len(rows) for row in rows):
            raise ValueError(f"stream 문장 수 metadata가 다릅니다: {stream_id}")

        stream = selected_streams[stream_id]
        cursor = 0
        for row in rows:
            source_id = str(row.get("source_id") or "")
            start = row.get("char_start")
            end = row.get("char_end")
            if source_id != str(stream.blocks[0]["source_id"]):
                raise ValueError(f"문장 source_id가 stream과 다릅니다: {stream_id}")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start != cursor
                or not start < end <= len(stream.text)
            ):
                raise ValueError(f"문장 문자 span이 연속되지 않습니다: {stream_id}")
            cursor = end
        if cursor != len(stream.text):
            raise ValueError(f"문장 덤프가 stream 끝까지 덮지 않습니다: {stream_id}")
        sentence_index_by_stream[stream_id] = {
            int(row["sentence_index"]): row for row in rows
        }

    distance_rows = [
        row
        for row in _read_gzip_jsonl(distances_path)
        if str(row.get("stream_id") or "") in selected_ids
    ]
    distances_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_boundaries: set[tuple[str, int]] = set()
    distance_values: list[float] = []
    selected: dict[str, set[int]] = {stream_id: set() for stream_id in selected_ids}

    for row in distance_rows:
        stream_id = str(row.get("stream_id") or "")
        source_id = str(row.get("source_id") or "")
        left = row.get("left_sentence_index")
        right = row.get("right_sentence_index")
        boundary_char = row.get("boundary_char")
        distance = row.get("distance")
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or right != left + 1
            or not isinstance(boundary_char, int)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or not 0 <= float(distance) <= 2
        ):
            raise ValueError(f"유효하지 않은 의미 거리 행입니다: {stream_id}")
        key = (stream_id, left)
        if key in seen_boundaries:
            raise ValueError(f"중복 의미 거리 경계입니다: {stream_id}:{left}")
        seen_boundaries.add(key)

        sentence_rows_by_index = sentence_index_by_stream[stream_id]
        if (
            source_id != str(selected_streams[stream_id].blocks[0]["source_id"])
            or left not in sentence_rows_by_index
            or right not in sentence_rows_by_index
            or boundary_char != int(sentence_rows_by_index[right]["char_start"])
        ):
            raise ValueError(f"거리 경계가 문장 덤프와 다릅니다: {stream_id}:{left}")
        distances_by_stream[stream_id].append(row)
        distance_values.append(float(distance))
        # 팀 결정: p83 값과 같은 경계는 포함하지 않고 실제로 더 큰 거리만 자른다.
        if float(distance) > threshold:
            selected[stream_id].add(left)

    for stream_id, sentence_map in sentence_index_by_stream.items():
        expected = max(0, len(sentence_map) - 1)
        actual = len(distances_by_stream.get(stream_id, []))
        if actual != expected:
            raise ValueError(
                f"stream 거리 수가 문장 수와 다릅니다: {stream_id} "
                f"({actual} != {expected})"
            )

    if not distance_values:
        raise ValueError("선택된 입력에 의미 거리 후보가 없습니다")
    frozen = {stream_id: frozenset(indices) for stream_id, indices in selected.items()}
    return SemanticBoundaryData(
        cut_after_by_stream=frozen,
        sentence_count=len(sentence_rows),
        stream_count=len(rows_by_stream),
        candidate_boundary_count=len(distance_values),
        selected_boundary_count=sum(len(values) for values in frozen.values()),
        streams_with_selected_boundary=sum(bool(values) for values in frozen.values()),
        distance_min=min(distance_values),
        distance_max=max(distance_values),
    )


_WORKER_SPLITTER: Any = None
_WORKER_KIWI: Any = None


def _chunk_one_document(
    payload: tuple[
        Mapping[str, Any],
        list[dict[str, Any]],
        AdvancedChunkConfig,
        dict[str, frozenset[int]],
    ],
) -> list[dict[str, Any]]:
    """워커 하나에서 문서 한 건을 시멘틱 경계로 청킹한다."""
    global _WORKER_SPLITTER, _WORKER_KIWI
    document, blocks, config, boundaries = payload
    if _WORKER_SPLITTER is None:
        _WORKER_SPLITTER = KssSentenceSplitter()
        _WORKER_KIWI = KiwiBm25Tokenizer()
    codec = TiktokenCodec(config.model_name, config.encoding_name)
    return build_advanced_chunk_corpus(
        [document],
        blocks,
        codec,
        config,
        _WORKER_SPLITTER,
        _WORKER_KIWI,
        boundaries,
    )


def build_semantic_corpus_with_progress(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[dict[str, Any]],
    config: AdvancedChunkConfig,
    boundaries: Mapping[str, frozenset[int]],
    workers: int,
) -> list[dict[str, Any]]:
    """문서 단위 병렬 처리 후 source_id 순서로 결정적인 corpus를 만든다."""
    if workers < 1:
        raise ValueError("--workers는 1 이상이어야 합니다")
    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_source[str(block["source_id"])].append(block)
    boundaries_by_source: dict[str, dict[str, frozenset[int]]] = defaultdict(dict)
    for stream_id, indices in boundaries.items():
        source_id = stream_id.split(":AS", 1)[0]
        boundaries_by_source[source_id][stream_id] = indices

    ordered_documents = sorted(documents, key=lambda row: str(row["source_id"]))
    payloads = [
        (
            document,
            blocks_by_source.get(str(document["source_id"]), []),
            config,
            boundaries_by_source.get(str(document["source_id"]), {}),
        )
        for document in ordered_documents
    ]
    results: list[list[dict[str, Any]]] = [[] for _ in payloads]
    total = len(payloads)

    def report(done: int, index: int) -> None:
        source_id = str(payloads[index][0]["source_id"])
        print(
            f"진행: {done}/{total} ({source_id}, 청크 {len(results[index])}개)",
            flush=True,
        )

    if workers == 1:
        for index, payload in enumerate(payloads):
            results[index] = _chunk_one_document(payload)
            report(index + 1, index)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_chunk_one_document, payload): index
                for index, payload in enumerate(payloads)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                results[index] = future.result()
                report(done, index)
    return [chunk for document_chunks in results for chunk in document_chunks]


def build_parser() -> argparse.ArgumentParser:
    """시멘틱 실험 경로·임계값·병렬 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--distances", type=Path, default=DEFAULT_DISTANCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--threshold-label",
        default=DEFAULT_THRESHOLD_LABEL,
        help="strategy_id와 보고서에 기록할 임계값 이름입니다(예: p83).",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """입력·거리 계약을 감사하고 검증된 시멘틱 청크만 저장한다."""
    args = build_parser().parse_args()
    os.umask(0o007)
    for path in (args.documents, args.blocks, args.sentences, args.distances):
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    if args.overlap_tokens != 0:
        raise ValueError("현재 시멘틱 실험은 자연 경계 overlap 0으로 고정합니다")

    documents_sha256 = sha256_file(args.documents)
    blocks_sha256 = sha256_file(args.blocks)
    sentences_sha256 = sha256_file(args.sentences)
    distances_sha256 = sha256_file(args.distances)
    documents = read_jsonl(args.documents)
    blocks = read_jsonl(args.blocks)
    full_input_validation = validate_advanced_inputs(documents, blocks)
    selected_documents, selected_blocks = select_documents(
        documents,
        blocks,
        args.max_documents,
    )
    selected_input_validation = validate_advanced_inputs(
        selected_documents,
        selected_blocks,
    )
    selected_streams = build_selected_text_streams(
        selected_documents,
        selected_blocks,
    )
    boundary_data = load_semantic_boundaries(
        args.sentences,
        args.distances,
        args.distance_threshold,
        selected_streams,
    )

    strategy_id = semantic_strategy_id_for(
        args.threshold_label,
        args.max_tokens,
        args.overlap_tokens,
    )
    config = AdvancedChunkConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tail_tokens=0,
        semantic_min_tokens=args.min_tokens,
        semantic_distance_threshold=args.distance_threshold,
        semantic_threshold_label=args.threshold_label,
        strategy_id=strategy_id,
    )
    codec = TiktokenCodec(config.model_name, config.encoding_name)
    provenance = {
        "semantic_mode": True,
        "semantic_sentence_embedding_model": SENTENCE_EMBEDDING_MODEL,
        "semantic_distance_metric": "cosine_distance_adjacent_sentences",
        "semantic_boundary_operator": ">",
        "semantic_distance_threshold": args.distance_threshold,
        "semantic_threshold_label": args.threshold_label,
        "semantic_min_tokens": args.min_tokens,
        "semantic_sentence_count": boundary_data.sentence_count,
        "semantic_stream_count": boundary_data.stream_count,
        "semantic_candidate_boundary_count": (boundary_data.candidate_boundary_count),
        "semantic_selected_boundary_count": boundary_data.selected_boundary_count,
        "semantic_streams_with_selected_boundary": (
            boundary_data.streams_with_selected_boundary
        ),
        "semantic_distance_min": boundary_data.distance_min,
        "semantic_distance_max": boundary_data.distance_max,
        "sentences_path": str(args.sentences.expanduser().resolve()),
        "sentences_sha256": sentences_sha256,
        "distances_path": str(args.distances.expanduser().resolve()),
        "distances_sha256": distances_sha256,
    }
    validation_report = {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "chunk_schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "strategy_id": config.strategy_id,
        "mode": "validate_only",
        "documents_path": str(args.documents.expanduser().resolve()),
        "documents_sha256": documents_sha256,
        "blocks_path": str(args.blocks.expanduser().resolve()),
        "blocks_sha256": blocks_sha256,
        "full_input_validation": full_input_validation,
        "selected_input_validation": selected_input_validation,
        "max_documents": args.max_documents,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "workers": args.workers,
        **provenance,
    }
    if args.validate_only:
        print(
            json.dumps(validation_report, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return

    ensure_output_policy((args.output, args.report), args.overwrite)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    print(
        f"시멘틱 청킹 시작: 문서 {len(selected_documents)}개, "
        f"stream {boundary_data.stream_count}개, 워커 {args.workers}개, "
        f"{args.threshold_label}>{args.distance_threshold}",
        flush=True,
    )
    chunks = build_semantic_corpus_with_progress(
        selected_documents,
        selected_blocks,
        config,
        boundary_data.cut_after_by_stream,
        args.workers,
    )
    print(f"청킹 완료: 청크 {len(chunks)}개, 검증 시작", flush=True)
    validation = validate_advanced_chunks(
        selected_documents,
        selected_blocks,
        chunks,
        codec,
        config,
    )
    if not validation.get("overall_pass"):
        failed = _failed_validation_gates(validation)
        raise ValueError(f"시멘틱 청킹 품질 검증 실패: {', '.join(failed)}")
    validate_no_embedding_prefix(chunks)

    applied_boundaries = sum(
        chunk.get("content_type") == "text"
        and chunk.get("semantic_boundary_cut") is True
        for chunk in chunks
    )
    temporary_output: Path | None = None
    temporary_report: Path | None = None
    try:
        temporary_output = write_deterministic_jsonl_gzip(args.output, chunks)
        output_sha256 = sha256_file(temporary_output)
        report = {
            **validation_report,
            "mode": "semantic_chunking",
            "started_at_utc": started_at.isoformat(timespec="seconds"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "output_path": str(args.output.expanduser().resolve()),
            "output_sha256": output_sha256,
            "output_chunk_count": len(chunks),
            "report_path": str(args.report.expanduser().resolve()),
            "semantic_applied_boundary_count": applied_boundaries,
            "semantic_skipped_boundary_count": (
                boundary_data.selected_boundary_count - applied_boundaries
            ),
            "page_marker_detector_id": PAGE_MARKER_DETECTOR_ID,
            "embedding_text_field": "embedding_text",
            "text_embedding_normalization": TEXT_EMBEDDING_NORMALIZATION_ID,
            "token_count_basis": "embedding_text",
            "summary": build_advanced_summary(
                selected_documents,
                selected_blocks,
                chunks,
                validation,
                codec,
                config,
            ),
        }
        temporary_report = write_report_temporary(args.report, report)
        temporary_output.replace(args.output)
        temporary_output = None
        temporary_report.replace(args.report)
        temporary_report = None
        args.output.chmod(0o660)
        args.report.chmod(0o660)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
