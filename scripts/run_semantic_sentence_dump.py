"""시멘틱 청킹 준비 1단계: KSS 문장 경계를 파일로 저장한다.

시멘틱 청킹은 문장을 임베딩해 이웃 문장 사이 거리가 튀는 곳에서 자른다. 그
경계를 정하려면 임계값을 여러 번 바꿔가며 결과 청크 크기 분포를 봐야 하는데,
그때마다 KSS를 다시 돌리면 시도당 35분이 든다. 그래서 문장 경계를 한 번만
계산해 저장하고, 이후 단계는 이 파일만 읽는다.

문장 경계는 ``resolve_stream_sentences()``로 계산한다. 512·1024 고정 크기 청킹과
**같은 함수**를 쓰므로 세 조건이 동일한 문장 경계를 공유한다. 경계가 조건마다
다르면 비교 결과가 청킹 방식 차이인지 경계 차이인지 구분할 수 없다.

표는 이 단계에 들어오지 않는다. 표는 8,191 예외를 받아 통짜로 유지되므로 세
조건에서 동일하고, 문장 분리 대상이 아니다.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunking.advanced_chunking import (  # noqa: E402
    AdvancedTextStream,
    KssSentenceSplitter,
    TiktokenCodec,
    build_advanced_streams,
    resolve_stream_sentences,
)

DEFAULT_DOCUMENTS = Path(
    "/home/data/advanced/preprocessed_v4/documents_advanced_v1.jsonl"
)
DEFAULT_BLOCKS = Path("/home/data/advanced/preprocessed_v4/blocks_advanced_v1.jsonl")
DEFAULT_OUTPUT = Path("/home/data/advanced/semantic_v1/sentences.jsonl.gz")
DEFAULT_REPORT = Path("/home/data/advanced/semantic_v1/sentence_dump_report.json")
MODEL = "text-embedding-3-small"
ENCODING = "cl100k_base"

_WORKER_SPLITTER: Any = None
_WORKER_CODEC: Any = None


def _dump_one_document(
    payload: tuple[dict[str, Any], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """문서 하나의 텍스트 stream을 문장 단위 레코드로 바꾼다."""
    global _WORKER_SPLITTER, _WORKER_CODEC
    document, blocks = payload
    if _WORKER_SPLITTER is None:
        _WORKER_SPLITTER = KssSentenceSplitter()
        _WORKER_CODEC = TiktokenCodec(MODEL, ENCODING)

    rows: list[dict[str, Any]] = []
    for stream in build_advanced_streams(document, blocks):
        # 표 stream은 dict로 온다. 문장 분리 대상이 아니므로 건너뛴다.
        if not isinstance(stream, AdvancedTextStream):
            continue
        resolved = resolve_stream_sentences(document, stream, _WORKER_SPLITTER)
        for span in resolved.spans:
            rows.append(
                {
                    "source_id": str(document["source_id"]),
                    "stream_id": stream.stream_id,
                    "stream_order": stream.stream_order,
                    "boundary_type": stream.boundary_type,
                    "boundary_id": stream.boundary_id,
                    "sentence_index": span.sentence_index,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    # 임베딩과 토큰 수는 정규화된 본문을 기준으로 한다.
                    # 청크의 embedding_text와 같은 정규화를 거친 문자열이다.
                    "text": span.normalized_text,
                    "token_count": len(_WORKER_CODEC.encode(span.normalized_text)),
                    "stream_sentence_count": len(resolved.spans),
                    "alignment_fallback": resolved.alignment_fallback,
                    "sanitized_character_count": resolved.sanitized_character_count,
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="문서를 병렬 처리할 프로세스 수입니다. 워커마다 KSS 모델을 들고 있습니다.",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """문장 경계를 계산해 결정적인 순서로 저장한다."""
    args = build_parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"출력이 이미 있습니다: {args.output} (--overwrite 필요)")

    documents = [json.loads(line) for line in args.documents.read_text().splitlines()]
    if args.max_documents:
        documents = documents[: args.max_documents]
    wanted = {str(document["source_id"]) for document in documents}

    blocks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.blocks.open() as handle:
        for line in handle:
            block = json.loads(line)
            source_id = str(block["source_id"])
            if source_id in wanted:
                blocks_by_source[source_id].append(block)

    payloads = [
        (document, blocks_by_source.get(str(document["source_id"]), []))
        for document in documents
    ]
    total = len(payloads)
    # 결과를 입력 문서 순서로 다시 이어 붙여 병렬이어도 순서가 같게 한다.
    results: list[list[dict[str, Any]]] = [[] for _ in range(total)]

    if args.workers <= 1:
        for index, payload in enumerate(payloads):
            results[index] = _dump_one_document(payload)
            print(
                f"진행: {index + 1}/{total} "
                f"({payload[0]['source_id']}, 문장 {len(results[index])}개)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_dump_one_document, payload): index
                for index, payload in enumerate(payloads)
            }
            done = 0
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                done += 1
                print(
                    f"진행: {done}/{total} "
                    f"({payloads[index][0]['source_id']}, "
                    f"문장 {len(results[index])}개)",
                    flush=True,
                )

    rows = [row for group in results for row in group]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    # 팀 전달·재현을 위해 timestamp 없는 결정적 gzip으로 쓴다.
    with (
        temporary.open("wb") as binary_stream,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=binary_stream, mtime=0
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text_stream,
    ):
        for row in rows:
            text_stream.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            text_stream.write("\n")
    temporary.chmod(0o660)
    temporary.replace(args.output)

    token_counts = [int(row["token_count"]) for row in rows]
    stream_ids = {row["stream_id"] for row in rows}
    fallback_streams = {row["stream_id"] for row in rows if row["alignment_fallback"]}
    single_sentence_streams = {
        row["stream_id"] for row in rows if int(row["stream_sentence_count"]) <= 1
    }
    ordered = sorted(token_counts)
    report = {
        "documents": total,
        "text_streams": len(stream_ids),
        "sentences": len(rows),
        "single_sentence_streams": len(single_sentence_streams),
        "alignment_fallback_streams": len(fallback_streams),
        "sentence_token_min": ordered[0] if ordered else 0,
        "sentence_token_median": ordered[len(ordered) // 2] if ordered else 0,
        "sentence_token_mean": round(sum(ordered) / len(ordered), 2) if ordered else 0,
        "sentence_token_p90": ordered[int(len(ordered) * 0.9)] if ordered else 0,
        "sentence_token_max": ordered[-1] if ordered else 0,
        "sentence_token_total": sum(ordered),
        "output_path": str(args.output),
        "tokenizer_model": MODEL,
        "tokenizer_encoding": ENCODING,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
