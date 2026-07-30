"""시멘틱 청킹 준비 2단계: 문장을 임베딩해 이웃 문장 사이 거리를 계산한다.

시멘틱 청킹은 "내용이 바뀌는 곳"에서 자른다. 그 지점을 찾는 방법은 이웃한 두
문장의 임베딩 코사인 거리를 재서, 거리가 크게 벌어지는 곳을 화제 전환으로 보는
것이다.

임계값(상위 몇 %에서 자를지)은 코퍼스마다 달라 한 번에 정할 수 없다. 그래서
임베딩과 거리를 파일로 저장하고, 3단계에서 임계값만 바꿔가며 결과를 본다.
API 재호출 없이 초 단위로 시도할 수 있다.

거리는 **같은 stream 안에서만** 계산한다. stream은 PDF page 또는 HWP
section_path 경계이므로, 그 경계를 넘는 거리는 애초에 자를 지점이 아니다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunking.advanced_chunking import TiktokenCodec  # noqa: E402

DEFAULT_SENTENCES = Path("/home/data/advanced/semantic_v1/sentences.jsonl.gz")
DEFAULT_EMBEDDINGS = Path("/home/data/advanced/semantic_v1/sentence_embeddings.npy")
DEFAULT_DISTANCES = Path("/home/data/advanced/semantic_v1/distances.jsonl.gz")
DEFAULT_REPORT = Path("/home/data/advanced/semantic_v1/distance_report.json")
MODEL = "text-embedding-3-small"
BATCH = 300
# text-embedding-3-small의 입력 상한이다. 거리 계산은 화제 전환 여부만 보므로
# 상한을 넘는 문장은 앞부분만 임베딩한다. 실제 청크는 원문 전체를 그대로 쓴다.
EMBEDDING_INPUT_LIMIT = 8191


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--distances", type=Path, default=DEFAULT_DISTANCES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """문장 임베딩과 이웃 거리를 계산해 저장한다."""
    args = build_parser().parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 설정되지 않았습니다")

    rows: list[dict] = []
    with gzip.open(args.sentences, "rt") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if args.max_sentences:
        rows = rows[: args.max_sentences]
    loaded = len(rows)

    # 문장이 1개뿐인 stream은 이웃이 없어 거리를 계산할 수 없다. 임베딩해도
    # 쓰이지 않으므로 대상에서 뺀다. 실측 기준 418개 stream(전체의 35.6%)이
    # 여기에 해당하고, 그중 하나는 8,191토큰을 넘어 API가 거부한다.
    rows = [row for row in rows if int(row["stream_sentence_count"]) > 1]
    print(
        f"문장 {loaded:,}개 로드 -> 거리 계산 대상 {len(rows):,}개"
        f" (문장 1개 stream 제외 {loaded - len(rows):,}개)",
        flush=True,
    )

    # 임베딩을 이미 만들어 뒀으면 재사용한다. 임계값 실험에서 API를 다시 부르지
    # 않게 하는 것이 이 단계의 목적이다.
    if args.embeddings.exists() and not args.overwrite:
        vectors = np.load(args.embeddings)
        if len(vectors) != len(rows):
            raise SystemExit(
                f"저장된 임베딩 수가 문장 수와 다릅니다: {len(vectors)} != {len(rows)}"
            )
        print(f"기존 임베딩 재사용: {args.embeddings}", flush=True)
    else:
        client = OpenAI()
        codec = TiktokenCodec(MODEL, "cl100k_base")
        truncated = 0

        def embedding_input(text: str) -> str:
            """상한을 넘는 문장은 앞부분만 남긴다."""
            nonlocal truncated
            token_ids = codec.encode(text)
            if len(token_ids) <= EMBEDDING_INPUT_LIMIT:
                return text
            truncated += 1
            # 문자 기준으로 잘라 UTF-8 대체 문자가 생기지 않게 한다. 토큰당
            # 최소 1글자이므로 상한 토큰 수만큼의 글자는 항상 상한 이하다.
            return text[:EMBEDDING_INPUT_LIMIT]

        collected: list[list[float]] = []
        for start in range(0, len(rows), BATCH):
            batch = [
                embedding_input(row["text"]) for row in rows[start : start + BATCH]
            ]
            response = client.embeddings.create(model=MODEL, input=batch)
            collected.extend(item.embedding for item in response.data)
            print(
                f"임베딩: {min(start + BATCH, len(rows)):,}/{len(rows):,}",
                flush=True,
            )
        vectors = np.asarray(collected, dtype=np.float32)
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings, vectors)
        args.embeddings.chmod(0o660)
        print(f"임베딩 저장: {args.embeddings} {vectors.shape}", flush=True)
        if truncated:
            print(
                f"  입력 상한 초과로 앞부분만 임베딩한 문장: {truncated}개", flush=True
            )

    # text-embedding-3-small은 L2 정규화된 벡터를 돌려주지만 의존하지 않고
    # 직접 정규화한다. 코사인 거리 = 1 - 내적.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms

    distances: list[dict] = []
    for index in range(len(rows) - 1):
        current, following = rows[index], rows[index + 1]
        # stream 경계를 넘는 쌍은 자를 지점이 아니다.
        if current["stream_id"] != following["stream_id"]:
            continue
        similarity = float(np.dot(unit[index], unit[index + 1]))
        distances.append(
            {
                "stream_id": current["stream_id"],
                "source_id": current["source_id"],
                "left_sentence_index": current["sentence_index"],
                "right_sentence_index": following["sentence_index"],
                "boundary_char": following["char_start"],
                "distance": round(1.0 - similarity, 6),
            }
        )

    args.distances.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.distances, "wt", encoding="utf-8") as handle:
        for row in distances:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    args.distances.chmod(0o660)

    values = np.asarray([row["distance"] for row in distances], dtype=np.float64)
    percentiles = {
        f"p{int(q)}": round(float(np.percentile(values, q)), 6)
        for q in (10, 25, 50, 75, 80, 83, 85, 90, 95, 99)
    }
    report = {
        "sentences_loaded": loaded,
        "sentences_embedded": len(rows),
        "single_sentence_streams_excluded": loaded - len(rows),
        "candidate_boundaries": len(distances),
        "streams_with_boundary": len({row["stream_id"] for row in distances}),
        "distance_min": round(float(values.min()), 6),
        "distance_mean": round(float(values.mean()), 6),
        "distance_max": round(float(values.max()), 6),
        "distance_std": round(float(values.std()), 6),
        "percentiles": percentiles,
        "embedding_model": MODEL,
        "embeddings_path": str(args.embeddings),
        "distances_path": str(args.distances),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
