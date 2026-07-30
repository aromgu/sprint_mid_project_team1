"""저장된 문장·거리 파일을 실제 stream에 연결하는 계약을 검증한다."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.run_semantic_chunking import load_semantic_boundaries
from src.chunking.advanced_chunking import AdvancedTextStream


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """테스트용 gzip JSONL을 만든다."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False))
            stream.write("\n")


def _stream() -> AdvancedTextStream:
    """3개 문장 span으로 나눌 수 있는 최소 텍스트 stream을 만든다."""
    source_id = "source-001"
    return AdvancedTextStream(
        stream_id=f"{source_id}:AS000001",
        stream_order=1,
        boundary_type="hwp_section_path",
        boundary_id=f"{source_id}:section:0:paragraph:1",
        blocks=({"source_id": source_id, "block_id": f"{source_id}:B000001"},),
        text="가나다라마바사",
        block_char_spans=(),
    )


def _sentence_rows(stream: AdvancedTextStream) -> list[dict]:
    """현재 stream을 2·2·3글자 문장으로 나타낸 덤프 행을 만든다."""
    spans = [(0, 2), (2, 4), (4, 7)]
    return [
        {
            "source_id": "source-001",
            "stream_id": stream.stream_id,
            "sentence_index": index,
            "char_start": start,
            "char_end": end,
            "text": stream.text[start:end],
            "stream_sentence_count": len(spans),
        }
        for index, (start, end) in enumerate(spans, start=1)
    ]


def _distance_rows(stream: AdvancedTextStream) -> list[dict]:
    """첫 경계만 p83을 넘는 인접 문장 거리 두 건을 만든다."""
    return [
        {
            "source_id": "source-001",
            "stream_id": stream.stream_id,
            "left_sentence_index": 1,
            "right_sentence_index": 2,
            "boundary_char": 2,
            "distance": 0.8,
        },
        {
            "source_id": "source-001",
            "stream_id": stream.stream_id,
            "left_sentence_index": 2,
            "right_sentence_index": 3,
            "boundary_char": 4,
            "distance": 0.6,
        },
    ]


def test_load_semantic_boundaries_selects_only_distances_above_threshold(
    tmp_path: Path,
) -> None:
    """임계값과 같은 값은 제외하고 더 큰 거리의 왼쪽 문장만 선택한다."""
    stream = _stream()
    sentences = tmp_path / "sentences.jsonl.gz"
    distances = tmp_path / "distances.jsonl.gz"
    _write_gzip_jsonl(sentences, _sentence_rows(stream))
    _write_gzip_jsonl(distances, _distance_rows(stream))

    loaded = load_semantic_boundaries(
        sentences,
        distances,
        0.706,
        {stream.stream_id: stream},
    )

    assert loaded.cut_after_by_stream == {stream.stream_id: frozenset({1})}
    assert loaded.sentence_count == 3
    assert loaded.candidate_boundary_count == 2
    assert loaded.selected_boundary_count == 1
    assert loaded.streams_with_selected_boundary == 1


def test_load_semantic_boundaries_rejects_stale_sentence_spans(
    tmp_path: Path,
) -> None:
    """문장 덤프가 현재 stream을 연속해서 덮지 않으면 재사용하지 않는다."""
    stream = _stream()
    rows = _sentence_rows(stream)
    rows[1]["char_start"] = 3
    sentences = tmp_path / "sentences.jsonl.gz"
    distances = tmp_path / "distances.jsonl.gz"
    _write_gzip_jsonl(sentences, rows)
    _write_gzip_jsonl(distances, _distance_rows(stream))

    with pytest.raises(ValueError, match="문장 문자 span"):
        load_semantic_boundaries(
            sentences,
            distances,
            0.706,
            {stream.stream_id: stream},
        )


def test_load_semantic_boundaries_rejects_missing_adjacent_distance(
    tmp_path: Path,
) -> None:
    """다문장 stream의 인접 거리 하나라도 빠지면 불완전 파일로 중단한다."""
    stream = _stream()
    sentences = tmp_path / "sentences.jsonl.gz"
    distances = tmp_path / "distances.jsonl.gz"
    _write_gzip_jsonl(sentences, _sentence_rows(stream))
    _write_gzip_jsonl(distances, _distance_rows(stream)[:1])

    with pytest.raises(ValueError, match="stream 거리 수"):
        load_semantic_boundaries(
            sentences,
            distances,
            0.706,
            {stream.stream_id: stream},
        )
