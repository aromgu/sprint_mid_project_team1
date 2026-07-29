from __future__ import annotations

import json
import gzip
import re
from collections import Counter
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from src.main_rag.answerability import classify_answer_status

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl_gzip(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chunk_index_summary() -> dict:
    chunks_path = PROJECT_ROOT / "data" / "main_advanced" / "chunks" / "chunks_advanced.jsonl.gz"
    report_path = PROJECT_ROOT / "reports" / "main_advanced" / "indexing_report.json"
    live_path = PROJECT_ROOT / "reports" / "main_advanced" / "live_index_status.json"
    chunks = load_jsonl_gzip(chunks_path)
    if not chunks:
        return {"status": "missing"}
    chunk_ids = [item["chunk_id"] for item in chunks]
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    dense = report.get("dense") or {}
    indexed_count = live.get("collection_count", dense.get("final_collection_count"))
    dense_current = indexed_count == len(chunks)
    section_count = sum(bool(str(item.get("section_path") or "").strip()) for item in chunks)
    requirement_count = sum(bool(re.search(r"\b[A-Z]{2,5}-\d{2,5}\b", item.get("raw_text") or "")) for item in chunks)

    return {
        "status": "ready" if dense_current else "stale_index",
        "chunk_count": len(chunks),
        "average_tokens": round(sum(item["token_count"] for item in chunks) / len(chunks), 1),
        "heading_preservation": round(section_count / len(chunks), 4),
        "requirement_preservation": round(requirement_count / len(chunks), 4),
        "embedding_missing": 0 if dense_current else len(chunks),
        "duplicate_chunk_ids": sum(count - 1 for count in Counter(chunk_ids).values() if count > 1),
        "bm25_index_current": False,
        "dense_index_current": dense_current,
        "collection_name": live.get("collection_name", dense.get("collection_name")),
        "embedding_model": live.get("embedding_model", dense.get("embedding_model")),
        "document_count": (report.get("source_document_count") or 0) + (live.get("upload_document_count") or 0),
    }


def golden_v3_evaluation() -> dict | None:
    base = PROJECT_ROOT / "reports" / "main_advanced"
    golden_path = PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl"
    answers_path = base / "answers_top10.jsonl"
    ragas_path = base / "ragas_top10.json"
    chunks_path = PROJECT_ROOT / "data" / "main_advanced" / "chunks" / "chunks_advanced.jsonl.gz"
    if not (golden_path.exists() and answers_path.exists() and ragas_path.exists()):
        return None
    golden = {item["question_id"]: item for item in load_jsonl(golden_path)}
    answers = {item["question_id"]: item for item in load_jsonl(answers_path)}
    chunks = {item["chunk_id"]: item for item in load_jsonl_gzip(chunks_path)}
    ragas_payload = json.loads(ragas_path.read_text(encoding="utf-8"))
    scores = {item["question_id"]: item for item in ragas_payload.get("details", [])}
    details = []
    for question_id, item in golden.items():
        answer = answers.get(question_id, {})
        score = scores.get(question_id, {})
        answer_status = answer.get("answer_status") or (
            "answered" if answer.get("is_answerable") else "unanswerable"
        )
        # Older result files can contain an `answered` label even when the answer
        # explicitly says that one requested fact could not be confirmed. Apply the
        # current classifier when serving historical results without rewriting the
        # immutable evaluation artifact.
        if answer_status == "answered":
            detected_status = classify_answer_status(
                answer.get("answer", ""), answer.get("citations", [])
            )
            if detected_status != "answered":
                answer_status = detected_status
        citations = []
        for citation in answer.get("citations", []):
            chunk = chunks.get(citation.get("chunk_id"), {})
            citations.append({**citation, "excerpt": (chunk.get("raw_text") or chunk.get("embedding_text") or "")[:1200]})
        details.append({
            "question_id": question_id,
            "question": item.get("question", ""),
            "source_document": item.get("source_document", ""),
            "difficulty": item.get("difficulty"),
            "query_type": item.get("query_type"),
            "expected_answerable": item.get("answerable"),
            "answer_status": answer_status,
            "predicted_answerable": answer_status != "unanswerable",
            "ground_truth": item.get("ground_truth", ""),
            "answer": answer.get("answer", ""),
            "citations": citations,
            "retrieved_chunk_ids": answer.get("retrieved_chunk_ids", []),
            "faithfulness": score.get("faithfulness"),
            "answer_relevancy": score.get("answer_relevancy"),
        })
    low_faithfulness = sum((row["faithfulness"] or 0) < 0.5 for row in details)
    low_relevancy = sum((row["answer_relevancy"] or 0) < 0.3 for row in details)
    false_rejections = sum(
        row["expected_answerable"] is True and row["predicted_answerable"] is False
        for row in details
    )
    no_citations = sum(not row["citations"] for row in details)
    relevancy_by_status = {}
    for status in ("answered", "partially_answered", "unanswerable"):
        values = [
            row["answer_relevancy"]
            for row in details
            if row["answer_status"] == status and isinstance(row["answer_relevancy"], (int, float))
        ]
        relevancy_by_status[status] = {
            "count": sum(row["answer_status"] == status for row in details),
            "scored_count": len(values),
            "average": sum(values) / len(values) if values else None,
        }
    latencies = sorted(
        float(item.get("search_latency_ms") or 0) + float(item.get("generation_latency_ms") or 0)
        for item in answers.values()
    )
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95)) if latencies else 0
    return {
        "run": {
            "name": "Main Advanced RAG · Golden Set v3 · Top-10",
            "generation_model": answers[next(iter(answers))].get("model") if answers else None,
            "evaluation_model": ragas_payload.get("model"),
            "question_count": len(details),
            "answered_question_count": len(answers),
            "scored_question_count": len(scores),
            "status": "complete" if len(details) == len(answers) == len(scores) else "partial",
        },
        "summary": {
            **ragas_payload.get("summary", {}),
            "answer_relevancy_by_status": relevancy_by_status,
        },
        "low_score_counts": {
            "faithfulness_below_05": low_faithfulness,
            "relevancy_below_03": low_relevancy,
            "false_rejections": false_rejections,
            "no_citations": no_citations,
        },
        "e2e": {
            "answerability_accuracy": sum(
                row["expected_answerable"] == row["predicted_answerable"] for row in details
            ) / len(details) if details else None,
            "citation_rate": sum(bool(row["citations"]) for row in details) / len(details) if details else None,
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "p95_latency_ms": latencies[p95_index] if latencies else None,
            "estimated_cost_usd": sum(float(item.get("estimated_cost_usd") or 0) for item in answers.values()),
        },
        "details": details,
        "report_url": "/api/evaluation/report/main-advanced-p1",
    }


@router.get("/summary")
def summary():
    v3_retrieval_path = PROJECT_ROOT / "reports" / "main_advanced" / "retrieval_summary.json"
    v3_answer_path = PROJECT_ROOT / "reports" / "main_advanced" / "answer_summary_top10.json"
    v3_ragas = golden_v3_evaluation()
    chunk_index = chunk_index_summary()
    return {
        "status": "ready" if v3_retrieval_path.exists() and v3_answer_path.exists() and v3_ragas else "not_run",
        "chunk_index": chunk_index,
        "retrieval": [],
        "ragas": [],
        "golden_v3_retrieval": [json.loads(v3_retrieval_path.read_text(encoding="utf-8"))] if v3_retrieval_path.exists() else [],
        "golden_v3_answers": json.loads(v3_answer_path.read_text(encoding="utf-8"))["summary"] if v3_answer_path.exists() else None,
        "golden_v3_ragas": v3_ragas,
    }


@router.get("/report/ragas-1st", response_class=PlainTextResponse)
def ragas_first_report():
    path = PROJECT_ROOT / "reports" / "evaluation_v3" / "ragas_1st_score_analysis.md"
    if not path.exists():
        return "# RAGAS 1차 분석\n\n보고서가 아직 생성되지 않았습니다."
    return path.read_text(encoding="utf-8")


@router.get("/report/main-advanced-p1", response_class=PlainTextResponse)
def main_advanced_report():
    path = PROJECT_ROOT / "reports" / "main_advanced" / "P1_REPORT.md"
    if not path.exists():
        return "# Main Advanced P1\n\n보고서가 아직 생성되지 않았습니다."
    return path.read_text(encoding="utf-8")
