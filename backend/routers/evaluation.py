from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from src.search.loader import file_fingerprint

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_index_summary() -> dict:
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    pages_path = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
    chunks = load_jsonl(chunks_path)
    pages = load_jsonl(pages_path)
    if not chunks:
        return {"status": "missing"}

    chunk_ids = [item["chunk_id"] for item in chunks]
    expected_headings = {
        (page["document_id"], heading)
        for page in pages for heading in page.get("headings", []) if heading.strip()
    }
    indexed_headings = {
        (chunk["document_id"], heading)
        for chunk in chunks for heading in chunk.get("section_path", []) if heading.strip()
    }
    expected_requirements = {
        (page["document_id"], requirement)
        for page in pages for requirement in page.get("requirement_ids", [])
    }
    indexed_requirements = {
        (chunk["document_id"], requirement)
        for chunk in chunks for requirement in chunk.get("requirement_ids", [])
    }
    fingerprint = file_fingerprint(chunks_path)
    index_meta = []
    for path in (PROJECT_ROOT / "data" / "indexes").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "chunks_sha256" in payload and "chunk_count" in payload:
            index_meta.append({"name": path.stem, **payload})
    current_indexes = [
        item for item in index_meta
        if item["chunks_sha256"] == fingerprint and item["chunk_count"] == len(chunks)
    ]
    dense_current = any(item["name"].startswith("dense_") for item in current_indexes)
    bm25_current = any(item["name"].startswith("bm25_") for item in current_indexes)

    return {
        "status": "ready" if dense_current and bm25_current else "stale_index",
        "chunk_count": len(chunks),
        "average_tokens": round(sum(item["token_count"] for item in chunks) / len(chunks), 1),
        "heading_preservation": round(len(expected_headings & indexed_headings) / len(expected_headings), 4) if expected_headings else None,
        "requirement_preservation": round(len(expected_requirements & indexed_requirements) / len(expected_requirements), 4) if expected_requirements else None,
        "embedding_missing": 0 if dense_current else len(chunks),
        "duplicate_chunk_ids": sum(count - 1 for count in Counter(chunk_ids).values() if count > 1),
        "bm25_index_current": bm25_current,
        "dense_index_current": dense_current,
    }


def golden_v3_evaluation() -> dict | None:
    base = PROJECT_ROOT / "reports" / "evaluation_v3"
    golden_path = PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl"
    answers_path = base / "answers.jsonl"
    ragas_path = base / "ragas.json"
    chunks_path = PROJECT_ROOT / "data" / "eval_corpus_v3" / "processed" / "chunks.jsonl"
    if not (golden_path.exists() and answers_path.exists() and ragas_path.exists()):
        return None
    golden = {item["question_id"]: item for item in load_jsonl(golden_path)}
    answers = {item["question_id"]: item for item in load_jsonl(answers_path)}
    chunks = {item["chunk_id"]: item for item in load_jsonl(chunks_path)}
    ragas_payload = json.loads(ragas_path.read_text(encoding="utf-8"))
    scores = {item["question_id"]: item for item in ragas_payload.get("details", [])}
    details = []
    for question_id, item in golden.items():
        answer = answers.get(question_id, {})
        score = scores.get(question_id, {})
        citations = []
        for citation in answer.get("citations", []):
            chunk = chunks.get(citation.get("chunk_id"), {})
            citations.append({**citation, "excerpt": chunk.get("text", "")[:1200]})
        details.append({
            "question_id": question_id,
            "question": item.get("question", ""),
            "source_document": item.get("source_document", ""),
            "difficulty": item.get("difficulty"),
            "query_type": item.get("query_type"),
            "expected_answerable": item.get("answerable"),
            "predicted_answerable": answer.get("is_answerable"),
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
    latencies = sorted(
        float(item.get("search_latency_ms") or 0) + float(item.get("generation_latency_ms") or 0)
        for item in answers.values()
    )
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95)) if latencies else 0
    return {
        "run": {
            "name": "Golden Set v3 · 1차 평가",
            "generation_model": answers[next(iter(answers))].get("model") if answers else None,
            "evaluation_model": ragas_payload.get("model"),
            "question_count": len(details),
            "answered_question_count": len(answers),
            "scored_question_count": len(scores),
            "status": "complete" if len(details) == len(answers) == len(scores) else "partial",
        },
        "summary": ragas_payload.get("summary", {}),
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
        "report_url": "/api/evaluation/report/ragas-1st",
    }


@router.get("/summary")
def summary():
    path = PROJECT_ROOT / "reports" / "retrieval" / "summary.json"
    ragas_path = PROJECT_ROOT / "reports" / "evaluation" / "ragas.json"
    v3_retrieval_path = PROJECT_ROOT / "reports" / "evaluation_v3" / "retrieval_summary.json"
    v3_answer_path = PROJECT_ROOT / "reports" / "evaluation_v3" / "answer_summary.json"
    v3_ragas = golden_v3_evaluation()
    chunk_index = chunk_index_summary()
    return {
        "status": "ready" if path.exists() or ragas_path.exists() or v3_retrieval_path.exists() or chunk_index.get("status") == "ready" else "not_run",
        "chunk_index": chunk_index,
        "retrieval": json.loads(path.read_text(encoding="utf-8")) if path.exists() else [],
        "ragas": json.loads(ragas_path.read_text(encoding="utf-8")) if ragas_path.exists() else [],
        "golden_v3_retrieval": json.loads(v3_retrieval_path.read_text(encoding="utf-8")) if v3_retrieval_path.exists() else [],
        "golden_v3_answers": json.loads(v3_answer_path.read_text(encoding="utf-8"))["summary"] if v3_answer_path.exists() else None,
        "golden_v3_ragas": v3_ragas,
    }


@router.get("/report/ragas-1st", response_class=PlainTextResponse)
def ragas_first_report():
    path = PROJECT_ROOT / "reports" / "evaluation_v3" / "ragas_1st_score_analysis.md"
    if not path.exists():
        return "# RAGAS 1차 분석\n\n보고서가 아직 생성되지 않았습니다."
    return path.read_text(encoding="utf-8")
