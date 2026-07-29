"""Single-call RAGAS-compatible faithfulness/relevance judge for Golden Set v3."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from statistics import fmean

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.search.service import PROJECT_ROOT


class JudgeResult(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    faithfulness_reason: str
    answer_relevancy_reason: str


INSTRUCTIONS = """RAG 평가 심사자다. 제공된 질문, 답변, 검색 근거만 평가한다.
faithfulness는 답변의 사실적 주장 중 검색 근거가 뒷받침하는 비율을 0~1로 채점한다.
answer_relevancy는 답변이 질문에 직접적이고 빠짐없이 대응하는 정도를 0~1로 채점한다.
근거에 없는 사실을 상식으로 보완하지 않는다. 답변 거절이 적절하면 관련성을 인정하되,
근거가 충분한데도 거절했다면 관련성을 낮춘다. 이유는 한국어 한 문장으로 쓴다."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=PROJECT_ROOT / "reports" / "main_advanced" / "answers.jsonl")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "main_advanced" / "chunks" / "chunks_advanced.jsonl.gz")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "main_advanced" / "llm_judge_details.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports" / "main_advanced" / "llm_judge.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env"); load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    from openai import OpenAI

    answers = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        answers = answers[:args.limit]
    opener = gzip.open if args.chunks.suffix == ".gz" else open
    with opener(args.chunks, "rt", encoding="utf-8") as stream:
        chunks = {
            row["chunk_id"]: row.get("raw_text") or row.get("embedding_text") or row.get("text", "")
            for line in stream if line.strip() and (row := json.loads(line))
        }
    completed = {}
    if args.resume and args.output.exists():
        completed = {row["question_id"]: row for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip() and (row := json.loads(line))}
    client = OpenAI(timeout=90)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        pending = [answer for answer in answers if answer["question_id"] not in completed]
        for index, answer in enumerate(pending, 1):
            cited = [citation["chunk_id"] for citation in answer.get("citations", [])]
            context_ids = cited or answer["retrieved_chunk_ids"][:3]
            context = "\n\n".join(chunks[cid] for cid in context_ids if cid in chunks)[:8_000]
            response = client.responses.parse(
                model=args.model,
                max_output_tokens=500,
                instructions=INSTRUCTIONS,
                input=f"질문:\n{answer['question']}\n\n답변:\n{answer['answer']}\n\n검색 근거:\n{context}",
                text_format=JudgeResult,
            )
            score = response.output_parsed
            if score is None:
                raise RuntimeError(f"No parsed judge output: {answer['question_id']}")
            row = {"question_id": answer["question_id"], "model": args.model, **score.model_dump()}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
            completed[row["question_id"]] = row
            print(f"{index}/{len(pending)} {row['question_id']}")
    rows = list(completed.values())
    summary = {
        "method": "single-call RAGAS-compatible LLM judge (not official ragas package output)",
        "model": args.model,
        "evaluated_questions": len(rows),
        "faithfulness": fmean(row["faithfulness"] for row in rows),
        "answer_relevancy": fmean(row["answer_relevancy"] for row in rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
