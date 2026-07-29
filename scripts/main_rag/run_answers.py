"""Generate resumable Golden v3 answers with one fresh BidMate session per item."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.golden_v3 import GoldenV3Item
from src.main_rag.generation.generate_answer import BidMateRAGSession
from src.main_rag.generation.gemini_session import GeminiBidMateRAGSession
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.service import MainAdvancedRAGService
from src.main_rag.settings import load_settings
from src.main_rag.answerability import classify_answer_status, is_answerable_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def retry_delay(error: Exception, attempt: int) -> float | None:
    message = str(error)
    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
        return None
    matches = re.findall(r"(?:retryDelay['\": ]+|retry in )(\d+(?:\.\d+)?)s?", message, re.I)
    return max(map(float, matches)) + 1 if matches else min(30.0 * (attempt + 1), 180.0)


async def generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    provider = str(settings.get("generation", "provider", "gemini"))
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"{key_name}가 설정되지 않았습니다")
    items = [
        GoldenV3Item.from_dict(json.loads(line))
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    completed = set()
    if args.resume and args.output.exists():
        existing = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        successful = [row for row in existing if not row.get("error")]
        completed = {row["question_id"] for row in successful}
        # Failed rows remain pending and are removed before appending a retry.
        if len(successful) != len(existing):
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in successful),
                encoding="utf-8",
            )
            temporary.replace(args.output)
    pending = [item for item in items if item.question_id not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    retriever = AdvancedRetriever(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def generate_one(item: GoldenV3Item) -> tuple[dict, bool]:
        document_id = source_map[item.source_document]["document_id"]
        # Every task owns a fresh session, so concurrent questions never share
        # response IDs or conversation history.
        session_class = GeminiBidMateRAGSession if provider == "gemini" else BidMateRAGSession
        session = session_class(
            api_key=api_key,
            model=str(settings.get("generation", "model", "gpt-5-nano")),
            max_context_chars=int(settings.get("generation", "max_context_chars", 7000)),
            max_docs=int(settings.get("generation", "max_docs", 6)),
        )
        service = MainAdvancedRAGService(settings=settings, retriever=retriever, session=session)
        for attempt in range(args.max_retries + 1):
            try:
                result = await service.answer(item.question, document_id=document_id, top_k=args.top_k)
                break
            except Exception as error:
                message = str(error)
                if "GenerateRequestsPerDay" in message or "free_tier_requests" in message:
                    raise RuntimeError("DAILY_QUOTA_EXHAUSTED") from error
                delay = retry_delay(error, attempt)
                if delay is None or attempt >= args.max_retries:
                    result = {"error": f"{type(error).__name__}: {error}"}
                    break
                await asyncio.sleep(delay)
        evidence = result.get("evidence") or []
        usage = result.get("_usage") or {}
        answer = str(result.get("answer") or "")
        answer_status = classify_answer_status(
            answer, evidence, needs_clarification=bool(result.get("needs_clarification")),
        )
        is_answerable = is_answerable_status(answer_status)
        row = {
            "question_id": item.question_id,
            "question": item.question,
            "source_document": item.source_document,
            "document_id": document_id,
            "answer": answer,
            "is_answerable": is_answerable,
            "answer_status": answer_status,
            "caveat": result.get("clarification_question"),
            "citations": [
                {
                    "source_id": str(index + 1), "chunk_id": str(value.get("chunk_id")),
                    "document_id": document_id, "document_name": value.get("source"),
                    "page_start": value.get("page"), "page_end": value.get("page"),
                    "quote": value.get("quote"), "score": value.get("score"),
                }
                for index, value in enumerate(evidence)
            ],
            "retrieved_chunk_ids": result.get("retrieved_chunk_ids") or [],
            "rejected_evidence": result.get("rejected_evidence") or [],
            "retriever": "main_advanced_dense",
            "model": settings.get("generation", "model", "gpt-5-nano"),
            "search_latency_ms": (result.get("latency") or {}).get("retrieval_seconds", 0) * 1000,
            "generation_latency_ms": (result.get("latency") or {}).get("generation_seconds", 0) * 1000,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "estimated_cost_usd": None, "error": result.get("error"),
        }
        return row, is_answerable

    semaphore = asyncio.Semaphore(args.max_workers)

    async def bounded(item: GoldenV3Item) -> tuple[dict, bool]:
        async with semaphore:
            return await generate_one(item)

    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        tasks = [asyncio.create_task(bounded(item)) for item in pending]
        try:
            for index, task in enumerate(asyncio.as_completed(tasks), 1):
                row, is_answerable = await task
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{index}/{len(pending)} {row['question_id']} answerable={is_answerable}", flush=True)
                if args.request_interval:
                    await asyncio.sleep(args.request_interval)
        except RuntimeError as error:
            if str(error) != "DAILY_QUOTA_EXHAUSTED":
                raise
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("Gemini 일일 quota 소진; --resume 필요", flush=True)
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset/golden_set_v3.jsonl")
    parser.add_argument("--source-map", type=Path, default=PROJECT_ROOT / "data/eval_corpus_v3/source_map.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/main_advanced/answers.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=4)
    return asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
