"""Generate Golden Set v3 answers. This calls OpenAI and incurs API cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext

import weave
from pathlib import Path

from src.evaluation.golden_v3 import GoldenV3Item
from src.generation.openai_generator import OpenAIRAGService
from src.search.service import PROJECT_ROOT, SearchService
from src.observability.wandb import init_wandb_run


def rate_limit_retry_delay(error: Exception, fallback: float) -> float | None:
    message = str(error)
    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
        return None
    matches = re.findall(r"(?:retryDelay['\": ]+|retry in )(\d+(?:\.\d+)?)s?", message, re.IGNORECASE)
    return max(float(value) for value in matches) + 1.0 if matches else fallback


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproducibility_config(args, service: SearchService, generator: OpenAIRAGService) -> dict:
    tracked_files = {
        "golden": args.golden,
        "source_map": args.source_map,
        "chunks": service.chunks_path,
        "search_config": args.config,
        "generation_config": PROJECT_ROOT / "configs" / "generation.yaml",
        "dependency_lock": PROJECT_ROOT / "uv.lock",
    }
    return {
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
        "execution": {
            "provider": args.provider,
            "model": generator.model_name_for(args.provider),
            "resume": args.resume,
            "limit": args.limit,
            "request_interval_seconds": args.request_interval,
            "max_retries": args.max_retries,
        },
        "search": service.config,
        "generation": generator.config,
        "inputs": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
            for name, path in tracked_files.items()
        },
    }


def log_reproducibility_artifact(run, args, service: SearchService) -> None:
    import wandb

    artifact = wandb.Artifact(
        name=f"golden-v3-run-{os.getenv('GITHUB_RUN_ID', run.id)}",
        type="rag-evaluation-run",
        metadata={
            "git_sha": os.getenv("GITHUB_SHA"),
            "provider": args.provider,
            "answer_count": sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
            if args.output.exists() else 0,
        },
    )
    files = [
        args.golden,
        args.source_map,
        args.config,
        PROJECT_ROOT / "configs" / "search.yaml",
        PROJECT_ROOT / "configs" / "generation.yaml",
        PROJECT_ROOT / "configs" / "ingestion.yaml",
        PROJECT_ROOT / "uv.lock",
        service.chunks_path,
        *sorted(service.index_dir.glob("*")),
        args.output,
    ]
    for path in files:
        if path.exists() and path.is_file():
            artifact.add_file(str(path), name=str(path.relative_to(PROJECT_ROOT)))
    run.log_artifact(artifact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl")
    parser.add_argument("--source-map", type=Path, default=PROJECT_ROOT / "data" / "eval_corpus_v3" / "source_map.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "search_eval_v3.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "answers.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional cost-control limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", choices=("openai", "gemini", "gemini-lite"), default="openai")
    parser.add_argument("--wandb", action="store_true", help="Log batch metrics to W&B")
    parser.add_argument("--wandb-name", help="Optional W&B run name")
    parser.add_argument("--request-interval", type=float, default=0.0, help="Minimum seconds between LLM requests")
    parser.add_argument("--max-retries", type=int, default=0, help="Retries per question after a 429 response")
    args = parser.parse_args()

    items = [GoldenV3Item.from_dict(json.loads(line)) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    completed = set()
    if args.resume and args.output.exists():
        completed = {json.loads(line)["question_id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [item for item in items if item.question_id not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    service = SearchService(args.config)
    generator = OpenAIRAGService(search_service=service)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    run_config = reproducibility_config(args, service, generator)
    run_config["execution"]["questions"] = len(pending)
    tracking = init_wandb_run(
        job_type="golden-v3-answers", name=args.wandb_name, project_root=PROJECT_ROOT,
        config=run_config,
    ) if args.wandb else nullcontext(None)
    with tracking as run, args.output.open(mode, encoding="utf-8") as handle:
        @weave.op(name="golden_v3.answer")
        def traced_answer(question: str, document_id: str):
            return generator.answer(question, document_ids={document_id}, provider=args.provider)

        totals = {"completed": 0, "answerable": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        try:
            last_request_at: float | None = None
            for index, item in enumerate(pending, 1):
                document_id = source_map[item.source_document]["document_id"]
                for attempt in range(args.max_retries + 1):
                    if last_request_at is not None and args.request_interval > 0:
                        remaining = args.request_interval - (time.monotonic() - last_request_at)
                        if remaining > 0:
                            time.sleep(remaining)
                    last_request_at = time.monotonic()
                    try:
                        result = traced_answer(item.question, document_id) if run else generator.answer(
                            item.question, document_ids={document_id}, provider=args.provider
                        )
                        break
                    except Exception as error:
                        delay = rate_limit_retry_delay(error, fallback=min(60.0 * (attempt + 1), 300.0))
                        if delay is None or attempt >= args.max_retries:
                            raise
                        print(
                            f"429 rate limit for {item.question_id}; retry "
                            f"{attempt + 1}/{args.max_retries} in {delay:.1f}s",
                            flush=True,
                        )
                        time.sleep(delay)
                        last_request_at = None
                row = {
                    "question_id": item.question_id,
                    "question": item.question,
                    "source_document": item.source_document,
                    "answer": result.answer,
                    "is_answerable": result.is_answerable,
                    "caveat": result.caveat,
                    "citations": [citation.model_dump() for citation in result.citations],
                    "retrieved_chunk_ids": result.retrieved_chunk_ids,
                    "retriever": result.retriever,
                    "model": result.model,
                    "search_latency_ms": result.search_latency_ms,
                    "generation_latency_ms": result.generation_latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "estimated_cost_usd": result.estimated_cost_usd,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                totals["completed"] = index
                totals["answerable"] += int(result.is_answerable)
                totals["input_tokens"] += result.input_tokens or 0
                totals["output_tokens"] += result.output_tokens or 0
                totals["cost"] += result.estimated_cost_usd or 0
                if run:
                    run.log({
                        "progress/completed": index,
                        "answer/is_answerable": int(result.is_answerable),
                        "latency/search_ms": result.search_latency_ms,
                        "latency/generation_ms": result.generation_latency_ms,
                        "tokens/input": result.input_tokens or 0,
                        "tokens/output": result.output_tokens or 0,
                        "cost/estimated_usd": result.estimated_cost_usd or 0,
                    }, step=index)
                print(f"{index}/{len(pending)} {item.question_id} answerable={result.is_answerable}", flush=True)
        finally:
            if run:
                run.summary["completed_questions"] = totals["completed"]
                run.summary["answerable_questions"] = totals["answerable"]
                run.summary["total_input_tokens"] = totals["input_tokens"]
                run.summary["total_output_tokens"] = totals["output_tokens"]
                run.summary["total_estimated_cost_usd"] = totals["cost"]
                log_reproducibility_artifact(run, args, service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
