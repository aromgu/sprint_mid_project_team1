"""Generate Golden Set v3 answers. This calls OpenAI and incurs API cost."""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import weave
from pathlib import Path

from src.evaluation.golden_v3 import GoldenV3Item
from src.generation.openai_generator import OpenAIRAGService
from src.search.service import PROJECT_ROOT, SearchService
from src.observability.wandb import init_wandb_run


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
    tracking = init_wandb_run(
        job_type="golden-v3-answers", name=args.wandb_name, project_root=PROJECT_ROOT,
        config={"provider": args.provider, "model": generator.model_name_for(args.provider), "questions": len(pending), "resume": args.resume},
    ) if args.wandb else nullcontext(None)
    with tracking as run, args.output.open(mode, encoding="utf-8") as handle:
        @weave.op(name="golden_v3.answer")
        def traced_answer(question: str, document_id: str):
            return generator.answer(question, document_ids={document_id}, provider=args.provider)

        for index, item in enumerate(pending, 1):
            document_id = source_map[item.source_document]["document_id"]
            result = traced_answer(item.question, document_id) if run else generator.answer(
                item.question, document_ids={document_id}, provider=args.provider
            )
            row = {
                "question_id": item.question_id,
                "question": item.question,
                "source_document": item.source_document,
                "answer": result.answer,
                "is_answerable": result.is_answerable,
                "caveat": result.caveat,
                "citations": [citation.model_dump() for citation in result.citations],
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "model": result.model,
                "search_latency_ms": result.search_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
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
            print(f"{index}/{len(pending)} {item.question_id} answerable={result.is_answerable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
