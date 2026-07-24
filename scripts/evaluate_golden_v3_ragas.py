"""Run LLM-based RAGAS metrics over the already generated Golden Set v3 answers."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

from src.search.service import PROJECT_ROOT


def _install_ragas_vertex_compatibility_shim() -> None:
    """RAGAS 0.4.3 imports a removed optional LangChain VertexAI module."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - never instantiated in OpenAI evals
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Golden Set v3 answers with RAGAS/OpenAI.")
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl")
    parser.add_argument("--answers", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "answers.jsonl")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "eval_corpus_v3" / "processed" / "chunks.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "ragas.json")
    parser.add_argument("--details-output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "ragas_details.jsonl")
    parser.add_argument("--model", default=os.getenv("RAGAS_MODEL", "gpt-5-nano"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-null", action="store_true")
    parser.add_argument("--metrics", choices=("all", "faithfulness", "answer_relevancy"), default="all")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    golden = {
        row["question_id"]: row
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line))
    }
    answers = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = {
        row["chunk_id"]: row["text"]
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line))
    }
    if len(answers) != len(golden):
        raise SystemExit(f"Expected {len(golden)} answers, found {len(answers)}")

    rows = []
    for answer in answers:
        item = golden[answer["question_id"]]
        cited_ids = [citation["chunk_id"] for citation in answer.get("citations", [])]
        context_ids = cited_ids or answer["retrieved_chunk_ids"][:3]
        selected_contexts = []
        remaining_chars = 8_000
        for chunk_id in context_ids:
            text = chunks.get(chunk_id, "")
            if not text or remaining_chars <= 0:
                continue
            selected_contexts.append(text[:remaining_chars])
            remaining_chars -= len(selected_contexts[-1])
        rows.append({
            "question_id": answer["question_id"],
            "user_input": item["question"],
            "response": answer["answer"],
            "retrieved_contexts": selected_contexts,
            "reference": item["ground_truth"],
        })

    _install_ragas_vertex_compatibility_shim()
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(ChatOpenAI(model=args.model, temperature=0, timeout=args.timeout, max_retries=2))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metric_names = ["faithfulness", "answer_relevancy"]
    selected_names = metric_names if args.metrics == "all" else [args.metrics]
    metric_factories = {"faithfulness": Faithfulness, "answer_relevancy": ResponseRelevancy}
    completed = {}
    if args.resume and args.details_output.exists():
        completed = {
            item["question_id"]: item
            for line in args.details_output.read_text(encoding="utf-8").splitlines()
            if line.strip() and (item := json.loads(line))
        }
    if args.retry_null:
        pending = [
            row for row in rows
            if row["question_id"] not in completed
            or any(completed[row["question_id"]].get(name) is None for name in selected_names)
        ]
    else:
        pending = [row for row in rows if row["question_id"] not in completed]
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.details_output.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            dataset = Dataset.from_list([{k: v for k, v in row.items() if k != "question_id"} for row in batch])
            result = evaluate(
                dataset,
                metrics=[metric_factories[name]() for name in selected_names],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(timeout=args.timeout, max_retries=3, max_wait=10, max_workers=args.max_workers),
                raise_exceptions=False,
                show_progress=False,
            )
            raw_records = result.to_pandas().to_dict(orient="records")
            for raw, source in zip(raw_records, batch, strict=True):
                record = dict(completed.get(source["question_id"], {"question_id": source["question_id"]}))
                for name in selected_names:
                    value = raw.get(name)
                    record[name] = float(value) if value is not None and math.isfinite(float(value)) else None
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[record["question_id"]] = record
            print(f"completed {min(start + len(batch), len(pending))}/{len(pending)}; total={len(completed)}/{len(rows)}", flush=True)
    records = [completed[row["question_id"]] for row in rows if row["question_id"] in completed]
    summary = {
        name: sum(row[name] for row in records if row.get(name) is not None) /
        sum(row.get(name) is not None for row in records)
        for name in metric_names
        if any(row.get(name) is not None for row in records)
    }
    payload = {"model": args.model, "evaluated_questions": len(records), "summary": summary, "details": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload | {"details": f"{len(records)} rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
