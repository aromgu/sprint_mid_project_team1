"""Run RAGAS metrics over a Golden set.

This intentionally calls the configured OpenAI model for each question and can incur cost.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.generation.openai_generator import OpenAIRAGService
from src.search.service import SearchService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation/ragas.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env"); load_dotenv(root / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for RAGAS evaluation")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except Exception as exc:
        raise SystemExit("RAGAS dependencies are incomplete; install the project's full dependencies (including langchain-google-vertexai): " + str(exc)) from exc

    search = SearchService(); generator = OpenAIRAGService(search_service=search)
    rows = []
    for line in args.golden.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        item = json.loads(line); doc_ids = set(item.get("reference_document_ids", [])) or None
        results = search.search(item["question"], top_k=args.top_k, document_ids=doc_ids)
        answer = generator.answer(item["question"], document_ids=doc_ids)
        rows.append({"user_input": item["question"], "response": answer.answer, "retrieved_contexts": [r.chunk.text for r in results], "reference": item.get("reference_answer"), "reference_contexts": item.get("reference_context_ids", [])})
    dataset = Dataset.from_list(rows)
    llm = LangchainLLMWrapper(ChatOpenAI(model=os.getenv("RAGAS_MODEL", "gpt-5-nano"), temperature=0))
    result = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()], llm=llm, raise_exceptions=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RAGAS result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
