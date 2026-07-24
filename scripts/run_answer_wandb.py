from __future__ import annotations

import argparse

import weave

from src.generation.openai_generator import OpenAIRAGService
from src.observability.wandb import init_wandb_run
from src.search.service import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one grounded answer and log it to W&B.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--document-id")
    parser.add_argument("--provider", choices=("openai", "gemini", "gemini-lite"), default="openai")
    parser.add_argument("--name", help="Optional W&B run name")
    args = parser.parse_args()

    rag = OpenAIRAGService()
    with init_wandb_run(
        job_type="rag-answer",
        name=args.name,
        project_root=PROJECT_ROOT,
        config={"provider": args.provider, "model": rag.model_name_for(args.provider), "top_k": rag.config["rag"]["top_k"]},
    ) as run:
        @weave.op(name="rag.answer")
        def traced_answer(question: str, document_id: str | None):
            """Create the parent RAG trace; OpenAI is captured as a child call."""
            return rag.answer(
                question,
                document_ids={document_id} if document_id else None,
                provider=args.provider,
            )

        result = traced_answer(args.query, args.document_id)
        run.log({
            "answer/is_answerable": result.is_answerable,
            "latency/search_ms": result.search_latency_ms,
            "latency/generation_ms": result.generation_latency_ms,
            "tokens/input": result.input_tokens or 0,
            "tokens/output": result.output_tokens or 0,
            "cost/estimated_usd": result.estimated_cost_usd or 0,
            "retrieval/chunk_count": len(result.retrieved_chunk_ids),
        })
        run.log({"results": __import__("wandb").Table(
            columns=["question", "answer", "answerable", "model", "citations"],
            data=[[args.query, result.answer, result.is_answerable, result.model,
                   ", ".join(citation.chunk_id for citation in result.citations)]],
        )})
        print(f"답변: {result.answer}")
        print(f"W&B: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
