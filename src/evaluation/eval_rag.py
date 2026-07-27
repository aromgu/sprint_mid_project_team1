# ruff: noqa: E402

import os
import sys
import types

vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:
    pass


vertexai_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = vertexai_stub

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.generation.generate_answer import BidMateRAGSession
from src.retrieval.retriever import search_documents

load_dotenv()


def load_eval_samples(json_path: str) -> List[Dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def docs_to_context_list(docs: List[Dict[str, Any]]) -> List[str]:
    contexts = []
    for doc in docs:
        text = (doc.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


async def run_single_query(
    session: BidMateRAGSession,
    question: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    rewritten_query = await session.rewrite_query(question)
    retrieved_docs = search_documents(rewritten_query, k=top_k)

    result = await session.ask(
        query=question,
        retrieved_docs=retrieved_docs,
        rewritten_query=rewritten_query,
    )

    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "retrieved_docs": retrieved_docs,
        "answer": result.get("answer", ""),
        "raw_result": result,
    }


async def build_ragas_dataset(
    eval_samples: List[Dict[str, str]],
    top_k: int = 5,
) -> EvaluationDataset:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    session = BidMateRAGSession(api_key=api_key)
    ragas_samples: List[SingleTurnSample] = []

    for idx, item in enumerate(eval_samples, start=1):
        question = item["question"]
        reference = item["reference"]

        print(f"[{idx}/{len(eval_samples)}] 평가 중: {question}")

        run_result = await run_single_query(
            session=session,
            question=question,
            top_k=top_k,
        )

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=docs_to_context_list(run_result["retrieved_docs"]),
            response=run_result["answer"],
            reference=reference,
        )
        ragas_samples.append(sample)

    return EvaluationDataset(samples=ragas_samples)


def save_dataset_preview(
    eval_dataset: EvaluationDataset,
    output_path: str,
) -> None:
    samples = []
    for sample in eval_dataset.samples:
        samples.append(
            {
                "user_input": sample.user_input,
                "retrieved_contexts": sample.retrieved_contexts,
                "response": sample.response,
                "reference": sample.reference,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


async def main():
    eval_json_path = CURRENT_DIR / "eval_samples.json"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_preview_path = output_dir / f"{timestamp}_ragas_eval_dataset_preview.json"
    result_path = output_dir / f"{timestamp}_ragas_result.json"

    eval_samples = load_eval_samples(str(eval_json_path))

    eval_dataset = await build_ragas_dataset(
        eval_samples=eval_samples,
        top_k=5,
    )

    save_dataset_preview(
        eval_dataset=eval_dataset,
        output_path=str(dataset_preview_path),
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    print("\n===== RAGAS 평가 결과 =====")
    print(result)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result._repr_dict, f, ensure_ascii=False, indent=2)

    print(f"\n평가 데이터 저장 완료: {dataset_preview_path}")
    print(f"평가 결과 저장 완료: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
