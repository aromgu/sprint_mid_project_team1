# src/evaluation/eval_ragas.py

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

from src.generation.generate_answer import BidMateRAGSession, InvalidModelResponseError
from src.retrieval.retriever import search_documents


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
EVAL_DATA_PATH = PROJECT_ROOT / "src/evaluation/eval_samples.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_PATH = PROJECT_ROOT / f"output/ragas_eval_results_{timestamp}.csv"

EMBED_MAX_CHARS = 8000
RAGAS_CONTEXT_MAX_CHARS = 1200
RAGAS_RESPONSE_MAX_CHARS = 800
RAGAS_REFERENCE_MAX_CHARS = 3000
RAGAS_CONTEXT_TOP_K = 2


def truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].strip()


def load_eval_rows(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"평가 데이터 파일을 찾을 수 없습니다: {file_path} "
            f"(cwd={Path.cwd()})"
        )

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("평가 데이터는 JSON 배열(list) 형식이어야 합니다.")

    rows = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            logger.warning("[%s] dict가 아닌 항목은 건너뜁니다: %r", idx, item)
            continue

        question = str(item.get("question", "")).strip()
        reference = str(item.get("reference", "")).strip()

        if not question:
            logger.warning("[%s] question이 비어 있어 건너뜁니다.", idx)
            continue

        rows.append(
            {
                "question": question,
                "ground_truth": reference,
            }
        )

    return rows


def extract_retrieved_contexts_from_docs(
    retrieved_docs: List[Dict[str, Any]],
    top_k: int = RAGAS_CONTEXT_TOP_K,
    max_chars: int = RAGAS_CONTEXT_MAX_CHARS,
) -> List[str]:
    """
    RAGAS retrieved_contexts에는 반드시 실제 본문(text)을 넣는다.
    faithfulness 안정화를 위해 너무 긴 문맥은 잘라서 사용한다.
    """
    contexts: List[str] = []

    for doc in (retrieved_docs or [])[:top_k]:
        if not isinstance(doc, dict):
            continue

        text = str(doc.get("text", "")).strip()
        if not text:
            continue

        text = truncate_text(text, max_chars=max_chars)
        if text:
            contexts.append(text)

    return contexts


def get_metric_score(result: Any, metric_name: str):
    try:
        return result[metric_name]
    except Exception:
        pass

    try:
        return getattr(result, metric_name)
    except Exception:
        pass

    try:
        if isinstance(result, dict):
            return result.get(metric_name)
    except Exception:
        pass

    return None


async def run_single_query(
    session: BidMateRAGSession,
    question: str,
    ground_truth: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    t_query = time.perf_counter()

    try:
        t_rewrite = time.perf_counter()
        rewritten_query = await session.rewrite_query(question)
        rewrite_elapsed = time.perf_counter() - t_rewrite

        t_search = time.perf_counter()
        retrieved_docs = search_documents(rewritten_query, k=top_k)
        search_elapsed = time.perf_counter() - t_search

        if not retrieved_docs:
            total_elapsed = time.perf_counter() - t_query
            return {
                "question": question,
                "answer": "",
                "contexts": [],
                "citations": [],
                "ground_truth": ground_truth or "",
                "status": "no_docs",
                "error": "검색된 문서가 없습니다.",
                "elapsed_sec": total_elapsed,
                "rewrite_sec": rewrite_elapsed,
                "search_sec": search_elapsed,
                "ask_sec": 0.0,
            }

        t_ask = time.perf_counter()
        result = await session.ask(
            query=question,
            retrieved_docs=retrieved_docs,
            rewritten_query=rewritten_query,
        )
        ask_elapsed = time.perf_counter() - t_ask

        contexts = extract_retrieved_contexts_from_docs(
            retrieved_docs=retrieved_docs,
            top_k=RAGAS_CONTEXT_TOP_K,
            max_chars=RAGAS_CONTEXT_MAX_CHARS,
        )

        citations = result.get("citations", [])
        if not isinstance(citations, list):
            citations = [str(citations)] if citations else []

        total_elapsed = time.perf_counter() - t_query

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "citations": citations,
            "ground_truth": ground_truth or "",
            "status": "success",
            "error": "",
            "elapsed_sec": total_elapsed,
            "rewrite_sec": rewrite_elapsed,
            "search_sec": search_elapsed,
            "ask_sec": ask_elapsed,
        }

    except InvalidModelResponseError as e:
        logger.error("[run_single_query] invalid model response: %s", e)
        total_elapsed = time.perf_counter() - t_query
        return {
            "question": question,
            "rewritten_query": "",
            "answer": "",
            "contexts": [],
            "citations": [],
            "ground_truth": ground_truth or "",
            "status": "parse_error",
            "error": str(e),
            "elapsed_sec": total_elapsed,
            "rewrite_sec": 0.0,
            "search_sec": 0.0,
            "ask_sec": 0.0,
        }

    except Exception as e:
        logger.exception("[run_single_query] unexpected error")
        total_elapsed = time.perf_counter() - t_query
        return {
            "question": question,
            "rewritten_query": "",
            "answer": "",
            "contexts": [],
            "citations": [],
            "ground_truth": ground_truth or "",
            "status": "runtime_error",
            "error": repr(e),
            "elapsed_sec": total_elapsed,
            "rewrite_sec": 0.0,
            "search_sec": 0.0,
            "ask_sec": 0.0,
        }


async def build_ragas_dataset(
    session: BidMateRAGSession,
    eval_rows: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    dataset = []

    total = len(eval_rows)
    for idx, row in enumerate(eval_rows, start=1):
        question = row["question"]
        ground_truth = row.get("ground_truth", "")

        logger.info("[%s/%s] 평가 중: %s", idx, total, question)

        run_result = await run_single_query(
            session=session,
            question=question,
            ground_truth=ground_truth,
            top_k=top_k,
        )

        logger.info(
            "[%s/%s] 완료 status=%s total=%.2fs rewrite=%.2fs search=%.2fs ask=%.2fs",
            idx,
            total,
            run_result["status"],
            run_result.get("elapsed_sec", 0.0),
            run_result.get("rewrite_sec", 0.0),
            run_result.get("search_sec", 0.0),
            run_result.get("ask_sec", 0.0),
        )

        if run_result["status"] != "success":
            logger.warning(
                "[%s/%s] skip: status=%s error=%s",
                idx,
                total,
                run_result["status"],
                run_result["error"],
            )
            continue

        contexts = run_result.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]

        contexts = [truncate_text(x, EMBED_MAX_CHARS) for x in contexts if str(x).strip()]

        dataset.append(
            {
                "user_input": truncate_text(question, EMBED_MAX_CHARS),
                "response": truncate_text(
                    run_result.get("answer", ""),
                    RAGAS_RESPONSE_MAX_CHARS,
                ),
                "retrieved_contexts": contexts,
                "reference": truncate_text(
                    ground_truth or "",
                    RAGAS_REFERENCE_MAX_CHARS,
                ),
            }
        )

    return dataset


async def main():
    t_all_start = time.perf_counter()

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        eval_model_name = os.getenv("RAGAS_EVAL_MODEL", model_name)
        eval_embedding_model = os.getenv("RAGAS_EVAL_EMBEDDING_MODEL", "text-embedding-3-small")
        top_k = int(os.getenv("RAG_TOP_K", "5"))

        session = BidMateRAGSession(api_key=openai_api_key, model=model_name)
        openai_client = AsyncOpenAI(api_key=openai_api_key)

        evaluator_llm = llm_factory(
            eval_model_name,
            client=openai_client,
            max_tokens=4096,
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                api_key=openai_api_key,
                model=eval_embedding_model,
            )
        )

        eval_rows = load_eval_rows(EVAL_DATA_PATH)
        logger.info("평가 데이터 로드 완료: %d건", len(eval_rows))

        logger.info("RAGAS 입력 dataset 생성 시작")
        t_dataset_start = time.perf_counter()
        eval_dataset = await build_ragas_dataset(
            session=session,
            eval_rows=eval_rows,
            top_k=top_k,
        )
        dataset_elapsed = time.perf_counter() - t_dataset_start
        logger.info("RAGAS 입력 dataset 생성 완료: %d건 (%.2fs)", len(eval_dataset), dataset_elapsed)

        if not eval_dataset:
            raise RuntimeError("No valid evaluation rows were produced.")

        df = pd.DataFrame(eval_dataset)

        print("=== RAGAS INPUT DATASET ===", flush=True)
        print(df.head(), flush=True)
        print()
        print("row count:", len(df), flush=True)
        print("columns:", list(df.columns), flush=True)
        print()

        ragas_dataset = Dataset.from_pandas(df, preserve_index=False)

        logger.info("RAGAS evaluate() 시작")
        logger.info("RAG generation model=%s", model_name)
        logger.info("RAGAS eval model=%s", eval_model_name)
        logger.info("RAGAS eval embedding model=%s", eval_embedding_model)
        logger.info(
            "RAGAS eval config: top_k=%s, context_top_k=%s, context_max_chars=%s, response_max_chars=%s, reference_max_chars=%s",
            top_k,
            RAGAS_CONTEXT_TOP_K,
            RAGAS_CONTEXT_MAX_CHARS,
            RAGAS_RESPONSE_MAX_CHARS,
            RAGAS_REFERENCE_MAX_CHARS,
        )

        t_eval_start = time.perf_counter()
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
        )
        eval_elapsed = time.perf_counter() - t_eval_start
        logger.info("RAGAS evaluate() 완료 (%.2fs)", eval_elapsed)

        print("=== RESULT TYPE ===", flush=True)
        print(type(result), flush=True)
        print()

        print("=== RAGAS EVALUATION RESULT ===", flush=True)
        print(result, flush=True)
        print()

        print("answer_relevancy", flush=True)
        print(get_metric_score(result, "answer_relevancy"), flush=True)

        print("faithfulness", flush=True)
        print(get_metric_score(result, "faithfulness"), flush=True)

        print("context_precision", flush=True)
        print(get_metric_score(result, "context_precision"), flush=True)

        print("context_recall", flush=True)
        print(get_metric_score(result, "context_recall"), flush=True)
        print()

        result_df = result.to_pandas()

        print("=== RAGAS RESULT DATAFRAME ===", flush=True)
        print(result_df.head(), flush=True)
        print("result_df columns:", list(result_df.columns), flush=True)
        print()

        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"평가 결과 CSV 저장 완료: {OUTPUT_CSV_PATH}", flush=True)

    finally:
        total_elapsed = time.perf_counter() - t_all_start
        logger.info("전체 실행 시간: %.2fs", total_elapsed)


if __name__ == "__main__":
    asyncio.run(main())