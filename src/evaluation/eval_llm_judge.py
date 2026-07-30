"""
LLM Judge evaluation runner for the BidMate RAG pipeline.

이 모듈은 BidMate의 실제 RAG 실행 흐름을 그대로 따라가며
질문별 답변을 생성한 뒤, 별도의 LLM Judge가 해당 답변의 품질을
루브릭 기반으로 평가하도록 만든 평가 스크립트이다.

이 파일의 목적은 단순한 정량 지표 계산(RAGAS)만으로는 확인하기 어려운
실무적 품질 요소를 별도의 평가 모델이 직접 심사하도록 하는 데 있다.
예를 들어, 답변이 질문에 충분히 직접적으로 답했는지,
검색 문맥에 근거했는지, 정답과 비교해 핵심을 빠뜨리지 않았는지,
설명이 명확한지 등을 종합적으로 판단할 수 있다.

이 모듈은 다음과 같은 순서로 동작한다.

1. 평가 데이터(JSON 배열)를 읽는다.
2. 각 질문에 대해 실제 운영 경로와 동일하게 RAG를 수행한다.
   - rewrite_query(question)
   - search_documents(rewritten_query, k=top_k)
   - session.ask(query=question, retrieved_docs=..., rewritten_query=...)
3. search_documents()가 반환한 실제 본문(text)만 추출해
   judge 입력용 retrieved_contexts를 구성한다.
4. 질문, 기준 정답(reference), 실제 생성 답변(answer), 검색 문맥(contexts)을
   judge LLM에 전달한다.
5. judge LLM은 사전에 정의한 루브릭에 따라 점수를 부여한다.
6. 결과를 DataFrame으로 정리하고 CSV로 저장한다.

입력 파일 형식:
[
  {
    "question": "...",
    "reference": "..."
  },
  ...
]

출력:
- 질문별 RAG 실행 결과
- 질문별 LLM Judge 평가 점수
- 질문별 평가 사유(reasoning)
- 누락 포인트(missing_points)
- 근거 부족 주장(unsupported_claims)
- 최종 CSV 저장

Judge 평가 항목:
- correctness:
  기준 정답(reference) 대비 실제 답변이 사실적으로 맞는가
- groundedness:
  답변이 retrieved_contexts에 근거하고 있는가
- relevance:
  질문에 직접적으로 답하고 있는가
- completeness:
  필요한 핵심 요소를 빠뜨리지 않았는가
- clarity:
  답변이 구조적이고 이해하기 쉬운가

점수 체계:
- 각 항목 1점 ~ 5점
- overall_score는 1.0 ~ 5.0
- verdict는 아래 규칙을 따른다
  - pass: overall_score >= 4.0
  - borderline: 3.0 <= overall_score < 4.0
  - fail: overall_score < 3.0

주의:
- 이 스크립트는 운영 RAG 흐름과 최대한 동일하게 평가하기 위해
  retrieval 단계를 반드시 포함한다.
- 따라서 session.ask(question) 단독 호출이 아니라
  rewrite_query -> search_documents -> ask(...) 순서로 동작한다.
- judge 입력의 retrieved_contexts에는 반드시 실제 검색 문서 본문이 들어가야 한다.
  "문맥 1", "문맥 2" 같은 라벨 문자열만 넣으면 groundedness 및 completeness 평가가 왜곡된다.
- judge 프롬프트가 과도하게 길어지면 평가 비용과 지연이 증가할 수 있으므로,
  질문/답변/문맥은 사전에 길이 제한을 적용한다.
- 이 파일은 RAGAS를 대체하는 것이 아니라 보완한다.
  RAGAS는 정량 지표 추적용,
  본 LLM Judge는 실무적 품질 심사용으로 함께 사용하는 것을 권장한다.

환경 변수:
- OPENAI_API_KEY:
  OpenAI API 키
- OPENAI_MODEL:
  실제 RAG 답변 생성에 사용할 모델명
- LLM_JUDGE_MODEL:
  judge 전용 모델명, 없으면 OPENAI_MODEL을 사용
- RAG_TOP_K:
  retrieval 시 가져올 문서 수
- JUDGE_CONTEXT_TOP_K:
  judge에 넘길 문맥 개수
- JUDGE_CONTEXT_MAX_CHARS:
  각 context 최대 길이
- JUDGE_ANSWER_MAX_CHARS:
  답변 최대 길이
- JUDGE_REFERENCE_MAX_CHARS:
  reference 최대 길이
- JUDGE_QUESTION_MAX_CHARS:
  question 최대 길이

주요 함수:
- load_eval_rows():
  평가용 JSON 파일을 읽고 question/reference를 정규화한다.
- extract_retrieved_contexts_from_docs():
  검색 결과에서 실제 본문 text만 추출해 judge 입력용 contexts를 만든다.
- run_single_query():
  질문 1건에 대해 실제 RAG 실행을 수행하고 answer/context를 반환한다.
- build_judge_prompt():
  judge 모델에 전달할 평가 프롬프트를 생성한다.
- parse_judge_json():
  judge 응답 텍스트에서 JSON을 안전하게 파싱한다.
- judge_single_response():
  judge LLM을 호출해 항목별 점수와 판정을 반환한다.
- main():
  전체 평가 파이프라인을 실행하고 CSV를 저장한다.

예외 처리 정책:
- RAG 실행 실패 시 status와 error를 결과에 남기고 다음 질문으로 진행한다.
- judge JSON 파싱 실패 시 judge_error로 기록한다.
- 전체 실행 중 일부 질문 실패가 있더라도 가능한 범위 내에서 CSV를 남긴다.

실행 예시:
    uv run -m src.evaluation.eval_llm_judge

권장 사용 방식:
- 정량 추적:
  src.evaluation.eval_ragas.py
- 정성/루브릭 기반 평가:
  src.evaluation.eval_llm_judge.py

이 모듈은 "정답 여부"만 보는 것이 아니라,
"왜 좋은 답인지 / 왜 부족한 답인지"를 분석 가능한 형태로 남기는 것을 목표로 한다.
"""

# src/evaluation/eval_llm_judge.py

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
from openai import AsyncOpenAI

from src.generation.generate_answer import BidMateRAGSession, InvalidModelResponseError
from src.retrieval.retriever import search_documents


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
EVAL_DATA_PATH = PROJECT_ROOT / "src/evaluation/eval_samples.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_PATH = PROJECT_ROOT / f"output/llm_judge_eval_results_{timestamp}.csv"

JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))

TOP_K = int(os.getenv("RAG_TOP_K", "5"))

JUDGE_CONTEXT_TOP_K = int(os.getenv("JUDGE_CONTEXT_TOP_K", "3"))
JUDGE_CONTEXT_MAX_CHARS = int(os.getenv("JUDGE_CONTEXT_MAX_CHARS", "2500"))
JUDGE_ANSWER_MAX_CHARS = int(os.getenv("JUDGE_ANSWER_MAX_CHARS", "4000"))
JUDGE_REFERENCE_MAX_CHARS = int(os.getenv("JUDGE_REFERENCE_MAX_CHARS", "3000"))
JUDGE_QUESTION_MAX_CHARS = int(os.getenv("JUDGE_QUESTION_MAX_CHARS", "1500"))


def truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return text[:max_chars] if len(text) > max_chars else text


def load_eval_rows(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"평가 데이터 파일을 찾을 수 없습니다: {file_path} (cwd={Path.cwd()})"
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
    top_k: int = JUDGE_CONTEXT_TOP_K,
    max_chars: int = JUDGE_CONTEXT_MAX_CHARS,
) -> List[str]:
    contexts: List[str] = []

    for doc in (retrieved_docs or [])[:top_k]:
        if not isinstance(doc, dict):
            continue

        text = str(doc.get("text", "")).strip()
        if not text:
            continue

        contexts.append(truncate_text(text, max_chars))

    return contexts


async def run_single_query(
    session: BidMateRAGSession,
    question: str,
    ground_truth: Optional[str] = None,
    top_k: int = TOP_K,
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
                "rewritten_query": rewritten_query,
                "answer": "",
                "contexts": [],
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

        contexts = extract_retrieved_contexts_from_docs(retrieved_docs)

        total_elapsed = time.perf_counter() - t_query

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": truncate_text(result.get("answer", ""), JUDGE_ANSWER_MAX_CHARS),
            "contexts": contexts,
            "ground_truth": truncate_text(ground_truth or "", JUDGE_REFERENCE_MAX_CHARS),
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
            "ground_truth": ground_truth or "",
            "status": "runtime_error",
            "error": repr(e),
            "elapsed_sec": total_elapsed,
            "rewrite_sec": 0.0,
            "search_sec": 0.0,
            "ask_sec": 0.0,
        }


def build_judge_prompt(
    question: str,
    reference: str,
    answer: str,
    contexts: List[str],
) -> str:
    joined_contexts = "\n\n".join(
        f"[Context {idx}]\n{ctx}" for idx, ctx in enumerate(contexts, start=1)
    )

    return f"""
You are a strict evaluator for a Korean public-procurement RAG system.

Evaluate the model answer using the question, reference answer, and retrieved contexts.

Scoring rubric:
- correctness: Does the answer match the reference answer factually and materially? (1-5)
- groundedness: Is the answer supported by the retrieved contexts only? Penalize unsupported claims. (1-5)
- relevance: Does the answer directly answer the user's question? (1-5)
- completeness: Does the answer cover the important points needed for the question? (1-5)
- clarity: Is the answer clear, well-structured, and easy to understand? (1-5)

Scoring guidance:
- 5 = excellent
- 4 = good, minor issues
- 3 = acceptable but noticeable gaps
- 2 = poor
- 1 = very poor / mostly wrong

Return ONLY valid JSON with this schema:
{{
  "correctness": 1,
  "groundedness": 1,
  "relevance": 1,
  "completeness": 1,
  "clarity": 1,
  "overall_score": 1.0,
  "verdict": "pass|borderline|fail",
  "reasoning": "brief explanation",
  "missing_points": ["..."],
  "unsupported_claims": ["..."]
}}

Rules:
- overall_score must be a float from 1.0 to 5.0
- verdict:
  - pass if overall_score >= 4.0
  - borderline if overall_score >= 3.0 and < 4.0
  - fail if overall_score < 3.0
- If reference is empty, judge correctness/completeness more conservatively using question and contexts.
- If contexts do not support an answer claim, reduce groundedness.
- Be strict and concise.

[Question]
{question}

[Reference Answer]
{reference}

[Retrieved Contexts]
{joined_contexts}

[Model Answer]
{answer}
""".strip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Judge JSON 파싱 실패: {text[:500]}")


async def judge_single_response(
    client: AsyncOpenAI,
    model: str,
    question: str,
    reference: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    prompt = build_judge_prompt(
        question=truncate_text(question, JUDGE_QUESTION_MAX_CHARS),
        reference=truncate_text(reference, JUDGE_REFERENCE_MAX_CHARS),
        answer=truncate_text(answer, JUDGE_ANSWER_MAX_CHARS),
        contexts=contexts[:JUDGE_CONTEXT_TOP_K],
    )

    response = await client.responses.create(
        model=model,
        input=prompt,
    )

    output_text = getattr(response, "output_text", "") or ""
    parsed = parse_judge_json(output_text)

    return {
        "correctness": parsed.get("correctness"),
        "groundedness": parsed.get("groundedness"),
        "relevance": parsed.get("relevance"),
        "completeness": parsed.get("completeness"),
        "clarity": parsed.get("clarity"),
        "overall_score": parsed.get("overall_score"),
        "verdict": parsed.get("verdict"),
        "reasoning": parsed.get("reasoning", ""),
        "missing_points": json.dumps(parsed.get("missing_points", []), ensure_ascii=False),
        "unsupported_claims": json.dumps(parsed.get("unsupported_claims", []), ensure_ascii=False),
        "judge_raw_output": output_text,
    }


async def main():
    t_all_start = time.perf_counter()

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        top_k = int(os.getenv("RAG_TOP_K", str(TOP_K)))

        session = BidMateRAGSession(api_key=openai_api_key, model=model_name)
        judge_client = AsyncOpenAI(api_key=openai_api_key)

        eval_rows = load_eval_rows(EVAL_DATA_PATH)
        logger.info("평가 데이터 로드 완료: %d건", len(eval_rows))
        logger.info("LLM Judge model=%s", JUDGE_MODEL)

        results = []
        total = len(eval_rows)

        for idx, row in enumerate(eval_rows, start=1):
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            logger.info("[%s/%s] 질의 실행 시작", idx, total)
            run_result = await run_single_query(
                session=session,
                question=question,
                ground_truth=ground_truth,
                top_k=top_k,
            )

            logger.info(
                "[%s/%s] 질의 실행 완료 status=%s total=%.2fs rewrite=%.2fs search=%.2fs ask=%.2fs",
                idx,
                total,
                run_result["status"],
                run_result.get("elapsed_sec", 0.0),
                run_result.get("rewrite_sec", 0.0),
                run_result.get("search_sec", 0.0),
                run_result.get("ask_sec", 0.0),
            )

            base_row = {
                "question": question,
                "rewritten_query": run_result.get("rewritten_query", ""),
                "reference": ground_truth,
                "answer": run_result.get("answer", ""),
                "retrieved_contexts": json.dumps(run_result.get("contexts", []), ensure_ascii=False),
                "status": run_result.get("status", ""),
                "error": run_result.get("error", ""),
                "query_elapsed_sec": run_result.get("elapsed_sec", 0.0),
                "rewrite_sec": run_result.get("rewrite_sec", 0.0),
                "search_sec": run_result.get("search_sec", 0.0),
                "ask_sec": run_result.get("ask_sec", 0.0),
            }

            if run_result["status"] != "success":
                results.append(
                    {
                        **base_row,
                        "correctness": None,
                        "groundedness": None,
                        "relevance": None,
                        "completeness": None,
                        "clarity": None,
                        "overall_score": None,
                        "verdict": "error",
                        "reasoning": "",
                        "missing_points": "[]",
                        "unsupported_claims": "[]",
                        "judge_raw_output": "",
                        "judge_elapsed_sec": 0.0,
                    }
                )
                continue

            logger.info("[%s/%s] LLM Judge 시작", idx, total)
            t_judge = time.perf_counter()

            try:
                judge_result = await judge_single_response(
                    client=judge_client,
                    model=JUDGE_MODEL,
                    question=question,
                    reference=ground_truth,
                    answer=run_result.get("answer", ""),
                    contexts=run_result.get("contexts", []),
                )
                judge_elapsed = time.perf_counter() - t_judge

                logger.info("[%s/%s] LLM Judge 완료 (%.2fs)", idx, total, judge_elapsed)

                results.append(
                    {
                        **base_row,
                        **judge_result,
                        "judge_elapsed_sec": judge_elapsed,
                    }
                )

            except Exception as e:
                judge_elapsed = time.perf_counter() - t_judge
                logger.exception("[%s/%s] LLM Judge 실패", idx, total)

                results.append(
                    {
                        **base_row,
                        "correctness": None,
                        "groundedness": None,
                        "relevance": None,
                        "completeness": None,
                        "clarity": None,
                        "overall_score": None,
                        "verdict": "judge_error",
                        "reasoning": repr(e),
                        "missing_points": "[]",
                        "unsupported_claims": "[]",
                        "judge_raw_output": "",
                        "judge_elapsed_sec": judge_elapsed,
                    }
                )

        if not results:
            raise RuntimeError("No evaluation results were produced.")

        df = pd.DataFrame(results)

        print("=== LLM JUDGE RESULT PREVIEW ===", flush=True)
        print(df.head(), flush=True)
        print()
        print("row count:", len(df), flush=True)
        print("columns:", list(df.columns), flush=True)
        print()

        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"평가 결과 CSV 저장 완료: {OUTPUT_CSV_PATH}", flush=True)

        if "overall_score" in df.columns:
            valid_scores = pd.to_numeric(df["overall_score"], errors="coerce")
            print("mean overall_score:", valid_scores.mean(), flush=True)
            print("pass count:", int((df["verdict"] == "pass").sum()), flush=True)
            print("borderline count:", int((df["verdict"] == "borderline").sum()), flush=True)
            print("fail count:", int((df["verdict"] == "fail").sum()), flush=True)

    finally:
        total_elapsed = time.perf_counter() - t_all_start
        logger.info("전체 실행 시간: %.2fs", total_elapsed)


if __name__ == "__main__":
    asyncio.run(main())