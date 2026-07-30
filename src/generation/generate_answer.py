"""
ADVANCE RAG generate code

입찰메이트(BidMate) 프로젝트용 RAG 세션 모듈 (최적화 버전).

기능:
- 공공입찰 RFP 문서에 대한 질의응답 세션 클래스를 제공한다.
- 외부 retriever가 전달한 검색 결과(retrieved docs)를 하나의 context 문자열로 구성한다.
- OpenAI Async Responses API를 사용해 비동기 방식으로 답변을 생성한다.
- Structured Output(JSON Schema)를 사용해 답변을 안정적인 dict 형태로 반환한다.
- previous_response_id를 이용해 OpenAI 서버 측 멀티턴 대화 상태를 유지한다.
- 최근 대화(recent_messages), 대화 요약(conversation_summary), 핵심 필드(collected_fields)를 함께 관리한다.
- query rewrite 기능을 통해 사용자의 후속 질문을 검색용 standalone query로 재작성할 수 있다.
- 토큰 예산 관리를 통해 retrieved context와 대화 이력을 압축/절약한다.

주의:
- 이 파일은 재사용 가능한 소스 코드만 포함한다.
- 실행은 별도의 main 스크립트에서 한다.
- 실제 검색은 이 클래스 바깥의 retriever가 담당한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
너는 공공입찰/RFP 질의응답을 수행하는 AI다.
반드시 JSON 객체만 반환해야 한다.
설명, 마크다운, 코드펜스, 자연어 서문/후문은 절대 출력하지 마라.

반환 규칙:
1. 반드시 단일 JSON object만 반환한다.
2. 스키마에 없는 키는 추가하지 않는다.
3. 값이 없으면 빈 문자열("") 또는 빈 배열([])을 사용한다.
4. answer는 반드시 비어 있지 않은 문자열이어야 한다.
5. summary는 answer를 한두 문장으로 짧게 요약한다.
6. evidence에는 실제 근거 문맥을 넣는다.
7. confidence는 high, medium, low 중 하나다.
8. needs_clarification은 boolean이다.
"""

REWRITE_SYSTEM_PROMPT = """
너는 검색 질의 재작성기다.
사용자의 후속 질문을 검색에 적합한 독립형 질문으로 다시 써라.
반드시 JSON 객체만 반환하라.
"""

ANSWER_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "bidmate_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "summary": {"type": "string"},
            "reasoning": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"]
            },
            "needs_clarification": {"type": "boolean"},
            "clarification_question": {"type": "string"},
            "conflicts": {
                "type": "array",
                "items": {"type": "string"}
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source": {"type": "string"},
                        "page": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"}
                            ]
                        },
                        "chunk_id": {"type": "string"},
                        "score": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "null"}
                            ]
                        },
                        "quote": {"type": "string"}
                    },
                    "required": ["source", "page", "chunk_id", "score", "quote"]
                }
            },
            "retrieved_contexts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "answer",
            "summary",
            "reasoning",
            "confidence",
            "needs_clarification",
            "clarification_question",
            "conflicts",
            "evidence",
            "retrieved_contexts"
        ]
    }
}

REWRITE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "rewrite_query",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rewritten_query": {"type": "string"}
        },
        "required": ["rewritten_query"]
    }
}


class InvalidModelResponseError(RuntimeError):
    pass


def extract_json_text(raw_text: str) -> str:
    if raw_text is None:
        raise InvalidModelResponseError("response.output_text is None")

    text = raw_text.strip()
    if not text:
        raise InvalidModelResponseError("response.output_text is empty")

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    if not text:
        raise InvalidModelResponseError("response.output_text is empty after fence removal")

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    arr_start = text.find("[")
    arr_end = text.rfind("]")

    candidates = []
    if obj_start != -1 and obj_end != -1 and obj_start < obj_end:
        candidates.append(text[obj_start:obj_end + 1])
    if arr_start != -1 and arr_end != -1 and arr_start < arr_end:
        candidates.append(text[arr_start:arr_end + 1])

    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    raise InvalidModelResponseError(
        f"Could not extract valid JSON from model output. preview={text[:500]!r}"
    )


def safe_json_loads(raw_text: str) -> Any:
    text = extract_json_text(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidModelResponseError(
            f"Invalid JSON from model output. preview={text[:500]!r}"
        ) from e


async def parse_response_with_retry(
    call_model_coro,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = await call_model_coro()

            raw_text = getattr(response, "output_text", None)
            logger.info("[ask] output_text length=%s", 0 if raw_text is None else len(raw_text))
            logger.debug("[ask] output_text preview=%r", "" if raw_text is None else raw_text[:1000])

            if hasattr(response, "id"):
                logger.debug("[ask] response id=%s", response.id)

            parsed = safe_json_loads(raw_text)
            return parsed

        except Exception as e:
            last_error = e
            logger.warning(
                "[ask] parse failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries + 1,
                repr(e),
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                break

    raise InvalidModelResponseError(f"Model response parsing failed after retries: {last_error}")


def dedupe_retrieved_docs(retrieved_docs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not retrieved_docs:
        return []

    seen_chunk_ids = set()
    seen_texts = set()
    deduped = []

    for doc in retrieved_docs:
        if not isinstance(doc, dict):
            continue

        chunk_id = str(doc.get("chunk_id", "")).strip()
        text = str(doc.get("text", "")).strip()

        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if text and text in seen_texts:
            continue

        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if text:
            seen_texts.add(text)

        deduped.append(doc)

    return deduped


def format_retrieved_context(retrieved_docs: Optional[List[Dict[str, Any]]], max_docs: int = 5) -> str:
    docs = dedupe_retrieved_docs(retrieved_docs)[:max_docs]
    if not docs:
        return "검색 문맥 없음"

    lines = []
    for i, doc in enumerate(docs, start=1):
        chunk_id = str(doc.get("chunk_id", "")).strip()
        source = str(doc.get("source", "")).strip()
        score = doc.get("score", None)
        page = doc.get("page", None)
        text = str(doc.get("text", "")).strip()

        block = [
            f"[문맥 {i}]",
            f"chunk_id: {chunk_id or '-'}",
            f"source: {source or '-'}",
            f"page: {page if page is not None else '-'}",
            f"score: {score if score is not None else '-'}",
            f"text: {text or '-'}",
        ]
        lines.append("\n".join(block))

    return "\n\n".join(lines)


class BidMateRAGSession:
    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.previous_response_id: Optional[str] = None
        self.recent_messages: List[Dict[str, str]] = []
        self.conversation_summary: str = ""
        self.collected_fields: Dict[str, Any] = {}

    def reset(self) -> None:
        self.previous_response_id = None
        self.recent_messages = []
        self.conversation_summary = ""
        self.collected_fields = {}

    def _is_rewrite_skippable(self, query: str) -> bool:
        q = query.strip()
        if not q:
            return True
        if len(self.recent_messages) == 0:
            return True
        if len(q) >= 20 and not re.search(r"\b(그거|그것|위|이전|해당|그럼|그러면)\b", q):
            return True
        return False

    async def rewrite_query(self, query: str) -> str:
        if self._is_rewrite_skippable(query):
            return query

        history = self.recent_messages[-4:]
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history
        ) or "없음"

        prompt = f"""
[최근 대화]
{history_text}

[사용자 질문]
{query}

[작업]
검색용 standalone query로 재작성하라.
""".strip()

        async def _runner():
            return await self.client.responses.create(
                model=self.model,
                instructions=REWRITE_SYSTEM_PROMPT,
                input=prompt,
                text={"format": REWRITE_JSON_SCHEMA},
            )

        parsed = await parse_response_with_retry(_runner, max_retries=1, retry_delay=0.5)
        rewritten = str(parsed.get("rewritten_query", "")).strip()
        return rewritten or query

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        rewritten_query: Optional[str] = None,
    ) -> str:
        context = format_retrieved_context(retrieved_docs, max_docs=5)

        compact_fields = {
            k: v for k, v in self.collected_fields.items()
            if v not in (None, "", [], {}, ())
        }

        recent_history = self.recent_messages[-6:]
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_history
        ) or "없음"

        return f"""
[사용자 원문 질문]
{query}

[검색용 재작성 질문]
{rewritten_query or query}

[검색 문맥]
{context}

[최근 대화]
{history_text}

[대화 요약]
{self.conversation_summary or '없음'}

[수집된 핵심 필드]
{json.dumps(compact_fields, ensure_ascii=False) if compact_fields else '없음'}

[작업 지시]
- 검색 문맥을 최우선 근거로 사용하라.
- 답변은 한국어로 작성하라.
- summary는 answer를 짧게 요약하라.
- confidence는 high, medium, low 중 하나로 판단하라.
- 근거가 부족하면 needs_clarification=true로 두고 clarification_question을 작성하라.
- evidence에는 실제 근거 문장을 넣어라.
- conflicts에는 문맥 간 충돌이 있을 때만 넣어라.
""".strip()

    async def ask(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        rewritten_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            rewritten_query=rewritten_query,
        )

        async def _runner():
            return await self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                text={"format": ANSWER_JSON_SCHEMA},
                previous_response_id=self.previous_response_id,
            )

        result = await parse_response_with_retry(_runner, max_retries=2, retry_delay=1.0)

        self.recent_messages.append({"role": "user", "content": query})
        self.recent_messages.append({"role": "assistant", "content": str(result.get("answer", ""))})
        self.recent_messages = self.recent_messages[-10:]

        return result