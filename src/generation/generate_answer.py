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

토큰 절약 최적화 전략:
1. Prompt Caching 최적화: 고정 시스템 지침 및 배경을 프롬프트 최상단(Prefix)에 배치.
2. 데이터 중복 제거(Deduping):
   - 검색된 docs 중 동일 chunk_id / 동일 text 중복 제거.
   - collected_fields 중 값이 채워진 필드만 프롬프트에 포함.
3. Query Rewrite 최적화:
   - 첫 턴이거나 질문이 이미 독립적/명확한 경우 Rewrite LLM 호출 Skip (휴리스틱).
   - Rewrite 프롬프트 크기 대폭 경량화 (최근 2턴 대화만 참고).
4. JSON Schema 구조 통합:
   - citations와 evidence_quotes를 단일 evidence 배열로 통합하여 출력 토큰 절감.

주의:
- 이 파일은 재사용 가능한 소스 코드만 포함한다.
- 샘플 입력, print 테스트, 실행용 코드는 main 스크립트에 분리해서 작성한다.
- 실제 검색은 이 클래스 바깥의 retriever가 담당한다.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from openai import AsyncOpenAI


@dataclass
class TokenBudget:
    """
    RAG 프롬프트 토큰 예산 관리용 설정값.
    """

    max_context_chars: int = 7000
    max_recent_turns: int = 4
    max_docs: int = 6
    min_score: float = 0.0


class BidMateRAGSession:
    """
    입찰메이트 사내 RAG용 세션 클래스 (토큰 및 비용 최적화 버전)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-nano",
        max_context_chars: int = 7000,
        max_recent_turns: int = 4,
        max_docs: int = 6,
        min_score: float = 0.0,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.budget = TokenBudget(
            max_context_chars=max_context_chars,
            max_recent_turns=max_recent_turns,
            max_docs=max_docs,
            min_score=min_score,
        )

        self.previous_response_id: Optional[str] = None
        self.recent_messages: List[Dict[str, str]] = []
        self.conversation_summary: str = ""
        self.last_rewritten_query: Optional[str] = None

        self.collected_fields: Dict[str, Optional[str]] = {
            "project_name": None,
            "organization": None,
            "budget": None,
            "duration": None,
            "deadline": None,
            "submission_method": None,
            "required_technology": None,
            "eligibility": None,
            "evaluation_criteria": None,
            "required_documents": None,
            "notes": None,
        }

    def reset(self) -> None:
        self.previous_response_id = None
        self.recent_messages = []
        self.conversation_summary = ""
        self.last_rewritten_query = None
        self.collected_fields = {key: None for key in self.collected_fields.keys()}

    def _prune_and_dedupe_docs(
        self, retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        점수 미달 문서 제거 및 동일 Chunk ID/내용에 대한 디두플리케이션(중복 제거)
        """
        filtered_docs = []
        seen_chunks: Set[str] = set()

        for doc in retrieved_docs:
            score = doc.get("score")
            if score is not None and float(score) < self.budget.min_score:
                continue

            metadata = doc.get("metadata", {}) or {}
            chunk_id = doc.get("chunk_id") or metadata.get("chunk_id")
            text = (doc.get("text") or "").strip()

            # Unique key 지정 (chunk_id 우선, 없을 경우 text hash 사용)
            dedupe_key = str(chunk_id) if chunk_id is not None else str(hash(text))

            if dedupe_key in seen_chunks:
                continue

            seen_chunks.add(dedupe_key)
            filtered_docs.append(doc)

        filtered_docs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return filtered_docs[: self.budget.max_docs]

    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> str:
        limit = max_chars or self.budget.max_context_chars
        docs = self._prune_and_dedupe_docs(retrieved_docs)

        blocks: List[str] = []
        total_len = 0

        for i, doc in enumerate(docs, start=1):
            text = (doc.get("text") or "").strip()
            source = doc.get("file_nm") or doc.get("source") or f"doc_{i}"
            score = doc.get("score")
            metadata = doc.get("metadata", {}) or {}

            page = (
                doc.get("page") if doc.get("page") is not None else metadata.get("page")
            )
            chunk_id = (
                doc.get("chunk_id")
                if doc.get("chunk_id") is not None
                else metadata.get("chunk_id")
            )

            header_parts = [f"[문서 {i}] source: {source}"]
            if page is not None:
                header_parts.append(f"page: {page}")
            if chunk_id is not None:
                header_parts.append(f"chunk_id: {chunk_id}")
            if score is not None:
                try:
                    header_parts.append(f"score: {float(score):.4f}")
                except (TypeError, ValueError):
                    pass

            header = " | ".join(header_parts)
            block = f"{header}\n{text}\n"

            if total_len + len(block) > limit:
                break

            blocks.append(block)
            total_len += len(block)

        return "\n".join(blocks)

    def _format_recent_messages(self, limit_turns: Optional[int] = None) -> str:
        if not self.recent_messages:
            return "(없음)"

        turns = limit_turns or self.budget.max_recent_turns
        selected_messages = self.recent_messages[-(turns * 2) :]

        lines = []
        for msg in selected_messages:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _format_collected_fields(self) -> str:
        """
        값 유무 디두플리케이션: 이미 확보된(null이 아닌) 필드만 전달하여 토큰 절약
        """
        filled_fields = [
            f"- {key}: {value}"
            for key, value in self.collected_fields.items()
            if value is not None and str(value).strip() != ""
        ]
        if not filled_fields:
            return "(수집된 정보 없음)"
        return "\n".join(filled_fields)

    def _merge_fields(self, new_fields: Dict[str, Any]) -> None:
        for key in self.collected_fields.keys():
            value = new_fields.get(key)
            if value is not None and str(value).strip() != "":
                self.collected_fields[key] = str(value).strip()

    async def _update_summary_if_needed(self) -> None:
        max_messages = self.budget.max_recent_turns * 2
        if len(self.recent_messages) <= max_messages:
            return

        old_messages = self.recent_messages[:-max_messages]
        keep_messages = self.recent_messages[-max_messages:]

        old_text = [
            f"{'사용자' if msg['role'] == 'user' else '어시스턴트'}: {msg['content']}"
            for msg in old_messages
        ]

        prompt = f"""
기존 요약:
{self.conversation_summary if self.conversation_summary else "(없음)"}

새로 요약할 이전 대화:
{chr(10).join(old_text)}

규칙:
1. 공공입찰 RFP 질의응답 핵심 맥락만 유지
2. 사업명, 기관명, 예산, 기간, 제출기한, 자격요건 등 핵심 사실 유지
3. 한국어로 300자 이내 설명 없이 요약문만 출력
""".strip()

        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        self.conversation_summary = response.output_text.strip()
        self.recent_messages = keep_messages

    def _should_skip_rewrite(self, query: str) -> bool:
        """
        Query Rewrite Skip 휴리스틱 판단 logic.
        1. 첫 대화 턴인 경우 Skip
        2. 질문의 길이가 충분히 길고 대명사/지시어가 없는 독립 문장인 경우 Skip
        """
        if not self.recent_messages:
            return True

        # 대명사 및 지시어 존재 여부 체크
        pronouns = [
            "그것",
            "이거",
            "거기",
            "이 사업",
            "해당",
            "위의",
            "아까",
            "전단계",
            "그 문서",
        ]
        has_pronoun = any(p in query for p in pronouns)

        # 질문이 30자 이상으로 상세하고 지시어가 없으면 독립 질문으로 간주
        if len(query) >= 30 and not has_pronoun:
            return True

        return False

    async def rewrite_query(self, query: str) -> str:
        # 휴리스틱에 따른 LLM 호출 Skip (비용 및 지연 시간 절감)
        if self._should_skip_rewrite(query):
            self.last_rewritten_query = query
            return query

        # 경량화된 Rewrite 프롬프트 (최근 2턴 대화만 참고)
        rewrite_prompt = f"""
너는 공공입찰 RFP 검색용 질의 재작성기다.
사용자의 질문을 검색기(retriever)가 잘 이해할 수 있는 독립적인 standalone query 한 줄로 재작성하라.

[요약 맥락]
{self.conversation_summary if self.conversation_summary else "(없음)"}

[최근 대화]
{self._format_recent_messages(limit_turns=2)}

[사용자 질문]
{query}

규칙:
- 불필요한 인사말, 예의 표현 제거
- 핵심 키워드(사업명, 기관명, 요구 항목) 보존
- 검색용 질문 한 줄만 한국어로 출력
""".strip()

        response = await self.client.responses.create(
            model=self.model,
            input=rewrite_prompt,
        )

        rewritten_query = response.output_text.strip()
        self.last_rewritten_query = rewritten_query
        return rewritten_query

    def _get_schema(self) -> Dict[str, Any]:
        """
        JSON Schema 최적화:
        citations와 evidence_quotes를 단일 evidence 객체 배열로 통합하여 Output 토큰 최소화.
        """
        return {
            "type": "json_schema",
            "name": "rfp_rag_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer": {"type": "string"},
                    "summary": {"type": "string"},
                    "fields": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "project_name": {"type": ["string", "null"]},
                            "organization": {"type": ["string", "null"]},
                            "budget": {"type": ["string", "null"]},
                            "duration": {"type": ["string", "null"]},
                            "deadline": {"type": ["string", "null"]},
                            "submission_method": {"type": ["string", "null"]},
                            "required_technology": {"type": ["string", "null"]},
                            "eligibility": {"type": ["string", "null"]},
                            "evaluation_criteria": {"type": ["string", "null"]},
                            "required_documents": {"type": ["string", "null"]},
                            "notes": {"type": ["string", "null"]},
                        },
                        "required": [
                            "project_name",
                            "organization",
                            "budget",
                            "duration",
                            "deadline",
                            "submission_method",
                            "required_technology",
                            "eligibility",
                            "evaluation_criteria",
                            "required_documents",
                            "notes",
                        ],
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "string"},
                                "page": {"type": ["integer", "null"]},
                                "chunk_id": {"type": ["string", "integer", "null"]},
                                "quote": {"type": "string"},
                                "score": {"type": ["number", "null"]},
                            },
                            "required": [
                                "source",
                                "page",
                                "chunk_id",
                                "quote",
                                "score",
                            ],
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "needs_clarification": {"type": "boolean"},
                    "clarification_question": {"type": ["string", "null"]},
                    "conflicts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "answer",
                    "summary",
                    "fields",
                    "evidence",
                    "confidence",
                    "needs_clarification",
                    "clarification_question",
                    "conflicts",
                ],
            },
        }

    def _build_user_input(
        self,
        query: str,
        context: str,
        rewritten_query: Optional[str] = None,
    ) -> str:
        """
        Prompt Caching 최적화 구조:
        [1. 고정 시스템 지침 & 배경 (Prefix)] -> [2. 요약/대화 이력] -> [3. 가변 동적 컨텍스트 & 질문 (Suffix)]
        """
        rewrite_text = rewritten_query or self.last_rewritten_query or "(없음)"

        return f"""[프로젝트 배경 및 역할]
    - 서비스: 입찰메이트 (공공입찰 RFP 핵심 정보 분석 시스템)
    - 관심 필드: 사업명, 발주기관, 예산, 수행기간, 제출기한, 제출방법, 요구기술, 참가자격, 평가기준, 필수서류, 유의사항

    [응답 가이드라인]
    1. 제공된 [현재 검색 문맥]을 최우선 근거로 답하세요.
    2. [사용자 질문]에서 직접적으로 물어본 내용에 대해서만 정확히 답변하세요. 질문에서 묻지 않은 정보(예: 수행기간, 사업명 등)는 답변에 포함하지 마세요.
    3. 문서에서 확인 불가능한 정보는 추측하지 말고 '문서에서 확인 불가'로 명시하세요.
    4. evidence에는 인용한 문서의 source, page, chunk_id, quote, score를 정확히 기재하세요.
    5. fields에는 이번 턴 검색 문맥에서 새로 확인된 필드만 추출하세요. (단, answer 본문 출력에는 영향 없이 필드 저장 용도로만 사용)
    6. 기존 수집된 필드 정보와 검색 문맥 정보가 충돌할 경우 conflicts 배열에 명시하세요.
    7. 장황한 수식어 없이 핵심 위주로 간결하게 한국어로 답변하세요.

    [누적 맥락 정보]
    - 이전 대화 요약: {self.conversation_summary if self.conversation_summary else "(없음)"}
    - 이미 확보된 핵심 필드:
    {self._format_collected_fields()}

    [최근 대화 이력]
    {self._format_recent_messages()}

    [검색용 재작성 질문]
    {rewrite_text}

    [현재 검색 문맥]
    {context if context.strip() else "(검색 문맥 없음)"}

    [사용자 질문]
    {query}
    """.strip()

    async def ask(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        rewritten_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._update_summary_if_needed()

        context = self.build_context(retrieved_docs)
        user_input = self._build_user_input(
            query=query,
            context=context,
            rewritten_query=rewritten_query,
        )
        schema = self._get_schema()

        instructions = (
            "당신은 공공입찰 제안요청서(RFP) 분석 어시스턴트입니다. "
            "제공된 검색 문맥에 기반하여 사실에 기반한 정밀한 답변을 간결하게 제공하세요."
        )

        req: Dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "text": {"format": schema},
            "store": True,
        }

        if self.previous_response_id:
            req["previous_response_id"] = self.previous_response_id

        response = await self.client.responses.create(**req)
        self.previous_response_id = response.id

        result = json.loads(response.output_text)
        self._merge_fields(result.get("fields", {}))

        self.recent_messages.append({"role": "user", "content": query})
        self.recent_messages.append(
            {"role": "assistant", "content": result.get("answer", "")}
        )

        return result
