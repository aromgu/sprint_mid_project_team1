"""
NAIVE RAG generate code

입찰메이트(BidMate) 프로젝트용 RAG 세션 모듈.

역할:
- 공공입찰 RFP 문서에 대한 질의응답을 담당하는 세션 클래스를 정의한다.
- 외부에서 전달된 검색 결과(retrieved docs)를 하나의 텍스트 context로 구성한다.
- GPT-5-NANO LLM을 호출하여, RFP 관련 질문에 대한 답변을 생성한다.
- Structured Output(JSON Schema)를 사용해 예산, 기간, 제출기한 등 핵심 필드를
  구조화된 형태로 추출한다.
- previous_response_id를 이용해 멀티턴(다중 턴) 대화 상태를 유지한다.

주의:
- 이 파일은 재사용 가능한 소스 코드만 포함한다.
- 샘플 입력, print 테스트, 실행용 코드는 main 스크립트에 분리해서 작성한다.
"""

# OpenAI API를 사용하기 위한 클라이언트 클래스 import
# 모델이 반환한 JSON 문자열을 파이썬 dict로 바꾸기 위해 import
import json

# 타입 힌트를 위한 typing 모듈 import
from typing import Any, Dict, List, Optional

from openai import OpenAI


class BidMateRAGSession:
    """
    입찰메이트 사내 RAG용 세션 클래스

    이 클래스는 다음 역할을 담당한다.
    1. 검색된 문서 조각(retrieved docs)을 하나의 context 문자열로 합친다.
    2. GPT-5-NANO 모델에 질의한다.
    3. Structured Output(JSON Schema) 형태로 응답을 받는다.
    4. previous_response_id를 저장해서 멀티턴 대화를 유지한다.

    이 파일은 '소스 파일'이므로
    실제 실행 테스트 코드(print, 샘플 입력 등)는 넣지 않는다.
    """

    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        """
        세션 객체 초기화

        Parameters
        ----------
        api_key : str
            OpenAI API 키
        model : str
            사용할 모델명, 기본값은 gpt-5-nano
        """

        # OpenAI 클라이언트 생성
        self.client = OpenAI(api_key=api_key)

        # 사용할 모델명 저장
        self.model = model

        # 멀티턴 대화 상태 유지를 위한 이전 response id 저장 변수
        # 첫 질문 전에는 이전 응답이 없으므로 None
        self.previous_response_id: Optional[str] = None

    def reset(self):
        """
        멀티턴 세션 초기화 함수

        새 대화를 시작하고 싶을 때 호출하면 된다.
        previous_response_id를 None으로 되돌린다.
        """

        self.previous_response_id = None

    def build_context(
        self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 7000
    ) -> str:
        blocks = []
        total_len = 0

        for i, doc in enumerate(retrieved_docs, start=1):
            text = doc.get("text", "").strip()

            source = doc.get("file_nm")
            score = doc.get("score")

            metadata = doc.get("metadata", {}) or {}

            page = doc.get("page")
            if page is None:
                page = metadata.get("page")

            chunk_id = doc.get("chunk_id")
            if chunk_id is None:
                chunk_id = metadata.get("chunk_id")

            header = f"[문서 {i}] source: {source}"

            if page is not None:
                header += f" | page: {page}"

            if chunk_id is not None:
                header += f" | chunk_id: {chunk_id}"

            if score is not None:
                header += f" | score: {score:.4f}"

            block = f"{header}\n{text}\n"

            if total_len + len(block) > max_chars:
                break

            blocks.append(block)
            total_len += len(block)

        return "\n".join(blocks)

    def ask(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        사용자의 질문과 검색 문맥을 바탕으로 RAG 응답을 생성한다.

        Parameters
        ----------
        query : str
            사용자 질문
        retrieved_docs : List[Dict[str, Any]]
            retriever가 뽑아준 문서 조각 리스트

        Returns
        -------
        Dict[str, Any]
            JSON Schema에 맞는 구조화 응답 dict
        """

        # 검색 결과들을 하나의 context 문자열로 합침
        context = self.build_context(retrieved_docs)

        # 모델에게 줄 시스템 수준 지침
        instructions = (
            "당신은 공공입찰 제안요청서(RFP) 분석을 돕는 사내 RAG 어시스턴트입니다. "
            "반드시 현재 턴에 제공된 검색 문맥을 최우선 근거로 사용하세요. "
            "이전 대화는 질문 해석에만 참고하고, 사실 판단은 현재 검색 문맥 기준으로 하세요. "
            "문맥에 없는 내용은 추측하지 말고 '문맥에서 확인 불가'라고 답하세요. "
            "답변은 반드시 한국어로 작성하세요."
        )

        # 사용자 입력 메시지 구성
        # 프로젝트 배경 + 현재 context + 질문 + 응답 규칙을 함께 전달
        # 사용자 입력 메시지 구성
        user_input = f"""user_input =
        [프로젝트 배경]
        - 서비스명: 입찰메이트
        - 목적: 공공입찰 RFP 문서에서 핵심 정보를 빠르게 추출/요약/질의응답
        - 주요 관심 정보: 사업명, 발주기관, 예산, 수행기간, 제출기한, 제출방법, 요구기술, 참가자격, 평가기준, 필수서류, 유의사항

        [현재 검색 문맥]
        {context}

        [사용자 질문]
        {query}

        [응답 지침]
        1. 반드시 현재 검색 문맥만 근거로 답할 것
        2. 정보가 부족하면 '문맥에서 확인 불가'로 쓸 것
        3. 직접 답변 + 요약 + 필드 추출 + 실제 citation['source'] 기반 근거를 포함할 것
        4. citations에는 반드시 source, page, chunk_id, score를 넣을 것
        5. citations의 source, page, chunk_id, score는 반드시 [현재 검색 문맥] 각 문서 헤더 값을 그대로 복사할 것
        6. score가 문서 헤더에 있으면 반드시 숫자로 반환할 것
        7. 값이 문맥 헤더에 없을 때만 null을 반환할 것
        8. evidence_quotes에도 가능하면 source, page, chunk_id를 포함할 것
        9. confidence는 0.0~1.0 사이 숫자로 반환할 것
        10. 질문이 애매하면 needs_clarification=true로 설정할 것
        11. 문서 간 충돌 정보가 있으면 conflicts에 기록할 것
"""

        # Structured Output용 JSON Schema 정의
        # 모델 출력 형식을 강제해서 후처리 안정성을 높임
        schema = {
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
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "string"},
                                "page": {"type": ["integer", "null"]},
                                "chunk_id": {"type": ["string", "integer", "null"]},
                                "score": {"type": ["number", "null"]},
                            },
                            "required": [
                                "source",
                                "page",
                                "chunk_id",
                                "score",
                            ],
                        },
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "string"},
                                "page": {"type": ["integer", "null"]},
                                "chunk_id": {"type": ["string", "integer", "null"]},
                                "quote": {"type": "string"},
                            },
                            "required": ["source", "page", "chunk_id", "quote"],
                        },
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "needs_clarification": {"type": "boolean"},
                    "clarification_question": {"type": ["string", "null"]},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "answer",
                    "summary",
                    "fields",
                    "citations",
                    "evidence_quotes",
                    "confidence",
                    "needs_clarification",
                    "clarification_question",
                    "conflicts",
                ],
            },
        }
        # OpenAI Responses API 요청 파라미터 구성
        req = {
            "model": self.model,  # 사용할 모델
            "instructions": instructions,  # 시스템 지침
            "input": user_input,  # 사용자 입력
            "text": {"format": schema},  # 구조화 출력 스키마
            "store": True,  # 멀티턴 상태 저장용
        }

        # 이전 턴이 있다면 previous_response_id를 함께 넣어서 멀티턴 연결
        if self.previous_response_id:
            req["previous_response_id"] = self.previous_response_id

        # 실제 API 호출
        response = self.client.responses.create(**req)

        # 이번 응답 ID를 저장해서 다음 턴에서 이어쓰기 가능하게 함
        self.previous_response_id = response.id

        # 모델 응답 텍스트를 JSON 문자열로 받아 dict로 변환
        return json.loads(response.output_text)
