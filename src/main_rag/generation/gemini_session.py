"""Gemini implementation of the BidMate session contract used by Main RAG."""

from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.main_rag.generation.generate_answer import BidMateRAGSession, TokenBudget


class GeminiEvidence(BaseModel):
    source: str
    page: int | None
    chunk_id: str | int | None
    quote: str
    score: float | None


class GeminiFields(BaseModel):
    project_name: str | None
    organization: str | None
    budget: str | None
    duration: str | None
    deadline: str | None
    submission_method: str | None
    required_technology: str | None
    eligibility: str | None
    evaluation_criteria: str | None
    required_documents: str | None
    notes: str | None


class GeminiRAGResponse(BaseModel):
    answer: str
    summary: str
    fields: GeminiFields
    evidence: list[GeminiEvidence]
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool
    clarification_question: str | None
    conflicts: list[str]


class GeminiBidMateRAGSession(BidMateRAGSession):
    """Retain BidMate context/state behavior while using Gemini generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_context_chars: int = 7000,
        max_recent_turns: int = 4,
        max_docs: int = 6,
        min_score: float = 0.0,
        max_output_tokens: int = 4000,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.budget = TokenBudget(
            max_context_chars=max_context_chars,
            max_recent_turns=max_recent_turns,
            max_docs=max_docs,
            min_score=min_score,
        )
        self.previous_response_id = None
        self.recent_messages = []
        self.conversation_summary = ""
        self.last_rewritten_query = None
        self.collected_fields = {
            "project_name": None, "organization": None, "budget": None,
            "duration": None, "deadline": None, "submission_method": None,
            "required_technology": None, "eligibility": None,
            "evaluation_criteria": None, "required_documents": None, "notes": None,
        }

    async def rewrite_query(self, query: str) -> str:
        if self._should_skip_rewrite(query):
            self.last_rewritten_query = query
            return query
        prompt = (
            "공공입찰 RFP 검색용 독립 질문 한 줄로 재작성하라. 설명은 쓰지 마라.\n"
            f"이전 요약: {self.conversation_summary or '(없음)'}\n"
            f"최근 대화:\n{self._format_recent_messages(limit_turns=2)}\n질문: {query}"
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=256),
        )
        rewritten = (response.text or query).strip()
        self.last_rewritten_query = rewritten
        return rewritten

    async def ask(self, query: str, retrieved_docs: list[dict[str, Any]], rewritten_query: str | None = None) -> dict[str, Any]:
        context = self.build_context(retrieved_docs)
        user_input = self._build_user_input(query, context, rewritten_query)
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "당신은 공공입찰 RFP 분석 어시스턴트다. 제공된 검색 문맥만 근거로 "
                    "간결하게 답하고 evidence의 chunk_id를 문맥에 표시된 값 그대로 쓴다."
                ),
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_schema=GeminiRAGResponse,
            ),
        )
        parsed = GeminiRAGResponse.model_validate_json(response.text or "{}")
        result = parsed.model_dump()
        usage = getattr(response, "usage_metadata", None)
        result["_usage"] = {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        }
        self._merge_fields(result["fields"])
        self.recent_messages.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["answer"]},
        ])
        return result
