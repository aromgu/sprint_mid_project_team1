from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from src.generation.models import ModelAnswer
from src.generation.openai_generator import OpenAIRAGService
from src.search.service import SearchService

from backend.models import (
    ActionItem, DeliverableItem, DeliverablesResponse, EligibilityItem,
    EligibilityResponse, Evidence, OverviewResponse, RequirementItem,
    RequirementsResponse, RiskItem, RisksResponse,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RAGClient:
    def __init__(self, search_service: SearchService | None = None, generator: OpenAIRAGService | None = None) -> None:
        self.search = search_service or SearchService()
        self.generator = generator or OpenAIRAGService(search_service=self.search)
        self._analysis_cache: dict[tuple[str, str], Any] = {}

    @staticmethod
    def _evidence(result, score: float | None = None) -> Evidence:
        chunk = result.chunk
        return Evidence(
            document_name=chunk.document_title, page_number=chunk.page_start,
            quote=(chunk.text[:300].replace("\n", " ") + ("…" if len(chunk.text) > 300 else "")),
            score=max(0.0, min(1.0, float(score if score is not None else result.score))),
            chunk_id=chunk.chunk_id, requirement_ids=chunk.requirement_ids,
        )

    def _retrieve(
        self, document_id: str, query: str, top_k: int = 5,
        content_types: set[str] | None = None,
    ):
        results = self.search.search(
            query, top_k=top_k, document_ids={document_id}, content_types=content_types,
        )
        return results

    def _retrieve_deliverables(self, document_id: str) -> list:
        """Collect both bid-time submissions and project-time outputs.

        A single keyword-heavy query tended to favor requirement tables and miss
        narrative clauses such as the start report and periodic reports.
        """
        queries = (
            "제출하여야 한다 제출해야 한다 제출서류 첨부서류 원본 사본 날인",
            "착수신고서 사업수행계획서 월간보고서 중간보고서 최종보고서 서약서",
            "산출정보 산출물 성과품 결과물 매뉴얼 소스코드 납품 인계",
        )
        merged: dict[str, Any] = {}
        for query in queries:
            for result in self._retrieve(document_id, query, 10):
                merged.setdefault(result.chunk.chunk_id, result)
        return list(merged.values())[:20]

    def _parse_structured(self, context: str, schema: type[T], prompt: str) -> T:
        """Run Workspace card extraction with the configured default provider."""
        provider = os.getenv("RAG_WORKSPACE_LLM_PROVIDER", "gemini-lite")
        settings = self.generator.config[provider]
        client = self.generator._get_client(provider)
        evidence_instruction = (
            "\n각 카드 항목의 source_ids에는 그 내용을 직접 뒷받침하는 출처 라벨"
            "([S1], [S2] 형식)만 넣어라. 근거가 없으면 빈 배열로 두어라."
        )
        prompt = prompt + evidence_instruction
        if provider == "openai":
            response = client.responses.parse(
                model=self.generator.model_name_for(provider),
                reasoning={"effort": settings.get("reasoning_effort", "minimal")},
                max_output_tokens=max(1200, settings.get("max_output_tokens", 1200)),
                instructions=prompt,
                input=context,
                text_format=schema,
            )
            parsed = response.output_parsed
        else:
            from google.genai import types

            response = client.models.generate_content(
                model=self.generator.model_name_for(provider),
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    max_output_tokens=max(1200, settings.get("max_output_tokens", 1200)),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            parsed = schema.model_validate_json(response.text)
        if parsed is None:
            raise RuntimeError("Structured analysis response was empty")
        return parsed

    @classmethod
    def _attach_source_evidence(cls, value: Any, source_map: dict[str, Any]) -> None:
        """Resolve model-selected source labels to the exact retrieved chunk."""
        for field in ("action_items", "risks", "items"):
            for item in getattr(value, field, []) or []:
                labels = getattr(item, "source_ids", []) or []
                result = next((source_map[label] for label in labels if label in source_map), None)
                item.evidence = cls._evidence(result) if result is not None else None

    @staticmethod
    def _excerpt(result, limit: int = 280) -> str:
        text = re.sub(r"\s+", " ", result.chunk.text).strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    @classmethod
    def _fallback_deliverables(cls, results: list) -> list[DeliverableItem]:
        items = []
        for index, result in enumerate(results[:6], 1):
            text = result.chunk.text
            section = next((part for part in reversed(result.chunk.section_path) if part.strip()), "")
            quantity_match = re.search(r"(\d+)\s*(?:부|식|개)", text)
            format_name = "PDF/전자파일" if re.search(r"PDF|전자(?:파일|문서|제출)", text, re.I) else (
                "서면" if re.search(r"서면|인쇄|책자", text) else "형식 확인 필요"
            )
            project_output = bool(re.search(
                r"착수|수행계획|월간|주간|중간보고|최종보고|완료보고|산출정보|산출물|성과품|결과물|매뉴얼|소스코드|납품|인계",
                f"{section} {text}",
            ))
            items.append(DeliverableItem(
                id=f"deliverable-review-{index}",
                name=section[:80] or f"제출물·산출물 검토 {index}",
                kind="project_deliverable" if project_output else "bid_submission",
                description=cls._excerpt(result),
                format=format_name,
                quantity=int(quantity_match.group(1)) if quantity_match else 1,
                requires_seal=bool(re.search(r"날인|직인", text)),
                requires_original=bool(re.search(r"원본", text)),
                status="pending",
                evidence=cls._evidence(result),
            ))
        return items

    @classmethod
    def _fallback_requirements(cls, results: list) -> list[RequirementItem]:
        category_by_prefix = {
            "SFR": "functional", "PER": "performance", "SER": "security",
            "QUR": "performance", "PMR": "personnel", "COR": "contract",
        }
        items = []
        for index, result in enumerate(results[:8], 1):
            requirement_id = result.chunk.requirement_ids[0] if result.chunk.requirement_ids else f"review-{index}"
            prefix = requirement_id.split("-", 1)[0].upper()
            section = next((part for part in reversed(result.chunk.section_path) if part.strip()), "")
            items.append(RequirementItem(
                id=requirement_id,
                category=category_by_prefix.get(prefix, "operation"),
                title=section[:100] or f"요구사항 {requirement_id}",
                description=cls._excerpt(result),
                priority="medium",
                review_status="pending",
                evidence=cls._evidence(result),
            ))
        return items

    def _extract(self, document_id: str, query: str, schema: type[T], prompt: str, top_k: int = 5) -> T:
        cache_key = (document_id, schema.__name__)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key].model_copy(deep=True)
        results = self._retrieve(document_id, query, top_k)
        if not results:
            value = schema(document_id=document_id)  # type: ignore[call-arg]
            self._analysis_cache[cache_key] = value
            return value.model_copy(deep=True)
        context, source_map = self.generator._build_context(results)
        parsed = self._parse_structured(f"문서 ID: {document_id}\n출처:\n{context}", schema, prompt)
        self._attach_source_evidence(parsed, source_map)
        self._analysis_cache[cache_key] = parsed
        return parsed.model_copy(deep=True)

    def answer(
        self, document_id: str, question: str,
        chat_history: list[dict[str, str]] | None = None, provider: str = "openai",
    ):
        history_key = "|".join(f"{item.get('role')}:{item.get('content')}" for item in (chat_history or [])[-6:])
        cache_key = (document_id, f"answer:{provider}:{history_key}:{question.strip()}")
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key].model_copy(deep=True)
        result = self.generator.answer(
            question, document_ids={document_id}, chat_history=(chat_history or [])[-6:],
            provider=provider,
        )
        self._analysis_cache[cache_key] = result
        return result.model_copy(deep=True)

    def overview(self, document_id: str) -> OverviewResponse:
        cache_key = (document_id, "OverviewResponse")
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key].model_copy(deep=True)
        schema = type("OverviewExtraction", (OverviewResponse,), {})
        prompt = """RFP 문서의 제공된 출처를 먼저 확인하고 Overview의 모든 필드를 가능한 한 채워라.
제출 마감일과 질의 마감일은 문서에 명시된 날짜·시간을 그대로 문자열로 넣어라.
참가 자격은 eligible, ineligible, review_required 중 하나를 선택하라.
위험 건수와 제출물 진행률은 출처에서 확인되는 수를 세고, 확인할 수 없을 때만 0을 사용하라.
즉시 조치가 명시되면 action_items에 넣어라. 근거 없는 사실은 만들지 말되, 단순히 모든 필드를 기본값으로 비워 두지 마라.
action_items의 evidence는 비워 두고, 직접 근거가 되는 출처 라벨은 source_ids에 넣어라."""
        results = self._retrieve(document_id, "제출 마감 질의 마감 참가 자격 실격 감점 불이익 위약금 제재 제출물", 10)
        if not results:
            value = OverviewResponse(document_id=document_id)
            self._analysis_cache[cache_key] = value
            return value.model_copy(deep=True)
        # Model extraction is deliberately kept conservative; evidence is attached by code.
        context, source_map = self.generator._build_context(results)
        parsed = self._parse_structured(context, OverviewResponse, prompt)
        self._attach_source_evidence(parsed, source_map)
        parsed.document_id = document_id
        parsed.confidence = min(1.0, max(0.0, len(results) / 8))
        self._analysis_cache[cache_key] = parsed
        return parsed.model_copy(deep=True)

    def risks(self, document_id: str) -> RisksResponse:
        risk_query = "실격 무효 제외 감점 불이익 위약금 손해배상 지체상금 비용부담 재작업 계약해지 제재 입찰참가 제한 보안위반"
        results = self._retrieve(document_id, risk_query, 12)
        if not results:
            return RisksResponse(document_id=document_id)
        prompt = """문서에서 실제 입찰 참여 회사에 부정적인 영향을 줄 수 있는 위험을 빠짐없이 추출하라.
- disqualification: 입찰 무효, 평가 제외, 참가 제한, 자격 박탈처럼 참여 또는 선정이 불가능해지는 조건
- deduction: 평가점수 감점뿐 아니라 위약금, 지체상금, 손해배상, 계약상대자 비용 부담, 무상 재작업·보완, 대금 삭감, 계약 해지와 기타 금전·계약상 불이익
- review: 불이익 가능성은 있으나 적용 조건이나 의무 여부가 원문에서 불명확하여 확인이 필요한 항목
단순한 일반 의무를 모두 위험으로 만들지 말고, 위반·미충족 시 회사에 발생하는 구체적인 부정적 결과가 원문에 있는 항목만 포함하라.
기능 요구사항에서 다른 심사 업무의 가·감점 기능을 구축하라는 내용은 이 입찰 참여 회사에 적용되는 불이익이 아니므로 제외하라.
각 항목에는 id,type,severity,title,description을 넣고 evidence는 null로 둬라. description에는 발생 조건과 회사에 미치는 결과를 함께 명시하라. 근거 없는 위험은 만들지 마라."""
        data = self._extract(document_id, risk_query, RisksResponse, prompt, top_k=12)
        data.document_id = document_id
        return data

    def eligibility(self, document_id: str) -> EligibilityResponse:
        prompt = "참가 자격과 필수 자격 조건을 추출하라. user_status는 unchecked로 두고, 근거 없는 조건은 만들지 마라."
        data = self._extract(document_id, "참가 자격 자격 요건 신청 조건", EligibilityResponse, prompt)
        results = self._retrieve(document_id, "참가 자격 자격 요건 신청 조건", max(5, len(data.items)))
        if not data.items and results:
            # Keep the MVP useful when the model conservatively returns an empty list.
            # This is explicitly a review item, not an asserted eligibility decision.
            data.items = [EligibilityItem(
                id=f"eligibility-review-{index}",
                title="참가 자격 확인 필요",
                description=result.chunk.text[:240].replace("\n", " "),
                user_status="unchecked",
                evidence=self._evidence(result),
            ) for index, result in enumerate(results[:5], 1)]
        data.document_id = document_id
        return data

    def deliverables(self, document_id: str) -> DeliverablesResponse:
        prompt = """필수 제출물과 산출물을 항목별로 하나씩 분리해 추출하라. 여러 서류가 한 문장이나 표 셀에 있어도 각각 별도 items 항목으로 만든다.
kind는 입찰·제안 단계에 제출하는 제안서, 증명서, 서약서, 입찰서류이면 bid_submission으로, 계약 이후 착수·수행·보고·검수·납품 단계에 만드는 계획서, 보고서, 매뉴얼, 소스코드, 성과품이면 project_deliverable로 분류하라.
format,quantity,requires_seal,requires_original,status 필드를 보수적으로 채우고 근거 없는 항목은 만들지 마라."""
        results = self._retrieve_deliverables(document_id)
        # Keep operational tabs responsive even when the external LLM is slow or
        # rate-limited. Optional enrichment can be enabled explicitly.
        if results and os.getenv("RAG_WORKSPACE_ENRICH_CARDS", "false").casefold() not in {"1", "true", "yes", "on"}:
            data = DeliverablesResponse(document_id=document_id, items=self._fallback_deliverables(results))
            self._analysis_cache[(document_id, "DeliverablesResponse")] = data
            return data.model_copy(deep=True)
        try:
            context, source_map = self.generator._build_context(results)
            data = self._parse_structured(
                f"문서 ID: {document_id}\n출처:\n{context}", DeliverablesResponse, prompt,
            ) if results else DeliverablesResponse(document_id=document_id)
            self._attach_source_evidence(data, source_map if results else {})
        except Exception:
            logger.exception("deliverable model extraction failed; using retrieval fallback")
            data = DeliverablesResponse(document_id=document_id)
        if not data.items and results:
            data.items = self._fallback_deliverables(results)
        data.document_id = document_id
        # Empty retrieval can be a transient index/mapping failure. Do not cache
        # it as if the source document had definitively contained no submissions.
        if data.items:
            self._analysis_cache[(document_id, "DeliverablesResponse")] = data
        return data

    def requirements(self, document_id: str) -> RequirementsResponse:
        prompt = "RFP 요구사항을 추출하라. category는 functional,performance,security,operation,personnel,output,contract 중 하나를 사용하라. 근거 없는 요구사항은 만들지 마라."
        results = self._retrieve(
            document_id, "요구사항 고유번호 요구사항 정의 기능 성능 보안 운영",
            10, content_types={"requirement", "requirement_table_row"},
        )
        if not results:
            results = self._retrieve(document_id, "요구사항 기능 성능 보안 운영 인력 산출물 계약", 8)
        if results and os.getenv("RAG_WORKSPACE_ENRICH_CARDS", "false").casefold() not in {"1", "true", "yes", "on"}:
            data = RequirementsResponse(document_id=document_id, items=self._fallback_requirements(results))
            self._analysis_cache[(document_id, "RequirementsResponse")] = data
            return data.model_copy(deep=True)
        try:
            data = self._extract(document_id, "요구사항 기능 성능 보안 운영 인력 산출물 계약", RequirementsResponse, prompt)
        except Exception:
            logger.exception("requirement model extraction failed; using retrieval fallback")
            data = RequirementsResponse(document_id=document_id)
        if not data.items and results:
            data.items = self._fallback_requirements(results)
        data.document_id = document_id
        self._analysis_cache[(document_id, "RequirementsResponse")] = data
        return data
