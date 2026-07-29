from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from src.generation.models import ModelAnswer
from src.generation.openai_generator import OpenAIRAGService
from src.search.service import SearchService
from src.search.models import SearchChunk, SearchResult
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.runtime import MainAdvancedSessionManager
from src.main_rag.answerability import classify_answer_status, is_answerable_status

from backend.models import (
    ActionItem, DeliverableItem, DeliverablesResponse, EligibilityItem,
    EligibilityResponse, Evidence, OverviewResponse, RequirementItem,
    RequirementsResponse, RiskItem, RisksResponse,
)
from src.generation.models import AnswerResponse, Citation

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RAGClient:
    def __init__(
        self,
        search_service: SearchService | None = None,
        generator: OpenAIRAGService | None = None,
        advanced_retriever: AdvancedRetriever | None = None,
        session_manager: MainAdvancedSessionManager | None = None,
    ) -> None:
        self.search = search_service or SearchService()
        self.generator = generator or OpenAIRAGService(search_service=self.search)
        self.advanced_retriever = advanced_retriever or AdvancedRetriever()
        self.sessions = session_manager or MainAdvancedSessionManager(retriever=self.advanced_retriever)
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
        self, document_id: str, query: str, top_k: int = 10,
        content_types: set[str] | None = None,
    ):
        documents = self.advanced_retriever.search_documents(
            query, top_k=top_k, document_id=document_id,
        )
        # Preserve compatibility for documents uploaded before incremental
        # Advanced indexing was introduced.
        if not documents and not document_id.startswith("eval_"):
            return self.search.search(
                query, top_k=top_k, document_ids={document_id}, content_types=content_types,
            )
        results = []
        for rank, document in enumerate(documents, 1):
            metadata = document.get("metadata") or {}
            content_type = str(metadata.get("content_type") or "text")
            if content_types and content_type not in content_types:
                continue
            section = metadata.get("section_path") or []
            if isinstance(section, str):
                section = [section] if section.strip() else []
            requirement_ids = re.findall(r"\b[A-Z]{2,5}-\d{2,5}\b", document.get("text") or "")
            page_start = int(document.get("page") or metadata.get("page_start") or 1)
            page_end = int(metadata.get("page_end") or page_start)
            chunk = SearchChunk(
                chunk_id=str(document["chunk_id"]),
                document_id=str(metadata.get("document_id") or document_id),
                document_title=str(document.get("file_nm") or metadata.get("project_name") or document_id),
                page_start=page_start,
                page_end=page_end,
                section_path=list(section),
                requirement_ids=list(dict.fromkeys(requirement_ids)),
                content_type=content_type,
                text=str(document.get("text") or ""),
                token_count=int(metadata.get("token_count") or 0),
            )
            results.append(SearchResult(
                chunk=chunk, rank=rank, score=float(document.get("score") or 0),
                retriever="main_advanced_dense",
            ))
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
        provider = os.getenv("RAG_WORKSPACE_LLM_PROVIDER", "openai")
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
                normalized_labels = [str(label).strip().strip("[]").strip() for label in labels]
                result = next(
                    (source_map[label] for label in normalized_labels if label in source_map),
                    None,
                )
                if result is None and source_map:
                    # Structured models occasionally omit source_ids even when
                    # the card text came directly from the supplied context.
                    # Recover the closest retrieved chunk so UI cards never
                    # become dead ends without silently inventing page data.
                    item_tokens = set(re.findall(
                        r"[0-9A-Za-z가-힣]{2,}",
                        f"{getattr(item, 'title', '')} {getattr(item, 'name', '')} {getattr(item, 'description', '')}".casefold(),
                    ))
                    candidates = list(source_map.values())
                    result = max(
                        candidates,
                        key=lambda candidate: (
                            len(item_tokens & set(re.findall(
                                r"[0-9A-Za-z가-힣]{2,}", candidate.chunk.text.casefold(),
                            ))),
                            -candidate.rank,
                        ),
                    )
                item.evidence = cls._evidence(result) if result is not None else None

    @staticmethod
    def _excerpt(result, limit: int = 280) -> str:
        text = re.sub(r"\s+", " ", result.chunk.text).strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    @classmethod
    def _fallback_deliverables(cls, results: list) -> list[DeliverableItem]:
        items_by_title: dict[str, DeliverableItem] = {}
        for index, result in enumerate(results, 1):
            text = result.chunk.text
            section = next((part for part in reversed(result.chunk.section_path) if part.strip()), "")
            parsed = cls._parse_requirement_table(text)
            title = cls._deliverable_title(text) or parsed["title"] or section[:80] or f"제출물·산출물 검토 {index}"
            description = parsed["description"] or cls._clean_card_summary(text)
            quantity_match = re.search(r"(\d+)\s*(?:부|식|개)", text)
            format_name = "PDF/전자파일" if re.search(r"PDF|전자(?:파일|문서|제출)", text, re.I) else (
                "서면" if re.search(r"서면|인쇄|책자", text) else "형식 확인 필요"
            )
            project_output = bool(re.search(
                r"착수|수행계획|월간|주간|중간보고|최종보고|완료보고|산출정보|산출물|성과품|결과물|매뉴얼|소스코드|납품|인계",
                f"{section} {text}",
            ))
            item = DeliverableItem(
                id=parsed["id"] or f"deliverable-review-{index}",
                name=title,
                kind="project_deliverable" if project_output else "bid_submission",
                description=description,
                format=format_name,
                quantity=int(quantity_match.group(1)) if quantity_match else 1,
                requires_seal=bool(re.search(r"날인|직인", text)),
                requires_original=bool(re.search(r"원본", text)),
                status="pending",
                evidence=cls._evidence(result),
            )
            key = re.sub(r"\s+", "", title).casefold()
            previous = items_by_title.get(key)
            if previous is None or len(item.description) > len(previous.description):
                items_by_title[key] = item
        return list(items_by_title.values())[:8]

    @staticmethod
    def _deliverable_title(text: str) -> str:
        names = (
            "사업수행계획서", "착수신고서", "입찰가격제안서", "가격제안서",
            "제안서 원본", "제안서 사본", "발표자료", "최종완료보고서", "완료보고서",
            "최종보고서", "중간보고서", "월간보고서", "주간보고서", "수시보고서",
            "교육계획서", "교육자료", "사용자 매뉴얼", "관리자 매뉴얼", "운영자 매뉴얼",
            "운영지침서", "소스코드", "보안서약서", "실적증명서",
            "신용평가등급확인서", "기술적용계획표",
        )
        return next((name for name in names if name in text), "")

    @staticmethod
    def _clean_card_summary(text: str, limit: int = 260) -> str:
        cells = []
        boilerplate = (
            "요구사항 분류", "요구사항 고유번호", "요구사항 명칭", "요구사항 상세설명",
            "산출정보", "관련요구사항", "---",
        )
        for raw_line in text.splitlines():
            for cell in raw_line.split("|"):
                value = re.sub(r"\s+", " ", cell).strip(" -:\t")
                if value and not any(value == label for label in boilerplate):
                    cells.append(value)
        summary = " ".join(cells)
        summary = re.sub(r"\s+/\s+", " ", summary)
        summary = re.sub(r"\s+", " ", summary).strip()
        return summary[:limit] + ("…" if len(summary) > limit else "")

    @classmethod
    def _fallback_requirements(cls, results: list) -> list[RequirementItem]:
        category_by_prefix = {
            "SFR": "functional", "PER": "performance", "SER": "security",
            "QUR": "quality", "SIR": "interface", "DAR": "data", "TER": "quality",
            "PMR": "project", "PSR": "operation", "COR": "contract",
        }
        items_by_id: dict[str, RequirementItem] = {}
        for index, result in enumerate(results, 1):
            parsed = cls._parse_requirement_table(result.chunk.text)
            requirement_id = parsed["id"] or (
                result.chunk.requirement_ids[0] if result.chunk.requirement_ids else f"review-{index}"
            )
            prefix = requirement_id.split("-", 1)[0].upper()
            section = next((part for part in reversed(result.chunk.section_path) if part.strip()), "")
            title = parsed["title"] or section[:100] or f"요구사항 {requirement_id}"
            description = parsed["description"] or cls._excerpt(result)
            item = RequirementItem(
                id=requirement_id,
                category=category_by_prefix.get(prefix, "operation"),
                title=title,
                description=description,
                priority="medium",
                review_status="pending",
                evidence=cls._evidence(result),
            )
            previous = items_by_id.get(requirement_id)
            if previous is None or len(item.description) > len(previous.description):
                items_by_id[requirement_id] = item
        return list(items_by_id.values())[:8]

    @staticmethod
    def _parse_requirement_table(text: str) -> dict[str, str]:
        """Extract one requirement from a markdown/plain-text RFP table chunk."""
        requirement_id = next(iter(re.findall(r"\b[A-Z]{2,5}-\d{2,5}\b", text)), "")
        lines: list[str] = []
        for raw_line in text.splitlines():
            for cell in raw_line.split("|"):
                value = re.sub(r"\s+", " ", cell).strip(" -:\t")
                if not value or re.fullmatch(r"[-: ]{3,}", value):
                    continue
                lines.append(value)

        def strip_category(value: str) -> str:
            return re.sub(
                r"^(?:기능요구사항|성능|보안|품질|인터페이스|데이터|테스트|제약사항|프로젝트관리|프로젝트 지원)\s*:\s*",
                "", value,
            ).strip()

        def value_after(marker: str) -> str:
            for position, value in enumerate(lines):
                if marker not in value:
                    continue
                for candidate in lines[position + 1:]:
                    candidate = strip_category(candidate)
                    if not candidate or candidate == requirement_id:
                        continue
                    if any(label in candidate for label in (
                        "요구사항 분류", "요구사항 고유번호", "요구사항 명칭",
                        "요구사항 상세설명", "산출정보", "관련요구사항",
                    )):
                        continue
                    return candidate
            return ""

        title = value_after("요구사항 명칭")
        detail_start = next((
            index for index, value in enumerate(lines)
            if re.sub(r"[\s/]", "", value) in {"세부", "내용", "세부내용"}
        ), None)
        detail_lines: list[str] = []
        if detail_start is not None:
            for value in lines[detail_start + 1:]:
                if "산출정보" in value or "관련요구사항" in value:
                    break
                value = strip_category(value)
                if value and re.sub(r"[\s/]", "", value) not in {"정의", "세부", "내용", "세부내용"}:
                    detail_lines.append(value)
        definition = value_after("정의")
        description = " ".join(detail_lines).strip() or definition
        # Advanced table rendering uses a spaced slash as a visual line break.
        # Preserve real terms such as "백업/복구", which have no surrounding spaces.
        description = re.sub(r"\s+/\s+", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
        return {"id": requirement_id, "title": title, "description": description}

    def _extract(self, document_id: str, query: str, schema: type[T], prompt: str, top_k: int = 10) -> T:
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

    async def answer(
        self, document_id: str, question: str,
        chat_history: list[dict[str, str]] | None = None, provider: str = "openai",
        conversation_id: str = "default",
    ) -> AnswerResponse:
        del chat_history  # Main session owns the authoritative conversation state.
        entry = self.sessions.get(conversation_id, document_id, provider)
        try:
            async with entry.lock:
                result = await entry.service.answer(question, document_id=document_id, top_k=10)
        except Exception:
            fallback_enabled = os.getenv("RAG_OPENAI_FALLBACK", "true").casefold() in {"1", "true", "yes", "on"}
            if provider == "openai" or not fallback_enabled:
                raise
            logger.warning("%s generation failed; retrying with OpenAI fallback", provider, exc_info=True)
            entry = self.sessions.get(conversation_id, document_id, "openai")
            async with entry.lock:
                result = await entry.service.answer(question, document_id=document_id, top_k=10)
        evidence = result.get("evidence") or []
        usage = result.get("_usage") or {}
        citations = [Citation(
            source_id=str(index),
            chunk_id=str(item["chunk_id"]),
            document_id=document_id,
            document_name=str(item.get("source") or document_id),
            page_start=int(item.get("page") or 1),
            page_end=int(item.get("page") or 1),
            requirement_ids=re.findall(r"\b[A-Z]{2,5}-\d{2,5}\b", str(item.get("quote") or "")),
            quote=str(item.get("quote") or "") or None,
            score=float(item.get("score")) if item.get("score") is not None else None,
        ) for index, item in enumerate(evidence, 1)]
        answer = str(result.get("answer") or "")
        answer_status = classify_answer_status(
            answer,
            evidence,
            needs_clarification=bool(result.get("needs_clarification")),
        )
        return AnswerResponse(
            question=question,
            answer=answer,
            is_answerable=is_answerable_status(answer_status),
            answer_status=answer_status,
            caveat=result.get("clarification_question"),
            confidence=float(result.get("confidence")) if result.get("confidence") is not None else None,
            conflicts=list(result.get("conflicts") or []),
            citations=citations,
            retrieved_chunk_ids=list(result.get("retrieved_chunk_ids") or []),
            retriever="main_advanced_dense",
            model=str(entry.service.session.model),
            search_latency_ms=float((result.get("latency") or {}).get("retrieval_seconds", 0)) * 1000,
            generation_latency_ms=float((result.get("latency") or {}).get("generation_seconds", 0)) * 1000,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            estimated_cost_usd=None,
        )

    def reset_conversation(self, conversation_id: str, document_id: str | None = None) -> int:
        return self.sessions.reset(conversation_id, document_id)

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
