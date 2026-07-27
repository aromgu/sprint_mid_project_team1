from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    document_name: str
    page_number: int
    quote: str
    score: float = Field(ge=0.0)
    chunk_id: str | None = None
    requirement_ids: list[str] = []


class ActionItem(BaseModel):
    id: str
    title: str
    description: str = ""
    priority: Literal["critical", "warning", "info", "high", "medium", "low"] = "info"
    due_date: str | None = None
    source_ids: list[str] = []
    evidence: Evidence | None = None


class RiskCounts(BaseModel):
    critical: int = 0
    warning: int = 0
    info: int = 0


class DeliverableProgress(BaseModel):
    completed: int = 0
    total: int = 0


class OverviewResponse(BaseModel):
    document_id: str = ""
    submission_deadline: str | None = None
    inquiry_deadline: str | None = None
    eligibility_summary: Literal["eligible", "ineligible", "review_required"] = "review_required"
    risk_counts: RiskCounts = Field(default_factory=RiskCounts)
    deliverable_progress: DeliverableProgress = Field(default_factory=DeliverableProgress)
    action_items: list[ActionItem] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RiskItem(BaseModel):
    id: str
    type: Literal["disqualification", "deduction", "review"]
    severity: Literal["critical", "warning", "info"]
    title: str
    description: str
    user_status: Literal["unchecked", "at_risk", "safe", "review_required"] = "unchecked"
    source_ids: list[str] = []
    evidence: Evidence | None = None


class RisksResponse(BaseModel):
    document_id: str = ""
    risks: list[RiskItem] = []


class EligibilityItem(BaseModel):
    id: str
    title: str
    description: str
    user_status: Literal["met", "not_met", "review_required", "unchecked"] = "unchecked"
    source_ids: list[str] = []
    evidence: Evidence | None = None


class EligibilityResponse(BaseModel):
    document_id: str = ""
    items: list[EligibilityItem] = []


class DeliverableItem(BaseModel):
    id: str
    name: str
    kind: Literal["bid_submission", "project_deliverable"] = "bid_submission"
    description: str = ""
    format: str = "확인 필요"
    quantity: int = 1
    requires_seal: bool = False
    requires_original: bool = False
    assignee: str | None = None
    status: Literal["pending", "in_progress", "completed"] = "pending"
    deadline: str | None = None
    source_ids: list[str] = []
    evidence: Evidence | None = None


class DeliverablesResponse(BaseModel):
    document_id: str = ""
    items: list[DeliverableItem] = []


class RequirementItem(BaseModel):
    id: str
    category: Literal["functional", "performance", "security", "operation", "personnel", "output", "contract"] = "functional"
    title: str
    description: str
    priority: Literal["high", "medium", "low"] = "medium"
    review_status: Literal["pending", "reviewed", "flagged"] = "pending"
    source_ids: list[str] = []
    evidence: Evidence | None = None


class RequirementsResponse(BaseModel):
    document_id: str = ""
    items: list[RequirementItem] = []


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_history: list[dict[str, str]] = []
    provider: Literal["openai", "gemini", "gemini-lite"] = "gemini-lite"


class EligibilityStatusUpdate(BaseModel):
    user_status: Literal["met", "not_met", "review_required"]


class RiskStatusUpdate(BaseModel):
    user_status: Literal["unchecked", "at_risk", "safe", "review_required"]


class DeliverableUpdate(BaseModel):
    assignee: str | None = None
    status: Literal["pending", "in_progress", "completed"] | None = None


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    organization: str
    difficulty: str | None = None
    document_date: str | None = None
    status: str = "ready"
