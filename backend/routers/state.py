from __future__ import annotations

from fastapi import APIRouter, Request

from backend.models import DeliverableUpdate, EligibilityStatusUpdate

router = APIRouter(prefix="/api/state/{document_id}", tags=["state"])


@router.get("")
def get_state(document_id: str, request: Request):
    return request.app.state.state_service.get(document_id)


@router.patch("/eligibility/{item_id}")
def update_eligibility(document_id: str, item_id: str, payload: EligibilityStatusUpdate, request: Request):
    return request.app.state.state_service.update(document_id, "eligibility", item_id, payload.model_dump())


@router.patch("/deliverable/{item_id}")
def update_deliverable(document_id: str, item_id: str, payload: DeliverableUpdate, request: Request):
    values = {key: value for key, value in payload.model_dump().items() if value is not None}
    return request.app.state.state_service.update(document_id, "deliverables", item_id, values)
