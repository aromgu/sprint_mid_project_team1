from __future__ import annotations

import logging
import json
import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.models import AskRequest

router = APIRouter(prefix="/api/analysis/{document_id}", tags=["analysis"])
logger = logging.getLogger(__name__)


def client(request: Request):
    return request.app.state.rag_client


@router.get("/overview")
def overview(document_id: str, request: Request):
    try:
        return client(request).overview(document_id)
    except Exception as exc:
        logger.exception("overview analysis failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {type(exc).__name__}") from exc


@router.get("/risks")
def risks(document_id: str, request: Request):
    try:
        result = client(request).risks(document_id)
        saved = request.app.state.state_service.get(document_id).get("risks", {})
        for item in result.risks:
            if item.id in saved:
                item.user_status = saved[item.id].get("user_status", item.user_status)
        return result
    except Exception as exc:
        logger.exception("risks analysis failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {type(exc).__name__}") from exc


@router.get("/eligibility")
def eligibility(document_id: str, request: Request):
    try:
        return client(request).eligibility(document_id)
    except Exception as exc:
        logger.exception("eligibility analysis failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {type(exc).__name__}") from exc


@router.get("/deliverables")
def deliverables(document_id: str, request: Request):
    try:
        return client(request).deliverables(document_id)
    except Exception as exc:
        logger.exception("deliverables analysis failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {type(exc).__name__}") from exc


@router.get("/requirements")
def requirements(document_id: str, request: Request):
    try:
        return client(request).requirements(document_id)
    except Exception as exc:
        logger.exception("requirements analysis failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {type(exc).__name__}") from exc


@router.post("/ask")
async def ask(document_id: str, payload: AskRequest, request: Request):
    try:
        return await client(request).answer(
            document_id, payload.question, payload.chat_history,
            payload.provider, payload.conversation_id,
        )
    except Exception as exc:
        logger.exception("answer failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"answer failed: {type(exc).__name__}") from exc


@router.post("/ask/stream")
async def ask_stream(document_id: str, payload: AskRequest, request: Request):
    """SSE-compatible MVP stream. The final structured answer is chunked for the UI."""
    try:
        result = await client(request).answer(
            document_id, payload.question, payload.chat_history,
            payload.provider, payload.conversation_id,
        )
    except Exception as exc:
        logger.exception("stream answer failed for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail=f"answer failed: {type(exc).__name__}") from exc

    async def events():
        text = result.answer or ""
        step = max(1, len(text) // 24)
        for index in range(step, len(text) + step, step):
            yield f"data: {json.dumps({'type': 'delta', 'text': text[:index]}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'type': 'done', 'result': result.model_dump()}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.delete("/conversation/{conversation_id}")
def reset_conversation(document_id: str, conversation_id: str, request: Request):
    """Reset one browser conversation for the selected document."""
    return {
        "conversation_id": conversation_id,
        "document_id": document_id,
        "removed_sessions": client(request).reset_conversation(conversation_id, document_id),
    }
