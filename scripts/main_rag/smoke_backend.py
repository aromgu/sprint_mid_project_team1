"""Exercise one P2 runtime provider without starting a web server."""

from __future__ import annotations

import argparse
import asyncio
import json

from backend.services.rag_client import RAGClient
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.runtime import MainAdvancedSessionManager


async def run(document_id: str, conversation_id: str, provider: str = "openai", *, followup: bool = True) -> dict:
    retriever = AdvancedRetriever()
    manager = MainAdvancedSessionManager(retriever=retriever)
    client = object.__new__(RAGClient)
    client.advanced_retriever = retriever
    client.sessions = manager
    client._analysis_cache = {}

    first = await client.answer(
        document_id, "기술평가 방식과 평가위원 구성은?",
        provider=provider, conversation_id=conversation_id,
    )
    second = None
    if followup:
        second = await client.answer(
            document_id, "그 점수 산정 방법도 알려줘.",
            provider=provider, conversation_id=conversation_id,
        )
    removed = client.reset_conversation(conversation_id, document_id)
    if first.retriever != "main_advanced_dense" or (second and second.retriever != "main_advanced_dense"):
        raise RuntimeError("backend가 Main Advanced retriever를 사용하지 않았습니다")
    if not first.answer or (second and not second.answer) or not first.citations:
        raise RuntimeError("답변 또는 citation이 비어 있습니다")
    if removed != 1:
        raise RuntimeError(f"conversation reset 대상이 1개가 아닙니다: {removed}")
    return {
        "first": first.model_dump(),
        "followup": second.model_dump() if second else None,
        "removed_sessions": removed,
        "remaining_sessions": manager.session_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", default="eval_01")
    parser.add_argument("--conversation-id", default="p2-openai-smoke")
    parser.add_argument("--provider", choices=("openai", "gemini", "gemini-lite"), default="openai")
    parser.add_argument("--single", action="store_true", help="후속 질문을 생략하고 API 호출을 1건으로 제한합니다.")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(
        args.document_id,
        args.conversation_id,
        args.provider,
        followup=not args.single,
    )), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
