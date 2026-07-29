"""In-memory conversation sessions for the Main Advanced demo runtime."""

from __future__ import annotations

import os
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from src.main_rag.generation.gemini_session import GeminiBidMateRAGSession
from src.main_rag.generation.generate_answer import BidMateRAGSession
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.service import MainAdvancedRAGService
from src.main_rag.settings import MainRAGSettings, load_settings


PROVIDERS = {
    "openai": ("OPENAI_API_KEY", BidMateRAGSession, "gpt-5-nano"),
    "gemini": ("GEMINI_API_KEY", GeminiBidMateRAGSession, "gemini-3.5-flash"),
    "gemini-lite": ("GEMINI_API_KEY", GeminiBidMateRAGSession, "gemini-3.5-flash-lite"),
}


@dataclass
class RuntimeSession:
    service: MainAdvancedRAGService
    touched_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MainAdvancedSessionManager:
    """Map browser conversation IDs to isolated, expiring BidMate sessions."""

    def __init__(
        self,
        *,
        settings: MainRAGSettings | None = None,
        retriever: AdvancedRetriever | None = None,
        ttl_seconds: int = 3600,
        max_sessions: int = 100,
    ) -> None:
        self.settings = settings or load_settings()
        self.retriever = retriever or AdvancedRetriever(self.settings)
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[tuple[str, str, str], RuntimeSession] = {}
        self._lock = threading.RLock()

    def _new_service(self, provider: str) -> MainAdvancedRAGService:
        try:
            key_name, session_class, default_model = PROVIDERS[provider]
        except KeyError as exc:
            raise ValueError(f"지원하지 않는 provider입니다: {provider}") from exc
        load_dotenv()
        api_key = os.getenv(key_name)
        if not api_key:
            raise RuntimeError(f"{key_name}가 설정되지 않았습니다")
        model = str(self.settings.get("runtime", provider.replace("-", "_"), default_model))
        session = session_class(
            api_key=api_key,
            model=model,
            max_context_chars=int(self.settings.get("generation", "max_context_chars", 7000)),
            max_docs=int(self.settings.get("generation", "max_docs", 6)),
        )
        return MainAdvancedRAGService(settings=self.settings, retriever=self.retriever, session=session)

    def _prune(self, now: float) -> None:
        expired = [key for key, value in self._sessions.items() if now - value.touched_at > self.ttl_seconds]
        for key in expired:
            self._sessions.pop(key, None)
        if len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].touched_at)
            self._sessions.pop(oldest, None)

    def get(self, conversation_id: str, document_id: str, provider: str) -> RuntimeSession:
        if not conversation_id.strip():
            raise ValueError("conversation_id는 비어 있을 수 없습니다")
        key = (conversation_id, document_id, provider)
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            entry = self._sessions.get(key)
            if entry is None:
                entry = RuntimeSession(self._new_service(provider))
                self._sessions[key] = entry
            entry.touched_at = now
            return entry

    def reset(self, conversation_id: str, document_id: str | None = None) -> int:
        with self._lock:
            keys = [
                key for key in self._sessions
                if key[0] == conversation_id and (document_id is None or key[1] == document_id)
            ]
            for key in keys:
                entry = self._sessions.pop(key)
                entry.service.reset()
            return len(keys)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
