# Handover v3 MVP backend

현재 RAG 엔진(`src/search`, `src/generation`)을 변경하지 않고 FastAPI adapter로
Handover v3 API 계약을 제공한다.

실행:

```bash
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

주요 API:

```text
GET  /api/health
GET  /api/documents
PUT  /api/documents/upload?filename=example.pdf&title=제목 (raw application/pdf body)
GET  /api/analysis/{document_id}/overview
GET  /api/analysis/{document_id}/risks
GET  /api/analysis/{document_id}/eligibility
GET  /api/analysis/{document_id}/deliverables
GET  /api/analysis/{document_id}/requirements
POST /api/analysis/{document_id}/ask
POST /api/analysis/{document_id}/ask/stream
DELETE /api/analysis/{document_id}/conversation/{conversation_id}
GET  /api/state/{document_id}
PATCH /api/state/{document_id}/eligibility/{item_id}
PATCH /api/state/{document_id}/deliverable/{item_id}
```

Q&A와 평가 문서의 Workspace 검색은 Main Advanced Chroma index를 사용한다.
브라우저는 `conversation_id`를 ask 요청에 전달하며 backend는 문서/provider별
`BidMateRAGSession`을 메모리에 유지한다.

개발 및 smoke test는 OpenAI `gpt-5-nano`, 데모 UI 기본 선택은
`gemini-3.5-flash-lite`다. Gemini 호출이 실패하면 기본적으로 OpenAI로 fallback한다.
`OPENAI_API_KEY`와 `GEMINI_API_KEY`는 `.env` 또는 `.env.local`에서 읽으며 저장소에
커밋하지 않는다. 모델 설정은 `configs/main_advanced_rag.yaml`에서 관리한다.
