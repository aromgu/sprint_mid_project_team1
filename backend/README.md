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
GET  /api/state/{document_id}
PATCH /api/state/{document_id}/eligibility/{item_id}
PATCH /api/state/{document_id}/deliverable/{item_id}
```

분석 API는 OpenAI 호출이 필요하다. `OPENAI_API_KEY`는 `.env` 또는 `.env.local`에서
읽으며 저장소에 커밋하지 않는다. API 기본 모델은 `configs/generation.yaml`의
`gpt-5-nano`다.
