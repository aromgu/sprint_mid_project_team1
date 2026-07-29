# Main Advanced RAG P2 상태

## 결론

P2 backend/UI 연결, 업로드 PDF 증분 Advanced indexing, 내부 Evaluation 교체,
실제 OpenAI 브라우저 통합 테스트를 완료했다. 평가 문서와 새 업로드 문서의 Q&A 및
Workspace 카드는 Main Advanced index를 사용한다. 2026-07-29에 Gemini
Flash-Lite 실제 1건 smoke도 통과해 데모 모델 access/quota까지 확인했다.

## 완료한 작업

- Main 결과를 기존 `AnswerResponse`로 변환하는 Q&A adapter
- answer, caveat, confidence, conflicts 변환
- evidence를 citation의 document/page/chunk/quote/score로 변환
- `RAGClient.answer()`를 `MainAdvancedRAGService`로 교체
- Q&A retrieval을 `AdvancedRetriever`로 교체
- Workspace Overview/Risks/Eligibility/Deliverables/Requirements 검색을 Advanced로 교체
- 브라우저별 `conversation_id` 생성 및 전달
- `(conversation_id, document_id, provider)`별 독립 `BidMateRAGSession`
- 동일 session 재사용, 동시 요청 직렬화, 1시간 TTL, 최대 100개 session 정리
- conversation reset API와 UI의 `대화 초기화` 버튼
- P1 평가 모델과 P2 runtime 모델 설정 분리
- Gemini/Flash-Lite 실패 시 OpenAI `gpt-5-nano` 자동 fallback
- 업로드 PDF의 Advanced 전처리·semantic chunking·OpenAI embedding·Chroma 증분 upsert
- UI 업로드 카드의 파일 선택→`api.upload()`→문서 목록 재조회→업로드 문서 선택 연결 확인
- 업로드 후 live collection/chunk/document count를 Evaluation에 반영
- 내부 Evaluation을 P1 Main Advanced retrieval, 104개 answer, RAGAS 결과로 교체
- 실제 브라우저에서 FastAPI→Advanced Chroma→OpenAI→SSE→reset 검증

## Runtime 모델

| 용도 | Provider/model |
|---|---|
| P1 평가 재현 | OpenAI `gpt-5-nano` |
| P2 데모 기본 선택 | Gemini `gemini-3.5-flash-lite` |
| 개발/smoke test | OpenAI `gpt-5-nano` |
| Gemini 실패 fallback | OpenAI `gpt-5-nano` |
| Workspace structured card enrichment 기본 | OpenAI `gpt-5-nano` |

설정 위치는 `configs/main_advanced_rag.yaml`의 `generation`과 `runtime`이다.

## OpenAI 실제 smoke test

실행 명령:

```bash
uv run python -m scripts.main_rag.smoke_backend
```

검증한 흐름:

```text
eval_01 첫 질문
  → Main Advanced retrieval
  → gpt-5-nano 답변/citation
  → 같은 conversation의 후속 질문/query rewrite
  → conversation reset
```

결과:

- 첫 질문과 후속 질문 모두 성공
- retriever: `main_advanced_dense`
- model: `gpt-5-nano`
- citation chunk/page 정상
- 첫 질문 검색 3.09초, 생성 49.16초
- 후속 질문 검색 0.76초, 생성 33.95초
- reset 제거 session 1개, reset 후 session 0개

OpenAI smoke test latency는 데모 UX에 느리므로 Flash-Lite를 데모 기본값으로 유지한다.

## 검증 결과

- Python: `52 passed, 9 warnings`
- Frontend production build: 성공
- Playwright UI regression: `9 passed`, 유료 live test `1 skipped`
- OpenAI 실제 backend 첫 질문/후속 질문/reset smoke: 성공
- 실제 업로드 PDF Advanced embedding/index/search smoke: 성공 (`1 chunk`, 검색 page `1`)
- 실제 브라우저/FastAPI/OpenAI 통합: `1 passed` (`28.1초`, API 처리 `25.2초`)
- 2026-07-29 재검증: Python `52 passed`, Playwright `10 passed` + live OpenAI `1 passed`, production build 성공
- Gemini Flash-Lite 실제 1건: 성공, model `gemini-3.5-flash-lite`, 검색 `1.45초`, 생성 `2.68초`, citation 정상
- Top-10 Golden 답변: `104/104`, 오류 0, 4-worker
- Top-10 공식 RAGAS: `104/104`, 8-worker, Faithfulness `0.8479`, Answer Relevancy `0.3742`
- 3단계 저장 판정: answered `88`, partially_answered `14`, unanswerable `2`; 잘못 거절 `1`
- 현재 UI 판정: 누락 문구 `확인할 수 없습니다`를 보정해 answered `87`, partially_answered `15`, unanswerable `2`

## 관련 코드

- `src/main_rag/runtime.py`: runtime provider와 in-memory conversation manager
- `src/main_rag/service.py`: Advanced retrieval/generation/citation 검증
- `backend/services/rag_client.py`: Q&A adapter 및 Workspace Advanced 연결
- `backend/routers/analysis.py`: ask/stream/reset API
- `backend/models.py`: conversation ID request model
- `src/generation/models.py`: citation quote/score 및 confidence/conflicts 응답
- `frontend/src/api.js`: conversation ID 전달과 reset 호출
- `frontend/src/main.jsx`: browser conversation ID와 reset/근거 표시 UI
- `scripts/main_rag/smoke_backend.py`: OpenAI P2 smoke test
- `src/main_rag/upload_indexer.py`: 업로드 PDF Advanced 증분 indexing
- `scripts/main_rag/smoke_upload_index.py`: 격리된 실제 embedding/index/search smoke
- `frontend/tests/ui/live-main-advanced.spec.js`: 실제 API 브라우저 통합 테스트
- `backend/routers/evaluation.py`, `frontend/src/evaluation_ui.jsx`: Main Advanced 평가 화면

## 남은 일

### 데모 당일 운영 확인

1. 외부 API quota와 네트워크 상태 확인
2. 실제 발표 환경의 화면 크기·프로젝터 가독성 확인

### 후속 개선

1. 현재 단일 프로세스 memory session을 multi-worker 환경에서는 Redis/DB로 교체
2. OpenAI 생성 latency와 과도한 reasoning/output token 최적화
3. 후속 smoke 답변의 근거 밖 문장(예: 불명확한 동점 처리 표현) 억제 강화
4. session 상태/모델 fallback 여부를 health 또는 운영 로그에 더 명확히 표시
5. 업로드 indexing 실패 시 이미 저장된 PDF 정리/재시도 UX 보강
6. 다중 사실 질문의 query decomposition과 생성 prompt 실험
7. prompt/검색 정책을 바꾸는 경우에만 104개 RAGAS·LLM Judge 재평가

## Answer Relevancy 표시 정책

- 전체 평균 `0.3742`는 부분 답변과 답변 불가를 포함한다.
- Evaluation UI는 전체 평균과 `answered` / `partially_answered` / `unanswerable`
  평균을 분리해 표시한다.
- 이 분리는 점수를 변경하지 않고, 정상적인 근거 부족 거절이 전체
  Answer Relevancy를 낮추는 구조를 투명하게 보여준다.

## 범위상 제한

- memory session은 단일 backend process에서만 공유된다.
- 서버 재시작 시 conversation session은 초기화된다.
- 업로드 처리는 요청 중 동기적으로 완료되므로 큰 PDF에서는 업로드 응답이 느릴 수 있다.
- Gemini quota가 없으면 자동으로 OpenAI fallback되어 응답은 가능하지만 느릴 수 있다.
