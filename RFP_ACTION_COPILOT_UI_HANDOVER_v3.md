# RFP Action Copilot — UI/UX 기획 핸드오버

**작성일**: 2026-07-21
**버전**: v3 (최종 — 실제 코드 구조 반영)
**대상**: UI 개발팀 / Codex(RAG 구현) 팀
**전제 조건**: RAG 파이프라인 코드 존재 (`src/`, `configs/`, `pipeline.py`), React + FastAPI 구현 경험 보유

> **v3 주요 변경**
> - 7절 디렉터리: 실제 프로젝트 구조 반영 (`src/`, `configs/` 등)
> - 13절 import 경로: 실제 파일 기준으로 수정
> - 15절 표 단위 청킹: `structured_chunker.py`에 이미 구현됨으로 확정
> - 16절 Codex 지시: 새로 만들지 않고 기존 코드 기반 어댑터 추가로 변경

---

## 1. 제품 정의

### 이 제품은 무엇인가

공공 RFP를 처음 검토하는 담당자가 **10초 안에 참여 여부를 판단**할 수 있도록 돕는 **RFP 검토 Copilot**이다.

일반적인 PDF 챗봇이 아니다. 사용자가 RFP를 열었을 때 가장 먼저 확인해야 하는 6가지 질문에 구조화된 답을 자동으로 제공한다.

| 번호 | 사용자가 알고 싶은 것 |
|------|-------------------|
| 1 | 제출 마감은 언제인가? |
| 2 | 질의 마감은 언제인가? |
| 3 | 참가 자격을 충족하는가? |
| 4 | 실격 또는 감점 위험은 무엇인가? |
| 5 | 필수 제출물은 무엇인가? |
| 6 | 지금 당장 해야 할 일은 무엇인가? |

채팅(AI 질문)은 보조 기능이다. 핵심은 구조화된 분석 결과를 즉시 보여주는 것이다.

### 기존 프로젝트와의 관계

기존 핸드오버(HANDOVER.md)의 목표는 **"제안서 자동 생성"** 이었다.
이 UI의 방향은 **"참여 결정 지원"** 이다.

| 항목 | 기존 방향 | 이 UI 방향 |
|------|----------|-----------|
| 사용자 | 제안서 작성팀 | RFP 검토 담당자 |
| 핵심 기능 | 제안서 문단 생성 | 자격/위험/마감 자동 추출 |
| RAG 역할 | 문서 기반 생성 | 정보 추출 및 구조화 |
| 비즈니스 가치 | 수주 지원 | 참여 결정 지원 |

---

## 2. 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| Frontend | React (TypeScript) | 3열 레이아웃, 근거 패널 고정 등 Streamlit으로 불가능한 UX |
| Backend | FastAPI (Python) | RAG 파이프라인과 동일 언어, 스트리밍 응답 지원 |
| 상태관리 | Zustand | 탭 간 선택 상태, 근거 패널 연동 |
| 사용자 상태 저장 | JSON 파일 (`user_state.json`) | 데모용. 운영 전환 시 PostgreSQL로 교체 |
| 타입 동기화 | openapi-ts | FastAPI 스키마 → TypeScript 타입 자동 생성 |
| Mock 데이터 | `src/mocks/*.json` | Phase 0 전용. `VITE_USE_MOCK=true` 환경변수로 스위칭 |
| RAG 파이프라인 | LangChain + OpenAI | 기존 `src/` 코드 재사용 (아래 7절 참고) |

### 지원 환경

- **해상도**: Desktop 1440px 기준. **반응형 미지원 (데모 범위 외)**
- 태블릿/모바일 대응은 운영 전환 시 별도 논의

---

## 3. 화면 구조

### Page 1: RFP List

사용자가 RFP를 업로드하고 목록을 관리하는 진입점이다.

각 RFP 카드에 표시할 항목:
- 사업명 / 발주기관
- 제출 D-Day (숫자 강조)
- 참여 상태 배지
- 위험 건수 (실격 N건 / 감점 N건)
- 제출물 완료율 (진행바)
- 분석 상태 (분석 중 / 완료 / 오류)

기능: 검색, 마감순 정렬, 상태 필터, 카드 클릭 → Workspace 이동

### Page 2: RFP Workspace

**3열 레이아웃** (1440px 기준):

```
┌─────────────┬──────────────────────────┬─────────────────┐
│ 문서 목록   │      중앙 탭 영역         │   근거 패널     │
│  (좌측)     │                          │   (우측 고정)   │
│             │  [Overview] [Go/No-Go]   │                 │
│ • RFP 1 ●  │  [실행준비] [요구사항]   │  📄 문서명      │
│ • RFP 2    │  [AI 질문]               │  📍 p.12        │
│             │                          │  💬 원문 인용   │
│             │      탭 콘텐츠           │  🎯 점수 0.94   │
└─────────────┴──────────────────────────┴─────────────────┘
```

**상단 상태바** (항상 고정):
```
제출 D-12  |  질의 D-3  |  참여 검토 중  |  실격 2건  |  감점 1건  |  제출물 3/8
```

### Page 3: Evaluation ⚠️ 내부 운영팀 전용

RAG 성능 지표 대시보드. 라우팅: `/internal/evaluation`
일반 사용자 네비게이션에 노출하지 않는다.

표시 항목: Retrieval Hit Rate, Citation Accuracy, Field Extraction Accuracy, 평균 응답 시간, Chunk Size, Top-K, 사용 모델

---

## 4. 탭별 상세 정의

### Tab 1: Overview

카드 6개 + 지금 할 일 목록 + 주요 일정 타임라인으로 구성한다.
카드 클릭 시 해당 탭으로 이동하거나 근거 패널을 업데이트한다.

### Tab 2: Go / No-Go

**참가 자격** 섹션에서 각 항목마다 사용자가 상태를 선택할 수 있어야 한다.

```
[ ✅ 충족 ]  [ ❌ 미충족 ]  [ ❓ 확인 필요 ]
```

선택 상태는 `user_state.json`에 저장한다 (새로고침 후에도 유지).

**위험 항목 심각도 표시**:

| 수준 | 색상 | 사용 조건 |
|------|------|----------|
| critical | 빨강 | 실격 조건 |
| warning | 주황 | 감점 조건 |
| info | 파랑 | 확인 필요 |

추가 섹션: 실격 조건, 감점 조건, 문서 간 불일치

### Tab 3: 실행 준비

**제출물 테이블** 컬럼:

| 상태 | 제출물명 | 형식 | 수량 | 날인 | 담당자 | 출처 |
|------|--------|------|------|------|------|------|

담당자 셀은 인라인 편집 가능. 변경값은 `user_state.json`에 저장.

**필터**: 미완료만 / 원본 필요 / 날인 필요 / 담당자 미지정 / 마감 임박

**일정**: 월간 캘린더 대신 D-Day 중심 목록으로 구현한다.

### Tab 4: 요구사항

**테이블** 컬럼: ID, 유형, 요구사항, 중요도, 검토 상태, 출처

**필터**: 전체 / 기능 / 성능 / 보안 / 운영 / 인력 / 산출물 / 계약

TanStack Table 사용 권장 (필터 + 정렬 + 페이지네이션 통합 관리).

### Tab 5: AI 질문

**추천 질문은 고정값이 아니다.** `activeTab` 값 기준으로 동적으로 변경된다.

예시 — 참여 결정 맥락 (Overview / Go-No-Go 탭 활성 시):
- 이 사업의 실격 조건을 정리해 줘
- 질의해야 할 모호한 항목을 찾아줘
- 일정상 가장 위험한 지점은?

예시 — 실행 준비 맥락 (실행 준비 / 요구사항 탭 활성 시):
- 필수 인력 조건을 표로 만들어 줘
- 제출물 중 원본이 필요한 항목은?
- 보안 요구사항만 추려줘

답변 구조: 답변 본문 + 근거 문서명 + 페이지 + 원문 인용 + 근거 부족 여부

FastAPI `StreamingResponse` + React `EventSource`로 타이핑 효과 구현.
스트리밍 에러 핸들링은 **9절 엣지 케이스** 참고.

---

## 5. 근거 패널 동작 규칙

모든 탭에서 오른쪽에 고정된다 (`position: sticky`).

**트리거**: 자격 항목, 위험 항목, 제출물 항목, 요구사항 항목 클릭

**표시 내용**:
- 문서명
- 페이지 번호
- 원문 인용 (하이라이트)
- 검색 점수 (0.00 ~ 1.00)
- 이전/다음 근거 네비게이션
- PDF 페이지 열기 버튼

**기본 상태**: "항목을 클릭하면 근거가 표시됩니다"

**근거 없음 상태**: "이 항목의 근거 문서를 찾지 못했습니다"

---

## 6. API 스키마 정의 ← RAG 팀과 UI 팀의 계약서

스키마가 확정되면 양쪽이 독립적으로 작업할 수 있다.

### 공통 Evidence 모델

```typescript
interface Evidence {
  document_name: string;   // ChunkRecord.document_title
  page_number: number;     // ChunkRecord.page_start
  quote: string;           // ChunkRecord.text 앞 100자
  score: number;           // reranker score (0.0 ~ 1.0)
}
```

### GET /api/analysis/{document_id}/overview

```typescript
interface OverviewResponse {
  submission_deadline: string | null;      // ISO 8601 또는 null
  inquiry_deadline: string | null;
  eligibility_summary: "eligible" | "ineligible" | "review_required";
  risk_counts: { critical: number; warning: number; info: number; };
  deliverable_progress: { completed: number; total: number; };
  action_items: ActionItem[];
  confidence: number;
}
```

### GET /api/analysis/{document_id}/risks

```typescript
interface RiskItem {
  id: string;
  type: "disqualification" | "deduction" | "review";
  severity: "critical" | "warning" | "info";
  title: string;
  description: string;
  evidence: Evidence;
}
interface RisksResponse { risks: RiskItem[]; }
```

### GET /api/analysis/{document_id}/eligibility

```typescript
interface EligibilityItem {
  id: string;
  title: string;
  description: string;
  user_status: "met" | "not_met" | "review_required" | "unchecked";
  evidence: Evidence;
}
interface EligibilityResponse { items: EligibilityItem[]; }
```

### GET /api/analysis/{document_id}/deliverables

```typescript
interface DeliverableItem {
  id: string;
  name: string;
  format: string;
  quantity: number;
  requires_seal: boolean;
  requires_original: boolean;
  assignee: string | null;
  status: "pending" | "in_progress" | "completed";
  deadline: string | null;
  evidence: Evidence;
}
interface DeliverablesResponse { items: DeliverableItem[]; }
```

### GET /api/analysis/{document_id}/requirements

```typescript
interface RequirementItem {
  id: string;
  category: "functional" | "performance" | "security" | "operation" | "personnel" | "output" | "contract";
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  review_status: "pending" | "reviewed" | "flagged";
  evidence: Evidence;
}
interface RequirementsResponse { items: RequirementItem[]; }
```

### POST /api/analysis/{document_id}/ask (SSE 스트리밍)

```typescript
interface AskRequest {
  question: string;
  chat_history: { role: "user" | "assistant"; content: string }[];
}
// SSE 이벤트 순서:
// 1. data: {"answer_chunk": "..."} (반복)
// 2. event: done / data: {"evidences": [...], "low_confidence": false}
// 3. event: error / data: {"message": "..."} (오류 시)
```

### PATCH /api/state/{document_id}/eligibility/{item_id}

```typescript
interface EligibilityStatusUpdate {
  user_status: "met" | "not_met" | "review_required";
}
```

### PATCH /api/state/{document_id}/deliverable/{item_id}

```typescript
interface DeliverableUpdate {
  assignee?: string;
  status?: "pending" | "in_progress" | "completed";
}
```

---

## 7. 프로젝트 디렉터리 구조 (실제 기준)

```
rfp-action-copilot/
│
├── [기존 RAG 코드 — Codex 담당]
│   ├── pipeline.py                   # 메인 파이프라인 + FastAPI 어댑터 함수 추가 예정
│   ├── configs/
│   │   ├── config.py                 # RAGConfig
│   │   └── prompt.py                 # 프롬프트 템플릿 (ANSWER_PROMPT, REWRITE_PROMPT)
│   ├── src/
│   │   ├── chunking/
│   │   │   ├── split_text.py
│   │   │   └── structured_chunker.py # ✅ 표 단위 청킹 이미 구현됨
│   │   ├── embeddings/
│   │   │   └── build_embeddings.py
│   │   ├── generation/
│   │   │   └── generate_answer.py    # AdvancedRAGChain — 스트리밍 추가 예정
│   │   ├── ingestion/
│   │   │   ├── models.py             # ChunkRecord, PageRecord
│   │   │   └── pdf_parser.py
│   │   ├── retrieval/
│   │   │   └── retriever.py          # ProductionRetriever
│   │   └── evaluation/
│   │       └── eval_rag.py
│   └── data/
│       ├── processed/
│       │   └── chunks.jsonl          # 벡터 인덱스
│       └── manifests/
│           └── documents.json
│
├── backend/                          # UI 개발팀 담당 (신규)
│   ├── main.py
│   ├── routers/
│   │   ├── documents.py              # 문서 업로드/목록
│   │   ├── analysis.py               # RAG 분석 결과
│   │   ├── state.py                  # 사용자 상태 저장/조회
│   │   └── chat.py                   # AI 질문 스트리밍
│   ├── services/
│   │   ├── rag_client.py             # pipeline.py 어댑터 함수 호출
│   │   └── state_service.py          # JSON 파일 CRUD
│   ├── models/                       # Pydantic 모델 (6절 스키마 기반)
│   └── data/
│       └── user_state.json
│
└── frontend/                         # UI 개발팀 담당 (신규)
    ├── src/
    │   ├── pages/
    │   │   ├── RfpList.tsx
    │   │   ├── Workspace.tsx
    │   │   └── Evaluation.tsx        # /internal/evaluation 전용
    │   ├── components/
    │   │   ├── EvidencePanel.tsx     ← 핵심 컴포넌트
    │   │   ├── StatusBar.tsx
    │   │   ├── StatusBadge.tsx
    │   │   └── RiskCard.tsx
    │   ├── tabs/
    │   │   ├── Overview.tsx
    │   │   ├── GoNoGo.tsx
    │   │   ├── ActionPrep.tsx
    │   │   ├── Requirements.tsx
    │   │   └── AiChat.tsx
    │   ├── mocks/                    # Phase 0 전용 Mock 데이터
    │   │   ├── overview.json
    │   │   ├── risks.json
    │   │   ├── eligibility.json
    │   │   ├── deliverables.json
    │   │   └── requirements.json
    │   ├── store/
    │   │   └── workspaceStore.ts
    │   └── api/
    │       └── client.ts             # VITE_USE_MOCK 분기 포함
    └── package.json
```

> **담당 분리**:
> - 기존 RAG 코드 (`pipeline.py`, `src/`, `configs/`) → Codex 담당
> - `backend/`, `frontend/` → UI 개발팀 담당
> - 접점: `backend/services/rag_client.py` → `pipeline.py` 어댑터 함수 호출

---

## 8. 전역 상태 관리 (Zustand)

```typescript
interface WorkspaceStore {
  selectedDocumentId: string | null;
  activeTab: "overview" | "gonogo" | "action" | "requirements" | "chat";
  selectedEvidence: Evidence | null;
  evidenceHistory: Evidence[];
  selectedEligibilityId: string | null;
  selectedRiskId: string | null;
  selectedDeliverableId: string | null;
  selectedRequirementId: string | null;
  chatHistory: { role: string; content: string }[];
}
```

---

## 9. 엣지 케이스 처리 규칙

| 상황 | UI 처리 |
|------|--------|
| 마감일 추출 실패 | "확인 필요" 배지 + 근거 없음 표시 |
| Evidence score < 0.7 | 항목 옆에 ⚠️ 표시 + "낮은 신뢰도" 툴팁 |
| 동일 항목 근거 여러 개 | 가장 높은 score 기준 표시, 하단에 "근거 N개" 링크 |
| RAG API 타임아웃 | 스켈레톤 UI + "분석 중 오류가 발생했습니다. 재시도" 버튼 |
| 근거 없는 항목 클릭 | 패널에 "이 항목의 근거 문서를 찾지 못했습니다" 표시 |
| 스트림 연결 실패 | `EventSource.onerror` → 에러 메시지 + 재시도 버튼, `es.close()` |
| 스트림 도중 중단 | 마지막 청크 수신 후 10초 무응답 → 타임아웃 처리 |
| 스트림 도중 서버 오류 | 서버가 `event: error` 전송 → UI에서 수신 후 에러 메시지 표시 |

### 스트리밍 에러 핸들링 구현 패턴

**FastAPI (서버)**

```python
async def stream():
    try:
        async for chunk in rag_client.stream_answer(document_id, question, history):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"
    finally:
        yield "event: done\ndata: [DONE]\n\n"
```

**React (클라이언트)**

```typescript
const es = new EventSource(`/api/analysis/${id}/ask`);

es.onmessage = (e) => appendAnswer(e.data);
es.addEventListener("error", () => {
  setError("답변 생성 중 오류가 발생했습니다.");
  es.close();
});
es.addEventListener("done", () => es.close());
es.onerror = () => { setError("서버 연결이 끊겼습니다."); es.close(); };

const timeout = setTimeout(() => {
  es.close();
  setError("응답 시간이 초과되었습니다.");
}, 10_000);
```

---

## 10. 구현 우선순위

### Phase 0 — UI 목업 검증 (RAG·API 없이)

**순수 목업이다.** RAG 없음, FastAPI 없음, DB 없음.
`src/mocks/` 의 고정 JSON을 화면에 뿌리는 것만 목표.

**가능한 것 / 불가능한 것**:

| 항목 | Phase 0 |
|------|---------|
| RfpList → Workspace 페이지 전환 | ✅ React Router, 서버 불필요 |
| Workspace 탭 전환 (Overview ~ AI 질문) | ✅ |
| 근거 패널 클릭 인터랙션 | ✅ Mock Evidence 데이터로 |
| 상단 상태바 D-Day 표시 | ✅ |
| 카드마다 다른 RFP 데이터 | ❌ 고정 Mock 1개 (Phase 1) |
| 자격 체크 상태 저장 (새로고침 유지) | ❌ Phase 1 |
| AI 질문 스트리밍 | ❌ Phase 2 |

**체크리스트**:
- [ ] `src/mocks/` 에 JSON 5개 배치 (별도 제공)
- [ ] `client.ts` 에 `VITE_USE_MOCK` 분기 추가
- [ ] 3열 레이아웃 + 탭 네비게이션 확인
- [ ] 근거 패널 클릭 인터랙션 확인
- [ ] 상태바 D-Day 카운트 확인

### Phase 1 — RAG 연동 (핵심 가치 증명)
- [ ] `pipeline.py` 어댑터 함수 완성 (Codex)
- [ ] `generate_answer.py` 스트리밍 + Evidence 추가 (Codex)
- [ ] `backend/` FastAPI 서버 구성
- [ ] `VITE_USE_MOCK=false` 전환 + 탭별 검증
- [ ] 사용자 체크 상태 저장 (`user_state.json`)

### Phase 2 — 기능 확장
- [ ] 실행 준비 탭 (제출물 테이블 + 담당자 편집)
- [ ] 요구사항 탭 (필터 + TanStack Table)
- [ ] AI 질문 탭 (스트리밍 + 에러 핸들링)

### Phase 3 — 완성도
- [ ] Evaluation 페이지 (`/internal/evaluation`)
- [ ] PDF 페이지 열기 연동
- [ ] 성능 최적화 (캐싱, 스켈레톤 UI)

---

## 11. 디자인 원칙

공공기관 업무용 도구답게 **차분하고 전문적인 분위기**를 유지한다.

- 과도한 그라데이션, 장식용 애니메이션 사용 금지
- 색상은 `theme.ts` 또는 `constants/colors.ts` 한 곳에서만 관리
- 텍스트 가독성 최우선 (충분한 행간, 명확한 대비)
- 실격 조건: 빨강 + 굵은 테두리
- 감점 조건: 주황
- 확인 필요: 파랑 또는 회색

기본 해상도: **Desktop 1440px 전용. 반응형 미지원 (데모 범위 외).**

---

## 12. Codex 구현 지시

### 전달 순서

**1단계 — 컨텍스트 설정 (첫 메시지)**

```
아래는 RFP Action Copilot 프로젝트 핸드오버 문서야.
전체를 읽고 내용을 이해했으면 "확인했습니다"라고만 답해줘.
지금은 아무것도 만들지 마.

[핸드오버 MD 전체 붙여넣기]
```

**2단계 — 기존 코드 기반 어댑터 추가 지시 (두 번째 메시지)**

```
기존 코드를 최대한 유지하면서 아래 2가지만 수정/추가해줘.

## 수정 1: pipeline.py — FastAPI 어댑터 함수 추가
기존 main() 함수는 건드리지 마.
아래 함수들을 추가해줘.

async def extract_overview(document_id: str) -> dict
async def extract_risks(document_id: str) -> dict
async def extract_eligibility(document_id: str) -> dict
async def extract_deliverables(document_id: str) -> dict
async def extract_requirements(document_id: str) -> dict

- 기존 build_retrievers(), build_advanced_chain() 재사용
- 반환값은 6절 스키마와 동일한 구조
- Evidence는 ChunkRecord의 document_title, page_start, text(앞 100자), score로 구성

## 수정 2: generate_answer.py — 스트리밍 메서드 추가
기존 AdvancedRAGChain.invoke()는 건드리지 마.
아래 메서드만 추가해줘.

async def stream_answer(
    self,
    question: str,
    chat_history: list[dict]
) -> AsyncGenerator[str, None]:
    # _advanced_retrieval_flow()로 문서 검색
    # 답변 텍스트를 chunk 단위로 yield: {"answer_chunk": "..."}
    # 종료 시 yield: {"evidences": [...], "low_confidence": bool}
    # Evidence score는 reranker score, 없으면 0.85 기본값

## 건드리지 않는 것
- structured_chunker.py: 표 단위 청킹 이미 완성됨
- retriever.py, embeddings.py, ingestion/: 변경 없음
```

**3단계 — Mock JSON 구조 검증 (세 번째 메시지)**

```
아래는 UI 팀이 Phase 0에서 사용 중인 Mock JSON이야.
네가 수정한 extract_overview()의 실제 출력이
이 Mock과 구조적으로 동일한지 확인하고, 다르면 맞춰줘.

[overview.json 붙여넣기]
```

---

## 13. RAG 연동 방식 (직접 import)

RAG 파이프라인은 별도 서버 없이 FastAPI와 **같은 프로세스**에서 실행된다.
`rag_client.py`가 `pipeline.py`의 어댑터 함수를 import해서 라우터에 제공한다.

```
FastAPI 라우터 → rag_client.py → pipeline.py 어댑터 함수
                                       ↓
                            src/generation/generate_answer.py
                            src/retrieval/retriever.py
                            src/chunking/structured_chunker.py
```

**`backend/services/rag_client.py`**

```python
from pipeline import (
    extract_overview,
    extract_risks,
    extract_eligibility,
    extract_deliverables,
    extract_requirements,
)
from src.generation.generate_answer import AdvancedRAGChain

async def get_overview(document_id: str) -> dict:
    return await extract_overview(document_id)

async def get_risks(document_id: str) -> dict:
    return await extract_risks(document_id)

# ... 나머지 동일 패턴

async def stream_answer(document_id: str, question: str, history: list):
    chain = AdvancedRAGChain(...)
    async for chunk in chain.stream_answer(question, history):
        yield chunk
```

**RAG 함수가 동기(`def`)인 경우**:

```python
import asyncio

async def get_overview(document_id: str) -> dict:
    return await asyncio.to_thread(extract_overview, document_id)
```

---

## 14. Phase 0 → Phase 1 전환 절차

**전환은 3곳만 건드린다.**

```
pipeline.py              ← Codex 어댑터 함수 완성
backend/services/rag_client.py  ← import 경로 확인
frontend/.env.local      ← VITE_USE_MOCK=false
```

**전환 전 Codex와 확인할 것**:

| 항목 | 확인 내용 |
|------|---------|
| 함수 시그니처 | `extract_overview(document_id)` 반환 타입이 6절 스키마와 일치하는지 |
| 비동기 여부 | `async def`인지 — 동기면 `asyncio.to_thread()` 필요 |
| Evidence 포함 | 모든 응답에 `document_name`, `page_number`, `quote`, `score` 있는지 |
| 스트리밍 | `stream_answer()`가 `AsyncGenerator`인지 |

**전환 후 검증 순서** (탭 순서대로):
```
Overview → Go/No-Go → 근거 패널 클릭 → 실행 준비 → 요구사항 → AI 질문
```

---

## 15. RAG 모델 성능 기준

### 임베딩 모델

| 모델 | 한국어 성능 | 로컬 실행 | 추천 |
|------|-----------|---------|------|
| `jhgan/ko-sroberta-multitask` | ✅ | ✅ (~400MB) | ✅ 1순위 |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | ✅ | ✅ | 대안 |
| `text-embedding-3-small` (OpenAI) | ✅ | ❌ API | 노트북 부하 회피 시 |
| `all-MiniLM-L6-v2` | ❌ 한국어 약함 | ✅ | ❌ 비추 |

### 청크 설정

| 항목 | 권장값 |
|------|-------|
| target_tokens | 700 (현재 `chunk_pages()` 기본값 유지) |
| max_tokens | 900 |
| overlap_tokens | 80 |
| Top-K | 3~5 |

> **표 단위 청킹은 `structured_chunker.py`에 이미 구현되어 있다.**
> `content_type: "table_row"`, `"requirement_table_row"` 로 표 행을 별도 처리 중.
> 추가 구현 불필요.

### LLM API 설정

| 항목 | 권장값 |
|------|-------|
| 모델 | GPT-4o-mini (비용·속도 균형) |
| Temperature | 0 (현재 `RAGConfig` 기본값 확인) |
| 응답 형식 | JSON mode |
| 프롬프트 | `configs/prompt.py`의 기존 프롬프트 기반 확장 |

### 데모 최소 성능 목표

| 지표 | 목표 | 측정 방법 |
|------|------|---------|
| 마감일 추출 정확도 | 90% 이상 | 테스트 RFP 10개 수동 검증 |
| Evidence score 평균 | 0.75 이상 | `score` 필드 평균 |
| 실격 조건 누락률 | 10% 이하 | 수동 검토 대비 |
| 평균 응답 시간 | 10초 이하 | Overview 탭 기준 |

---

## 16. Codex 디버깅 질문 대응 가이드

**원칙: 로그/응답값 먼저, 추측 나중**

| Codex가 말할 때 | 되물어야 할 것 |
|----------------|--------------|
| "안 돼요" | "어떤 에러 메시지야? 콘솔/터미널 로그 붙여줘" |
| "값이 이상해요" | "Evidence `quote`랑 `score`가 뭐야?" |
| "프론트가 깨져요" | "Network 탭 응답 raw 값 붙여줘" |
| "스트리밍이 이상해요" | "curl로 직접 쳐봤어? 청크가 오긴 해?" |

### 유형 1. 추출값이 틀리거나 누락

| 증상 | 원인 | 조치 |
|------|------|------|
| `score` < 0.5 | 청크가 잘못 잘렸거나 임베딩이 표현을 못 잡음 | `chunk_pages()` target_tokens 조정 |
| `quote`가 엉뚱한 문장 | 프롬프트 추출 기준 모호 | `configs/prompt.py` few-shot 예시 추가 |

### 유형 2. API 응답이 스키마와 불일치

1. FastAPI `/docs` 또는 `curl`로 raw 응답 확인
2. 6절 스키마 필드명·타입 대조 (`snake_case` vs `camelCase`)
3. `evidence` 필드가 `null` 대신 빈 객체 `{}`로 오는 경우 → `null`로 통일

### 유형 3. 스트리밍 불량

```bash
# 스트리밍 직접 확인
curl -N -X POST http://localhost:8000/api/analysis/{id}/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "실격 조건은?", "chat_history": []}'
```

청크가 한 줄씩 흐르면 서버 정상. 한 번에 오면 `yield` 문제.

확인 순서:
1. `stream_answer()`가 `yield`로 청크를 내보내는지
2. FastAPI 라우터에 `StreamingResponse` + `media_type="text/event-stream"` 설정 여부
3. 각 청크가 `data: {text}\n\n` 형식인지 (개행 2개 필수)

### 유형 4. Phase 0 → Phase 1 전환 후 화면 깨짐

1. `VITE_USE_MOCK=true`로 되돌려서 Mock은 정상인지 먼저 확인
2. Network 탭에서 실제 API 응답 raw 값 확인
3. `src/mocks/` Mock JSON과 실제 응답 diff
4. 14절 전환 체크리스트 재확인

---

**작성자**: 기획 검토 세션
**다음 단계**: Phase 0 Mock UI 착수 (UI 팀) → Codex 구현 지시 (12절 순서대로) → Phase 1 연동

---

## 17. 채팅 히스토리 관리 (컨텍스트 압축)

### 두 종류의 컨텍스트

| 종류 | 위치 | 설명 |
|------|------|------|
| 채팅 히스토리 | Tab 5 AI 질문 | `chat_history` 배열로 매 요청마다 LLM에 전송 |
| RAG 검색 컨텍스트 | `generate_answer.py` | 검색된 청크 텍스트를 LLM 프롬프트에 주입 |

### 문제: 채팅이 길어지면 토큰 비용·속도 증가

`chat_history`가 쌓일수록 매 요청의 토큰 수가 늘어난다.
데모 수준에서는 **프론트엔드 슬라이딩 윈도우**로 처리한다.

```typescript
// frontend/src/api/client.ts
// 전송 전 최근 6턴만 슬라이싱 (3왕복)
const trimmedHistory = chatHistory.slice(-6);

const res = await fetch(`/api/analysis/${documentId}/ask`, {
  method: "POST",
  body: JSON.stringify({
    question,
    chat_history: trimmedHistory,  // 전체가 아닌 최근 6턴만
  }),
});
```

**왜 프론트엔드에서 처리하나**: 백엔드를 건드리지 않아도 되고,
Zustand `chatHistory` 배열은 전체를 유지하되 전송할 때만 슬라이싱하면
UI에서는 전체 히스토리가 보이고 LLM에는 최근 6턴만 간다.

### 운영 전환 시 고려할 방법

| 방법 | 설명 | 적용 시점 |
|------|------|---------|
| 슬라이딩 윈도우 (현재) | 최근 N턴만 전송 | 데모 |
| 요약 압축 | 오래된 히스토리를 LLM으로 한 줄 요약 | Phase 3 |
| LangChain Memory | `ConversationSummaryBufferMemory` 사용 | 운영 전환 시 |

### RAG 컨텍스트 압축은 별도 영역

청크 수(Top-K) 조정과 재랭킹은 `src/retrieval/retriever.py`의 문제다.
채팅 히스토리 관리와 독립적으로 다룬다.
