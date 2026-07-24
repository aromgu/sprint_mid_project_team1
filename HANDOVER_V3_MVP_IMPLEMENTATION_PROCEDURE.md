# Handover v3 전체 MVP 구현 절차

## 1. 목표

공공 RFP를 처음 검토하는 담당자가 짧은 시간 안에 다음을 확인할 수 있는 MVP를 만든다.

- 제출 마감일
- 질의 마감일
- 참가 자격
- 실격·감점 위험
- 필수 제출물
- 지금 해야 할 일
- 각 판단의 문서·페이지·요구사항 근거

핵심 목표는 완성형 운영 시스템이 아니라, 실제 9개 RFP 문서를 대상으로 검색·분석·인용이 동작하는 발표 가능한 제품이다.

## 2. 기본 원칙

- 현재 RAG 엔진은 유지하고 adapter만 추가한다.
- API 계약을 먼저 고정한 뒤 프론트와 백엔드를 연결한다.
- 모든 분석 결과는 `Evidence`를 포함한다.
- 검색 결과나 근거가 부족하면 답변·판정을 유보한다.
- 모든 탭을 동시에 개발하지 않고 세로 방향(vertical slice)으로 완성한다.
- 처음에는 일반 JSON 응답으로 안정화한 뒤 필요한 경우 스트리밍을 추가한다.

## 3. 저장소와 재사용 범위

현재 GIT 프로젝트를 기준 저장소로 사용한다.

재사용 대상:

- `src/search/`: BM25, Dense, Hybrid, Reranker
- `src/generation/`: OpenAI RAG 답변 생성
- `data/processed/`: pages·chunks·manifest
- Team1_RAG frontend의 채팅, 문서 뷰어, TOC, 출처 태그, 하이라이트 UI

새로 구성할 대상:

- Handover v3 Workspace와 탭 구조
- FastAPI 분석 라우터
- RAG adapter
- Evidence 모델
- 사용자 상태 저장

MVP에서 제외:

- 인증·권한
- PostgreSQL
- 모바일 반응형
- 복잡한 관리자 기능
- 완성형 평가 대시보드

## 4. 단계별 구현 순서

### 1단계 — 범위와 구조 확정

확정할 항목:

- 기준 저장소와 실행 명령
- 문서 ID 규칙
- 프론트·백엔드 디렉터리
- MVP에서 제외할 기능
- 개발용 API base URL

완료 기준:

- 작업 대상과 제외 범위가 문서로 고정됨
- 팀 간 파일 소유권과 실행 방법이 명확함

### 2단계 — API 계약과 Pydantic 모델 고정

먼저 다음 공통 모델을 만든다.

```text
Evidence
OverviewResponse
RiskItem
EligibilityItem
DeliverableItem
RequirementItem
AnswerResponse
```

라우트:

```text
GET  /api/analysis/{document_id}/overview
GET  /api/analysis/{document_id}/risks
GET  /api/analysis/{document_id}/eligibility
GET  /api/analysis/{document_id}/deliverables
GET  /api/analysis/{document_id}/requirements
POST /api/analysis/{document_id}/ask
PATCH /api/state/{document_id}/eligibility/{item_id}
PATCH /api/state/{document_id}/deliverable/{item_id}
```

초기에는 mock 응답이어도 된다. 필드명·타입·오류 형식을 먼저 고정하는 것이 목적이다.

### 3단계 — 현재 RAG용 adapter 추가

기존 검색·생성 코드를 직접 수정하지 않고 다음 계층을 추가한다.

```text
FastAPI router
    ↓
backend/services/rag_client.py
    ↓
SearchService / OpenAIRAGService
```

구현 순서:

1. `retrieve_evidence()`
2. `answer_question()`
3. `extract_overview()`
4. `extract_risks()`
5. `extract_eligibility()`
6. `extract_deliverables()`
7. `extract_requirements()`

각 함수는 반드시 동일한 `Evidence` 모델을 사용한다.

### 4단계 — 첫 번째 세로 슬라이스

다음 흐름을 먼저 실제 RAG와 연결한다.

```text
문서 선택
→ Overview API 호출
→ 분석 카드 표시
→ 카드 클릭
→ Evidence 패널 표시
```

이 흐름이 동작하면 RAG, API, 프론트 상태, 근거 표시의 기본 구조가 검증된다.

### 5단계 — Overview와 Evidence 패널

Overview 카드:

- 제출 마감일
- 질의 마감일
- 참여 검토 상태
- 위험 건수
- 제출물 진행률
- 지금 할 일
- confidence

각 항목 클릭 시 오른쪽 Evidence 패널에 다음을 표시한다.

- 문서명
- 페이지
- 요구사항 ID
- 원문 인용
- 점수 또는 confidence

### 6단계 — Go/No-Go와 사용자 상태

- 자격 조건 목록
- `충족 / 미충족 / 확인 필요` 선택
- 실격·감점·검토 필요 위험 표시
- PATCH API 구현
- MVP에서는 `user_state.json`으로 저장

### 7단계 — 실행 준비와 요구사항

실행 준비 탭:

- 제출물명
- 형식·수량
- 날인·원본 필요 여부
- 담당자
- 제출 상태
- 마감일
- Evidence

요구사항 탭:

- 요구사항 ID
- 유형
- 제목·설명
- 중요도
- 검토 상태
- Evidence

MVP에서는 복잡한 테이블 라이브러리보다 일반 HTML table로 먼저 구현할 수 있다.

### 8단계 — AI 질문

초기 구현:

```text
질문
→ Hybrid 검색
→ gpt-5-nano
→ 답변 + citations + low_confidence
```

일반 JSON 응답으로 안정화한 뒤 필요할 때 `StreamingResponse`와 SSE를 추가한다.

질문 답변에도 반드시 다음을 포함한다.

- 답변
- 인용 문서·페이지
- 사용 청크
- 근거 부족 여부

### 9단계 — 오류와 신뢰도 처리

공통 처리 항목:

- 검색 결과 없음
- 근거 없음
- 낮은 confidence
- LLM timeout
- API 오류
- 분석 중 상태
- 재시도 버튼

유효한 Evidence가 없으면 답변이나 판정을 확정하지 않는다.

### 10단계 — 실제 문서 검증과 발표 시나리오

다음 질문으로 최소 검증한다.

- 제출 마감은 언제인가?
- 참가 자격은 무엇인가?
- 실격 조건은 무엇인가?
- 필수 제출물은 무엇인가?
- 특정 요구사항의 원문은 어디에 있는가?
- 근거가 없을 때 답변을 유보하는가?

발표용으로는 대표 RFP 한 건과 성공 질문 세 개를 고정한다.

## 5. 권장 작업 순서

```text
API 모델·계약
→ RAG adapter
→ Overview + Evidence
→ Go/No-Go
→ 사용자 상태 저장
→ 실행 준비
→ 요구사항
→ AI 질문
→ 오류 처리
→ 발표 검증
```

## 6. 예상 일정

| 단계 | 예상 기간 |
|---|---:|
| API 계약·백엔드 골격 | 1일 |
| RAG adapter·Evidence | 1~2일 |
| Overview·Evidence 패널 | 1~2일 |
| Go/No-Go·상태 저장 | 1일 |
| 실행 준비·요구사항 | 1~2일 |
| AI 질문·오류 처리 | 1일 |
| 통합·발표 검증 | 1일 |
| 합계 | 약 7~10일 |

## 7. 완료 기준

Handover v3 MVP는 다음을 만족하면 완료로 본다.

- 실제 RFP 문서를 선택할 수 있다.
- Overview가 실제 검색·LLM 결과를 표시한다.
- Go/No-Go 항목을 확인하고 상태를 저장할 수 있다.
- 제출물과 요구사항을 표 형태로 확인할 수 있다.
- 모든 핵심 판단에 Evidence가 연결된다.
- AI 질문에 OpenAI가 답하고 인용을 반환한다.
- 근거 부족·오류·timeout 상태가 사용자에게 표시된다.
- 대표 발표 시나리오를 처음부터 끝까지 재현할 수 있다.

## 8. 효율화 체크리스트

- [ ] 프론트 전체를 먼저 만들지 않고 Overview 세로 슬라이스부터 완료
- [ ] 검색·생성 핵심 코드를 직접 수정하지 않고 adapter로 연결
- [ ] API 모델과 Evidence 필드를 먼저 고정
- [ ] mock과 실제 API 응답의 JSON 구조를 동일하게 유지
- [ ] 스트리밍은 일반 응답 검증 이후 추가
- [ ] user state는 MVP에서 JSON 파일로 제한
- [ ] 발표용 대표 문서·질의·응답을 미리 고정
