# RFP Action Copilot MVP 범위와 남은 작업

작성일: 2026-07-22

## 1. 현재 프로젝트 목표

9개 RFP PDF를 대상으로 문서 검색, RFP 분석, 근거 확인, 제출 준비 상태 관리를 제공하는 부트캠프 수준의 RAG MVP를 구현한다.

핵심 사용 흐름은 다음과 같다.

```text
문서 선택
  → Overview 확인
  → Go / No-Go 판단
  → 제출물·요구사항 확인
  → AI 질문
  → 근거 페이지 검증
```

## 2. 구현 완료 범위

### 데이터·검색

- 9개 PDF ingestion
- 페이지·chunk 생성
- OCR 후보 페이지 처리
- chunk 품질 리포트
- BM25 검색
- Dense 검색
- Hybrid/RRF 검색
- 선택형 reranker
- 검색 설정 YAML 분리
- 검색 latency 프로파일링

### RAG·분석

- OpenAI RAG 답변
- `gpt-5-nano` 기본 모델 설정
- 구조화된 Overview 분석
- Go / No-Go 분석
- 참가 자격 분석
- 제출물 분석
- 요구사항 분석
- 답변 인용 검증
- 최근 6턴 채팅 히스토리
- SSE-compatible 답변 API

### Workspace UI

- 5개 탭 구조
  - Overview
  - Go / No-Go
  - 실행 준비
  - 요구사항
  - AI 질문
- Overview KPI 카드
- 실격·감점 위험 카드
- 참가 자격 상태 변경
- 제출물 상태 변경
- 우측 Evidence 패널
- 원문 페이지 검색
- TOC 및 페이지 뷰어
- 근거 페이지 이동
- AI 추천 질문
- 오류 재시도 버튼
- 밝은 목표 UI 테마

### 문서·운영 기능

- Mock 모드

```bash
VITE_USE_MOCK=true npm run dev
```

- 실제 API 모드
- PDF 업로드 후 페이지·chunk 생성
- 업로드 문서 검색 인덱스 갱신
- 자동 API Smoke Test
- 내부 Evaluation 화면
- 파이프라인 단계별 Evaluation 탭
- Golden set 입력 변환 스크립트
- RAGAS 실행기 기반

## 3. 현재 검증 상태

```text
Python tests: 19 passed
Frontend build: 성공
```

검증 명령:

```bash
.venv/bin/pytest -q
cd frontend && npm run build
```

API Smoke Test:

```bash
uv run python -m scripts.smoke_test
```

## 4. 반드시 남은 작업

### 4.1 Golden set 기반 평가

가장 중요한 남은 작업이다.

- Golden set JSONL 수령
- 질문·정답·근거 chunk 검증
- Retriever 평가 실행
- Recall@k·MRR·nDCG 산출
- RAGAS 실행
- Faithfulness 산출
- Answer Relevancy 산출
- Context Precision 산출
- Context Recall 산출
- Evaluation 화면에 실제 점수 표시
- 실패 질문과 누락 근거 분석

실행 예:

```bash
uv run python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all

uv run python -m scripts.evaluate_ragas \
  --golden data/eval/golden_set.jsonl
```

### 4.2 실제 마감일·D-Day 데이터

현재 RFP 본문에 날짜가 없는 경우 다음처럼 표시한다.

```text
입찰공고 참조
RFP 본문 미기재
```

향후 선택지는 다음과 같다.

- 나라장터 입찰공고 연동
- 별도 metadata JSON/DB 관리
- 사용자 수동 마감일 입력

### 4.3 사용자 피드백 반영

- 카드 크기·간격 조정
- 좌측 문서 목록 폭 조정
- Evidence 패널 정보량 조정
- KPI 배치 조정
- 탭 명칭 및 순서 조정
- 모바일·작은 화면 대응

### 4.4 운영 안정화

- 업로드 실패 메시지 개선
- 중복 PDF 처리
- 대용량 PDF timeout 처리
- 인덱스 재빌드 진행 상태 표시
- OpenAI timeout·재시도 정책
- API 오류 유형별 메시지
- 로그·latency·비용 모니터링
- 배포용 실행 스크립트

## 5. 선택적 후속 작업

- 실제 토큰 단위 OpenAI streaming 최적화
- 내부 Evaluation 실행 버튼
- Evaluation 결과 JSON 다운로드
- 실패 질문 상세 화면
- Chunk / Index 자동 health metric 연결
- E2E 업무 성공률 지표
- 사용자 인증·권한
- 업로드 문서 영속 저장소
- SQLite/PostgreSQL 상태 저장
- Docker 배포

## 6. 권장 진행 순서

```text
1. Golden set 수령
2. Retriever 평가 실행
3. RAGAS 평가 실행
4. Evaluation 화면에 숫자 연결
5. 실패 질문 원인 분석
6. 검색·chunk·prompt 보정
7. 사용자 UI feedback 반영
8. 업로드·운영 안정화
9. 최종 데모 리허설
```

## 7. 현재 결론

부트캠프 데모에 필요한 MVP 기능은 구현 완료 상태다. 이후 작업의 중심은 새로운 대형 기능 추가가 아니라 Golden set 기반 품질 평가, 실패 사례 개선, 실제 사용자 feedback 반영, 배포 안정화다.
