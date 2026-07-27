# 오늘 할 일 — 2026-07-24

## 오늘의 목표

현재 구현된 임베딩 검색이 실제 RAG 답변 경로에서도 사용되도록 정리하고,
자동 테스트와 CI로 동작을 검증한다.

## P0. AI 답변 검색에 Hybrid 적용

현재 기본 검색기는 `hybrid`이지만 `src/generation/openai_generator.py`의 답변
생성 경로는 하위 질문 검색 시 `retriever="bm25"`를 강제로 사용한다.

- [ ] BM25 강제 지정을 제거한다.
- [ ] 전달된 `retriever`가 있으면 해당 값을 사용한다.
- [ ] 값이 없으면 `configs/search.yaml`의 기본값인 `hybrid`를 사용한다.
- [ ] 복합 질문의 모든 하위 질문에도 같은 검색기를 적용한다.
- [ ] 응답의 `retriever` 필드가 실제 사용한 검색기를 표시하는지 확인한다.

완료 조건:

- 기본 AI 답변 요청에서 BM25와 Dense embedding 검색이 함께 실행된다.
- `retriever="bm25"`를 명시하면 기존 BM25 전용 검색도 가능하다.
- 관련 단위 테스트가 통과한다.

## P0. 테스트 및 GitHub Actions 정상화

현재 `.github/workflows/rag-test.yml`은 저장소에 없는 `requirements.txt`와
`test_rag_pipeline.py`를 참조한다.

- [ ] 의존성 설치를 `uv sync --dev` 또는 프로젝트에 맞는 `uv sync`로 변경한다.
- [ ] 테스트 대상을 현재 `tests/` 디렉터리로 변경한다.
- [ ] 필요한 경우 Git LFS 파일 처리 여부를 워크플로에 명시한다.
- [ ] API 키가 없는 PR에서도 비과금 단위 테스트가 실행되도록 분리한다.
- [ ] 평가 작업은 API 키가 있을 때만 실행하도록 조건을 설정한다.

완료 조건:

```bash
uv run pytest
```

위 명령이 로컬과 GitHub Actions에서 성공한다.

## P1. Embedding 검색 검증

현재 Dense 인덱스 상태:

- 모델: `jhgan/ko-sroberta-multitask`
- 청크 수: 1,042개
- 차원: 768
- 인덱스: `data/indexes/dense_jhgan--ko-sroberta-multitask.npy`
- 저장 방식: Vector DB가 아닌 NumPy 메모리 매핑 및 내적 검색

- [ ] Dense 인덱스 메타데이터와 현재 `chunks.jsonl` fingerprint가 일치하는지 확인한다.
- [ ] 동일 질문을 BM25, Dense, Hybrid로 검색해 상위 결과를 비교한다.
- [ ] Hybrid 검색 결과에 `bm25`, `dense` component rank가 모두 기록되는지 확인한다.
- [ ] 첫 요청과 warm 요청의 latency를 측정한다.

검증 명령:

```bash
uv run python -m scripts.run_search \
  --query "SFR-007 예약 시스템 기능" \
  --retriever bm25 \
  --top-k 5

uv run python -m scripts.run_search \
  --query "SFR-007 예약 시스템 기능" \
  --retriever dense \
  --top-k 5

uv run python -m scripts.run_search \
  --query "SFR-007 예약 시스템 기능" \
  --retriever hybrid \
  --top-k 5

uv run python -m scripts.profile_latency
```

## P1. 복합 질문 답변 품질 개선

- [ ] 매칭되지 않은 하위 질문도 검색 계획에 보존한다.
- [ ] 하위 질문별 검색 결과와 인용을 LLM 입력에서 구분한다.
- [ ] 일부 항목만 근거가 있을 때 부분 답변을 허용한다.
- [ ] 부분 답변과 `is_answerable` 판정 기준을 테스트로 고정한다.
- [ ] 검색된 근거가 없는 출처 라벨은 계속 제거한다.

## P2. Golden set 소규모 재평가

P0와 P1 수정 후 먼저 20~30문항으로 평가한다.

- [ ] OpenAI와 Gemini를 동일 문항으로 실행한다.
- [ ] Faithfulness와 Answer Relevancy를 재측정한다.
- [ ] 답변 정확도, latency, 토큰 및 예상 비용을 비교한다.
- [ ] Gemini 가격 설정을 추가해 비용을 실제로 기록한다.
- [ ] 결과에 따라 기본 모델과 fallback 정책을 결정한다.

## Vector DB 결정

오늘은 FAISS나 Chroma로 이전하지 않는다. 현재 1,042개 청크에서는 NumPy 전체
검색으로 충분하며, 저장소 변경만으로 답변 정확도가 개선되지는 않는다.

다음 조건이 생기면 다시 검토한다.

- FAISS: 청크가 수십만 개 이상으로 증가하고 검색 성능이 중요할 때
- Chroma: PDF 추가·삭제와 메타데이터 필터 관리가 빈번할 때
- 서버형 Vector DB: 여러 서버나 사용자가 동일 인덱스에 접근해야 할 때

## 마무리 체크

- [ ] `uv run pytest` 통과
- [ ] `cd frontend && npm run build` 통과
- [ ] AI 답변 응답에서 실제 검색기 확인
- [ ] 변경 전후 대표 질문 검색 결과 비교
- [ ] 변경 내용과 평가 결과를 작업 로그에 기록

