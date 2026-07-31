# Main Pipeline 기반 10개 문서 RAG 적용안

## 1. 목적

`../sprint_mid_project_team1_main/PIPELINE_COMPARISON_GIT_ADVAN.md`의 비교 결과를
기준으로, `GIT_advan`의 FastAPI·React 서비스 구조를 유지하면서 Main Pipeline의
검색 방식을 10개 문서 corpus에 적용하는 방안을 정의한다.

이 방안은 Main Pipeline의 98개 문서 인덱스를 그대로 이식하는 것이 아니다.
동일한 검색 방법을 사용하는 별도의 10문서 서비스 계약을 구축하는 것이다.

```text
질의 재작성
  -> BM25 후보 검색
  -> Dense MMR 후보 검색
  -> RRF 결합
  -> 선택적 Qwen3 reranking
  -> 최종 top 5
  -> 답변 생성
  -> citation allowlist 검증
```

## 2. 98개 기준안과의 차이

| 항목 | Main Pipeline 기준 | 10개 제한안 |
|---|---:|---:|
| corpus | 고유 문서 98개 | 명시적으로 선정한 10개 |
| chunk | 1024 tokens / overlap 102 | 동일 |
| Dense | OpenAI embedding + Chroma MMR | 동일 |
| Sparse | BM25 필수 | 동일하게 권장 |
| 결합 | BM25 0.7 + Dense 0.3, RRF | 동일하게 적용 가능 |
| reranker | Qwen3-Reranker-0.6B | 기능 플래그로 적용 권장 |
| 최종 결과 | top 5 | top 5 |
| 초기 인덱싱 비용 | 큼 | 크게 감소 |
| corpus-wide 검색 범위 | 98개 | 10개 한정 |
| 평가 일반화 | 상대적으로 높음 | 낮음 |

문서 수가 작아져도 BM25의 필요성이 사라지지는 않는다. 기관명, 조항 번호,
별지 번호, 제출 부수처럼 정확한 문자열이 중요한 질의는 Dense 검색만으로 놓칠 수
있다. 다만 후보 수와 reranker 적용 여부는 10개 corpus에 맞춰 재측정해야 한다.

## 3. 인덱스 계약

현재 보고서에 기록된 9개 문서, 512/51 인덱스에 문서 하나만 추가해서는 안 된다.
10개 문서를 확정한 후 전체 문서를 1024/102로 다시 청킹하고, 동일한 청크 집합으로
Chroma와 BM25를 다시 구축해야 한다.

권장 설정은 다음과 같다.

```yaml
corpus:
  contract_id: main_advanced_10_v1
  expected_document_count: 10
  manifest: data/main_advanced/manifest/documents_10.jsonl

chunking:
  strategy_id: advanced_1024_102_v1
  chunk_size: 1024
  chunk_overlap: 102

paths:
  chroma: data/main_advanced/chroma_10_1024
  bm25: data/main_advanced/bm25_10_1024/bm25_index.pkl
  bm25_manifest: data/main_advanced/bm25_10_1024/bm25_manifest.json

index:
  collection_name: ai11_policy_advanced_10_v1_1024
  embedding_model: text-embedding-3-small
  embedding_dimension: 1536

retrieval:
  mode: hybrid_rerank
  bm25_weight: 0.7
  dense_weight: 0.3
  dense_search: mmr
  dense_candidate_k: 12
  bm25_candidate_k: 12
  rrf_candidate_k: 15
  rerank_top_k: 5
  reranker:
    enabled: true
    fallback_to_rrf: true
```

원본의 `ai11_policy_advanced_v2_1024` 이름은 사용하지 않는다. 같은 이름을 쓰면
98개 문서 인덱스와 호환된다는 오해가 생길 수 있다.

서비스 시작 시 다음을 검증한다.

1. manifest의 고유 문서 수가 정확히 10개인지 확인
2. SHA-256 기준 중복 문서 확인
3. 10개 문서의 전처리 및 청킹 완료 여부 확인
4. Chroma collection, record 수, embedding 설정 확인
5. BM25 artifact와 manifest 존재 여부 확인
6. Dense와 BM25의 `chunk_id` 집합 일치 여부 확인
7. source chunks SHA-256과 chunk strategy ID 확인
8. BM25 색인 및 질의 tokenizer 정책 확인

불일치 시 검색을 진행하지 않고 `index_contract_mismatch`와 같은 복구 가능한
오류를 API에서 반환한다.

## 4. 검색 및 생성 계층

`AdvancedRetriever`는 현재 Dense similarity 검색 대신 다음 단계를 담당한다.

1. `document_id`가 동일하게 적용된 Dense MMR 및 BM25 검색
2. RRF 결합과 중복 chunk 제거
3. 선택적 Qwen3 reranking
4. 최종 top 5 반환
5. 단계별 rank와 score 기록

권장 검색 결과 계약은 다음과 같다.

```json
{
  "chunk_id": "...",
  "text": "...",
  "score": 0.86,
  "score_type": "reranker",
  "dense_rank": 4,
  "bm25_rank": 1,
  "rrf_score": 0.029,
  "reranker_score": 0.86,
  "retriever": "hybrid_rrf_qwen"
}
```

Qwen reranker를 사용할 수 없을 때는 RRF 결과로 fallback할 수 있지만, 실제 사용된
검색 모드와 fallback 이유를 응답 및 관측 로그에 남긴다.

```json
{
  "retriever": "hybrid_rrf",
  "reranker_applied": false,
  "fallback_reason": "reranker_unavailable"
}
```

현재 서비스의 다음 기능은 그대로 유지한다.

- OpenAI/Gemini provider 선택
- conversation/document/provider별 세션
- SSE 답변
- citation allowlist 검증
- YAML 및 workspace 상대경로 설정
- GO/NO-GO, 위험, 제출물, 요구사항 분석
- 내부 평가 화면

## 5. 검색 범위

사용자 UI의 기본 검색은 선택한 문서로 제한한다.

```text
10개 전체 인덱스
  -> 사용자가 문서 A 선택
  -> document_id 필터
  -> 문서 A의 청크만 검색
```

이 방식에서는 문서 수가 10개인지 98개인지보다 선택 문서의 청크 수, 후보 수,
reranker 실행 환경이 사용자 응답시간에 더 큰 영향을 줄 수 있다.

전체 문서 검색을 제공하려면 다음을 별도 기능으로 추가해야 한다.

- 검색 범위 `현재 문서 / 전체 문서` 선택
- 기관, 연도, 문서 유형 필터
- 문서별 검색 결과 그룹
- 여러 문서 citation 표시

10개 제한안의 전체 검색은 다음 용도에는 한계가 있다.

- 다수 기관 공고의 경향 비교
- 연도별 조건 변화 분석
- 다양한 산업과 사업 유형의 유사 사례 검색
- 대규모 corpus 기반 추천

## 6. 업로드 정책

10개 제한을 적용하려면 11번째 문서 처리 정책이 필요하다. 권장 정책은 자동 삭제가
아니라 업로드 전 거부 후 사용자가 교체 대상을 선택하도록 하는 것이다.

UI에는 `등록 문서 8 / 10`과 같은 사용량을 표시한다. 문서별 상태는 다음과 같이
관리한다.

```text
uploading
  -> parsing
  -> chunking
  -> dense_indexing
  -> bm25_indexing
  -> validating
  -> ready | failed
```

`ready` 이전에는 해당 문서의 분석 탭과 질문 기능을 비활성화한다.

BM25 갱신은 전역 pickle을 매번 덮어쓰기보다 문서별 shard를 생성하고 검색 시
결합하는 방식을 우선 검토한다. 이는 동시 업로드와 파일 손상 위험을 줄인다.

## 7. UI 영향

화면 레이아웃을 전면 변경할 필요는 없다. 주요 변경은 상태와 점수 표현이다.

### 7.1 답변 진행 상태

reranker가 추가되면 응답 지연이 증가할 수 있으므로 다음 단계를 표시한다.

```text
질문 분석 중
  -> 문서 검색 중
  -> 근거 순위 조정 중
  -> 답변 생성 중
```

### 7.2 근거 신뢰도

Dense relevance, BM25, RRF, reranker 점수는 서로 동일한 척도가 아니다. 기존처럼
모든 `score < 0.7`을 낮은 신뢰도로 표시하면 잘못된 경고가 발생할 수 있다.

사용자 UI에는 정규화된 최종 reranker score만 표시하거나 점수 대신 `근거 순위`를
표시한다. 내부 평가 UI에서는 단계별 score와 rank를 모두 제공할 수 있다.

### 7.3 분석 결과 변화

정확한 문자열 검색이 강화되면서 다음 결과의 항목 수, 순서, 근거 페이지가 바뀔 수
있다.

- GO/NO-GO 실격 및 감점 카드
- 제출물 목록
- 요구사항 목록
- overview 요약
- AI 답변 citation

따라서 API schema가 동일하더라도 UI snapshot과 live E2E 기대값을 재검증한다.

### 7.4 내부 평가 화면

다음 항목을 표시한다.

- Dense, BM25, Hybrid, Reranker별 Recall@k와 MRR
- 단계별 p50/p95 latency
- candidate 수와 rerank 수
- corpus contract 및 chunk strategy ID
- Chroma-BM25 계약 검증 상태
- 실제 적용된 retriever 및 fallback 여부

## 8. 평가 설계

10개 제한의 가장 큰 약점은 평가 일반화가 낮아진다는 점이다. 10개 문서 전체에서
질문을 균형 있게 작성한다.

권장 기준은 다음과 같다.

- 문서당 10~20개 질문
- 전체 100~200개 질문
- 정확 문자열형 질문
- 의미형 질문
- 표 기반 질문
- 답이 없는 질문
- 여러 페이지 근거가 필요한 질문
- 유사 조항 때문에 오검색하기 쉬운 질문

평가는 두 종류로 분리한다.

1. 선택 문서 검색: 질문과 정답 `document_id`를 제공한다.
2. 전체 10개 검색: 문서 필터 없이 정답 문서와 청크를 찾아야 한다.

선택 문서 검색만 평가하면 검색 문제가 지나치게 쉬워져 성능이 과대평가될 수 있다.

비교 순서는 다음과 같다.

1. Dense only
2. BM25 + Dense RRF
3. BM25 + Dense RRF + Qwen reranker

각 단계의 Recall@k, MRR, 답변 품질, p50/p95 latency를 측정한 뒤 기본 모드를
결정한다. 긴 평가 실행은 `--max-workers`와 `--resume`을 제공하고, 실제 동시성과
경과 시간을 기록한다.

## 9. Main Pipeline 대비 제외되는 범위

10개 제한안에는 다음 Main Pipeline 특성이 포함되지 않는다.

- 98개 전체 corpus 호환성
- 원본 `ai11_policy_advanced_v2_1024` 인덱스 계약
- 원본 평가 결과와의 직접 비교
- 98개 문서를 가로지르는 검색
- 다양한 문서 유형에 대한 일반화 근거
- 원본 Chroma/BM25 artifact의 직접 재사용
- Qwen reranker 상시 실행 보장
- 원본 CLI와 완전히 동일한 실행 결과
- `/home/data` 절대경로와 코드 상수 기반 설정
- 단일 provider 및 단일 session 구조

반대로 FastAPI, React, SSE, 다중 provider, 문서별 세션, 업로드, citation 검증,
운영 UI는 현재 `GIT_advan`의 기능을 유지한다.

## 10. 적용 순서

1. 대상 10개 문서와 corpus manifest 확정
2. 1024/102 chunk 계약 등록
3. 10개 전체 전처리 및 청킹
4. 동일 chunks로 새 Chroma와 BM25 구축
5. 시작 시 인덱스 계약 검증 추가
6. Hybrid RRF retriever 구현
7. Qwen reranker를 기능 플래그로 추가
8. backend의 retriever 이름과 score 계약 수정
9. 업로드 제한, BM25 갱신, 문서 상태 API 추가
10. UI 진행 상태와 근거 점수 표현 수정
11. Dense/Hybrid/Reranker targeted 평가
12. 전체 regression 및 live smoke 후 기본 모드 결정

## 11. 결론

10개 제한은 선택한 RFP를 분석하는 웹 MVP에 적합하다. 인덱싱 비용과 운영 복잡도를
낮추면서 BM25 + Dense MMR + RRF + 선택적 reranking의 검색 품질 개선을 적용할 수
있다.

다만 이 결과는 Main Pipeline의 98개 인덱스와 동일하거나 호환되는 pipeline이
아니다. `main_advanced_10_v1`이라는 별도 corpus 계약으로 관리하고, 문서 수 제한,
전체 검색 범위, 평가 일반화의 한계를 명시해야 한다.
