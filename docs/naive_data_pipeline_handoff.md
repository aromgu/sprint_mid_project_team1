# Naive 데이터 파이프라인 수정·인계 정리

## 한눈에 보는 현재 상태

```text
원본 HWP·PDF 100개
  → 중복 정리 후 대표 문서 98개
  → 구조 전처리 v2
  → Naive 청킹 v3
  → 사업 메타데이터 보강 v1
  → [대기] GCP에서 임베딩·공용 Vector DB 생성
```

현재 로컬에서 완료한 범위는 **사업 메타데이터 보강까지**입니다. 임베딩과
Vector DB 생성은 팀원이 같은 결과를 사용할 수 있도록 로컬에서 실행하지 않고
GCP 실행을 기다립니다.

| 단계 | 상태 | 핵심 결과 |
| --- | --- | --- |
| PDF 전처리 보정 | `naive` 머지 완료 | 불완전한 PDF 표에서 누락되던 텍스트 복구 |
| Naive 청킹 v3 | `naive` 머지 완료 | 병합표 청크 폭증 방지, 최종 31,627청크 |
| 사업 메타데이터 보강 | PR #24 검토 중 | 98문서 모두 연결, 미연결 0 |
| 임베딩·Vector DB | GCP 실행 대기 | 로컬 실행하지 않음 |

## 왜 수정했는가

처음 `naive` 코드를 실제 원본 파일로 점검했을 때 다음 세 가지 문제가
확인되었습니다.

1. PDF에서 표 테두리는 인식했지만 긴 병합 셀의 본문을 놓치면, 해당 영역의
   텍스트가 표와 본문 양쪽에서 모두 빠질 수 있었습니다.
2. HWP 병합표의 참조 문구가 반복되면서 표 하나가 최대 2,532청크로 잘리고,
   전체 청크가 73,301개까지 늘어났습니다.
3. 최종 청크에는 사업명·발주기관·공고번호·사업금액·사업요약이 붙지 않아
   Vector DB에서 사업별 필터와 출처 표시를 할 수 없었습니다.

## 1. PDF 표 텍스트 누락 보정

- 관련 PR: [#21 fix: preserve text from incomplete PDF tables](https://github.com/aromgu/sprint_mid_project_team1/pull/21)
- 주요 코드: `src/preprocessing/clean_text.py`

### 기존 문제

`pdfplumber`가 표의 경계는 찾았지만 셀 내용을 충분히 추출하지 못하는 경우가
있었습니다. 기존 코드는 표 영역으로 판단한 단어를 본문에서 제거했기 때문에,
불완전한 표 행렬에도 없는 텍스트가 최종 결과에서 사라질 수 있었습니다.

### 변경 내용

- 표 영역 원문 글자 수와 셀 행렬 글자 수를 비교합니다.
- 원문이 40자 이상이고 표 행렬의 보존율이 80% 미만이면 불완전한 표로
  판정합니다.
- 불완전한 표 행렬은 색인에서 제외하고, 표 경계 안의 원문 단어를 별도 본문
  블록으로 복구합니다.
- 복구된 블록에는 `pdf_table_text_fallback` 품질 플래그를 기록합니다.

### 실제 전체 결과

- PDF 4개 모두 처리 성공
- 불완전한 PDF 표 139개, 60페이지에서 원문 fallback 적용
- 검출된 불완전 표 영역의 원문을 보존하도록 구조 전처리 결과 재생성

## 2. 병합표 청크 폭증 보정

- 관련 PR: [#23 fix: prevent merged table chunk explosion](https://github.com/aromgu/sprint_mid_project_team1/pull/23)
- 주요 코드: `src/chunking/split_text.py`
- 청킹 전략: `naive_langchain_recursive_cl100k_base_512_102_v3`

### 기존 문제

표시용 Markdown에는 `[병합 ... 계속: ... 참조]` 문구가 반복됩니다. 이를 그대로
청킹하면 같은 내용이 여러 번 색인되고, 긴 표 헤더가 매 청크에 반복되면서
비정상적으로 많은 청크가 만들어졌습니다.

### 변경 내용

- 병합표는 표시용 Markdown을 보존하되, 검색·청킹에는 중복을 제거한
  `retrieval_text`를 한 열짜리 GFM Markdown으로 변환해 사용합니다.
- 긴 헤더 뒤에 실제 행을 담을 토큰 여유가 너무 적으면 표 전체를 안전한
  한 열 Markdown으로 평탄화합니다.
- 병합표 변환 여부를 `merged_table_flattened_to_gfm` 품질 플래그로
  추적합니다.
- 임베딩 전에 파일 SHA·청크 수·문서 수·토큰 수를 확인할 수 있도록 v3 입력
  계약을 추가했습니다.

### 최종 청킹 기준과 결과

| 항목 | 값 |
| --- | ---: |
| 최대 청크 크기 | 512토큰 |
| 이전 청크와 중복 | 102토큰 |
| 토크나이저 | `cl100k_base` |
| 대상 문서 | 98개 |
| 최종 청크 | 31,627개 |
| 전체 토큰 | 10,414,025 |
| 표 하나의 최대 청크 | 29개 |
| 품질 게이트 | 전체 통과 |

원문 청크 파일의 고정 SHA-256은
`8d5107140ff20c5f78fa3b3a88c06a2149a1a31397a22e8fb1ca6cd32f3f7c09`입니다.

## 3. 사업 메타데이터 보강

- 관련 PR: [#24 feat: add chunk metadata enrichment](https://github.com/aromgu/sprint_mid_project_team1/pull/24)
- 주요 코드: `src/chunking/enrich_metadata.py`
- 실행 코드: `scripts/run_metadata_enrichment.py`

### 보강에 사용한 자료

1. `data_list` CSV에서 정리한 사업 메타데이터
2. 팀 검토를 반영한 사업 필드 교정 CSV
3. 98개 문서의 사업요약 원문 대조 결과
4. SHA-256 중복 검사에서 확인한 대표 파일과 중복 파일 별칭

### 청크에 추가한 주요 값

- 사업명: `project_name`
- 발주기관: `issuer`
- 공고번호·차수: `notice_number`, `notice_round`
- 사업금액과 상태: `project_amount_won`, `project_amount_status`
- 공고일·입찰기간: `published_at`, `bid_start_at`, `bid_end_at`
- 사업요약: `project_summary`
- 검토 상태와 교정 근거
- 대표 문서에 연결되는 중복 파일 별칭: `filename_aliases`

사업요약은 원문 대조 결과가 `pass`이면 기존 요약을 사용하고, `revise`이면
수정 요약을 사용합니다. 검토 상태는 청크에 함께 기록해 결과의 출처를 확인할
수 있게 했습니다.

> 사업요약의 `pass`·`revise`는 Codex가 원문과 대조한 결과입니다. 현재
> 파이프라인에서 사용할 수 있도록 반영했지만, 사람의 최종 승인과 같은 의미는
> 아닙니다. 팀에서는 특히 수정 19건을 표본 확인하는 것을 권장합니다.

### 전체 보강 결과

| 항목 | 결과 |
| --- | ---: |
| 전체 문서 | 98개 |
| 전체 청크 | 31,627개 |
| 메타데이터 연결 문서 | 98개 |
| 미연결 문서 | 0개 |
| 사업 필드 교정 문서 | 19개 |
| 사업요약 원문 대조 `pass` | 79개 |
| 사업요약 수정본 적용 `revise` | 19개 |
| 중복 파일 별칭이 있는 대표 문서 | 2개 |
| 청크 ID·본문·토큰 변경 | 0개 |

보강 결과 gzip은 실행 시각을 제외한 결정적 방식으로 생성합니다. 따라서 같은
입력으로 다시 실행하면 같은 파일 SHA가 만들어집니다.

현재 보강 결과의 SHA-256은
`4c77826f4705f8df70dfa15d180312ec133d624ab25ace85cfd32f0c9f8f9194`입니다.

## Git에 포함되는 것과 포함되지 않는 것

### Git에 포함

- 전처리·청킹·메타데이터 보강 코드
- 자동 테스트
- 이 인계 문서

### Git에 포함하지 않음

- 원본 HWP·PDF
- `data_list` 및 검토용 CSV
- 전처리·청킹 결과 JSONL
- 보강 결과 gzip
- 임베딩 결과와 Vector DB
- API 키와 기타 비밀값

실제 데이터와 결과는 private 저장소 또는 GCP에서 관리합니다.

## GCP 인계 전 확인 사항

임베딩 담당자는 **메타데이터가 없는 v3 파일이 아니라 보강된 v1 파일**을
사용해야 합니다.

권장 GCP 파일명:

```text
/home/data/advanced/chunks/chunks_naive_rcts_v3_metadata_v1.jsonl.gz
/home/data/reports/metadata_enrichment_report_v1.json
```

GCP 임베딩 전에 다음 작업이 남아 있습니다.

1. PR #24를 팀원 1명 이상 검토·승인 후 `naive`에 머지합니다.
2. 보강 결과 SHA를 `src/embeddings/build_embeddings.py`의 승인된 입력 계약에
   등록합니다.
3. Chroma에 저장할 사업 메타데이터 필드와 중복 파일 별칭 필드를 최종
   확인합니다.
4. GCP 환경변수에 `OPENAI_API_KEY`를 설정합니다.
5. 먼저 100청크 smoke test를 실행해 1,536차원 임베딩, 메타데이터, Chroma
   저장을 확인합니다.
6. smoke test 통과 후 31,627청크 전체를 임베딩하고 공용 Vector DB를
   저장합니다.

## 팀원이 지금 확인할 내용

- PR #24에서 메타데이터 필드와 검증 기준이 충분한지 확인
- 사업요약 수정 19건을 현재 버전으로 사용할지 확인
- GCP의 청크·보고서·Vector DB 저장 경로 확정
- 공용 Vector DB를 GCS 파일로 공유할지, GCP 서비스에서 직접 제공할지 결정

임베딩은 위 항목이 합의되고 GCP 환경이 준비되기 전까지 실행하지 않습니다.
