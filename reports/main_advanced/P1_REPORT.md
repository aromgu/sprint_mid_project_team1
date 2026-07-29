# Main Advanced RAG P1 수행 보고서

## 1. 결론

P1의 목표였던 **Main Advanced RAG를 현재 Golden v3 평가 시스템에 연결하고,
retrieval·답변 생성·RAGAS·LLM Judge를 전체 실행하는 작업**을 완료했다.

- Retrieval 평가는 answerable 질문 95개를 대상으로 완료했다.
- Golden 질문 104개의 답변을 `gpt-5-nano`로 모두 생성했다.
- 생성 오류는 0건이며, 잘못된 chunk citation도 0건이다.
- 결정론적 Golden 평가, 공식 RAGAS, 별도 LLM Judge를 104개 전체에 실행했다.
- 회귀 테스트 47개가 통과했다.

다만 UI 연결 전에 개선을 검토할 부분이 있다. 검색 Hit@5는 0.8842로 양호하지만
필수 사실 coverage와 hard/colloquial 질문 성능은 낮고, `gpt-5-nano` 생성 latency는
질문당 약 27~29초 수준이었다. 따라서 P1 실행 자체는 완료됐지만, P2 데모 품질을
위해서는 retrieval/generation 최적화가 권장된다.

---

## 2. P1에서 무엇을 했는가

### 2.1 Retrieval 평가 연결

Main Advanced Retriever의 결과를 현재 Golden v3 evaluator가 읽을 수 있는 형식으로
변환했다. 변환 시 chunk ID, document ID, page, score, rank를 보존하고, Golden의
원본 파일명과 평가용 document ID를 `source_map.json`으로 연결했다.

관련 코드:

- `src/main_rag/evaluation.py`: 평가용 retrieval adapter
- `scripts/main_rag/evaluate_retrieval.py`: Golden retrieval 평가 실행기
- `src/main_rag/retrieval/advanced_retriever.py`: Chroma dense retrieval

### 2.2 Golden 답변 batch 생성

104개 Golden 질문 각각에 대해 다음 흐름을 실행했다.

```text
Golden 질문
  → 질문 전용 새 session
  → query rewrite
  → 문서 ID로 제한한 Advanced dense retrieval
  → structured answer 생성
  → evidence/citation 검증
  → answers.jsonl 저장
```

질문마다 session을 새로 생성해 이전 질문의 `previous_response_id`, 대화 history,
요약이 다음 평가 질문에 섞이지 않도록 했다. 중단 시 성공한 질문 이후부터 다시
시작할 수 있도록 `--resume`을 구현했고, 전체 실행 시간을 줄이기 위해
`--max-workers` 기반 병렬 생성도 추가했다.

관련 코드:

- `scripts/main_rag/run_answers.py`: resumable/parallel answer batch
- `src/main_rag/service.py`: retrieval과 generation 연결 및 citation 검증
- `src/main_rag/generation/generate_answer.py`: OpenAI structured generation
- `src/main_rag/generation/gemini_session.py`: Gemini structured generation
- `configs/main_advanced_rag.yaml`: provider/model/path/retrieval 설정

### 2.3 Citation 검증

모델이 반환한 evidence의 `chunk_id`가 실제 해당 질문에서 검색된 chunk 목록에 있는지
검증했다. 검색 결과에 없는 chunk는 citation으로 내보내지 않고
`rejected_evidence`에 분리했다. 최종 104개 답변에서 거부된 evidence는 0건이었다.

### 2.4 결정론적 Golden 평가

LLM 호출 없이 다음 항목을 계산했다.

- answerability 정확도
- required fact coverage
- 숫자 recall/precision
- ground-truth token coverage
- 난이도 및 query type별 breakdown

관련 코드:

- `scripts/main_rag/evaluate_answers.py`

### 2.5 공식 RAGAS 평가

생성 답변과 citation/retrieved context를 RAGAS 입력 형식으로 변환하고 다음 metric을
실행했다.

- Faithfulness
- Answer relevancy

RAGAS 실행 중 timeout이 발생한 항목은 결과를 즉시 저장한 뒤 `--retry-null`로 해당
metric만 재시도했다. 최종적으로 104개 모두 null 없는 결과를 만들었다. RAGAS의
telemetry 요청은 평가와 무관하고 네트워크 지연을 만들 수 있어 재현 명령에서는
`RAGAS_DO_NOT_TRACK=true`를 사용한다.

관련 코드:

- `scripts/main_rag/evaluate_ragas.py`

### 2.6 별도 LLM Judge 평가

공식 RAGAS와 별도로 질문·답변·검색 근거를 한 번에 입력하는 structured LLM Judge를
실행했다. 이는 RAGAS package 결과가 아니라, faithfulness와 answer relevancy를
0~1로 직접 판정하는 보조 평가다.

관련 코드:

- `scripts/main_rag/evaluate_llm_judge.py`

---

## 3. 왜 이렇게 했는가

### 3.1 Retrieval과 generation을 분리 평가한 이유

최종 답변이 틀렸을 때 원인이 검색 실패인지, 검색된 근거를 모델이 잘못 사용한
것인지 구분해야 한다. 따라서 먼저 retrieval 자체를 평가하고, 이후 동일 retrieval
결과를 사용한 답변 품질과 faithfulness를 평가했다.

### 3.2 질문마다 새 session을 사용한 이유

Golden 질문은 서로 독립적인 평가 sample이다. session을 공유하면 앞 질문의 내용이
뒷 질문의 query rewrite 또는 답변에 영향을 주어 평가 결과가 오염된다.

### 3.3 문서 ID filter를 사용한 이유

Golden 항목은 정답 문서가 지정되어 있다. P1은 Advanced pipeline의 문서 내부 검색과
생성 품질을 검증하는 단계이므로, 해당 문서로 검색 범위를 제한했다. 다른 문서를
검색하는 오류도 별도로 확인했으며 0건이었다.

### 3.4 생성과 평가 모델을 분리한 이유

최종 답변은 사용자 결정에 따라 `gpt-5-nano`로 생성했다. 공식 RAGAS는 내부적으로
temperature 0.01/0.3을 지정하는데 `gpt-5-nano`는 기본 temperature 외 값을 지원하지
않아 metric 호출이 실패했다. 따라서 RAGAS와 LLM Judge에는 호환되는
`gpt-4o-mini`를 사용했다. 생성 모델과 Judge 모델을 분리하면 동일 모델의 자기평가
편향도 줄일 수 있다.

### 3.5 Gemini 전체 생성을 중단한 이유

- `gemini-3.5-flash`: 무료 tier 일일 model/project quota로 23개까지만 생성
- `gemini-3.5-flash-lite`: 동일 quota 제약으로 16개까지만 생성
- `gemini-2.5-flash-lite`: 신규 사용자에게 제공되지 않아 API 404 발생
- Gemini billing 활성화 최소 결제액이 25,000원이어서 사용하지 않기로 결정

따라서 Gemini partial 결과를 덮어쓰지 않고 별도 보존한 뒤, 104개 전체 평가는
`gpt-5-nano`로 새로 생성했다.

---

## 4. 어떻게 실행했는가

### 4.1 Retrieval 평가

```bash
uv run python -m scripts.main_rag.evaluate_retrieval
```

### 4.2 답변 생성

```bash
uv run python -m scripts.main_rag.run_answers --resume --max-workers 4
```

동시성 4로 질문별 독립 session을 병렬 처리했다. 각 완료 결과는 즉시 JSONL 한 줄로
flush되므로 실행이 중단돼도 완료 결과가 보존된다.

### 4.3 결정론적 평가

```bash
uv run python -m scripts.main_rag.evaluate_answers
```

### 4.4 공식 RAGAS

```bash
RAGAS_DO_NOT_TRACK=true uv run python -m scripts.main_rag.evaluate_ragas \
  --resume --batch-size 16 --max-workers 8 --max-retries 1 --timeout 60
```

null 항목이 있으면 metric별로 재실행한다.

```bash
RAGAS_DO_NOT_TRACK=true uv run python -m scripts.main_rag.evaluate_ragas \
  --resume --retry-null --metrics faithfulness

RAGAS_DO_NOT_TRACK=true uv run python -m scripts.main_rag.evaluate_ragas \
  --resume --retry-null --metrics answer_relevancy
```

### 4.5 별도 LLM Judge

```bash
uv run python -m scripts.main_rag.evaluate_llm_judge --resume
```

### 4.6 회귀 테스트

```bash
uv run pytest -q
```

---

## 5. 결과

### 5.1 Retrieval 결과

대상은 answerable Golden 질문 95개다.

| 지표 | 결과 |
|---|---:|
| Hit@1 | 0.6105 |
| Hit@3 | 0.7789 |
| Hit@5 | 0.8842 |
| MRR@10 | 0.7123 |
| Fact coverage@5 | 0.6877 |
| Full section hit@5 | 0.4421 |
| 평균 검색 latency | 294.81 ms |
| 최대 검색 latency | 2495.65 ms |
| 빈 검색 | 0 |
| 잘못된 문서 검색 | 0 |

해석:

- 정답 근거가 상위 5개 안에 포함되는 비율은 88.42%로 데모 가능한 수준이다.
- 그러나 필요한 사실 전체가 같은 검색 결과에 충분히 포함되는 정도는 더 낮다.
- 특히 full section hit@5가 44.21%이므로 multi-fact 답변의 누락 가능성이 있다.

### 5.2 `gpt-5-nano` 답변 생성 결과

| 항목 | 결과 |
|---|---:|
| 전체 질문 | 104 |
| 성공 | 104 |
| API/생성 오류 | 0 |
| Answerable 판정 | 83 |
| Unanswerable 판정 | 21 |
| 거부된 잘못된 evidence | 0 |

### 5.3 결정론적 Golden 평가

| 지표 | 전체 결과 |
|---|---:|
| Answerability accuracy | 0.7885 |
| Required fact coverage | 0.6079 |
| Numeric recall | 0.7003 |
| Numeric precision | 0.5893 |
| Ground-truth token coverage | 0.3761 |

난이도별 주요 결과:

| 난이도 | 질문 수 | Answerability | Fact coverage |
|---|---:|---:|---:|
| Easy | 27 | 0.8148 | 0.6600 |
| Medium | 46 | 0.8261 | 0.6813 |
| Hard | 31 | 0.7097 | 0.4667 |

Query type별 주요 결과:

| 유형 | 질문 수 | Answerability | Fact coverage |
|---|---:|---:|---:|
| Keyword | 32 | 0.7813 | 0.5893 |
| Semantic | 17 | 0.8824 | 0.6905 |
| Multihop | 48 | 0.7917 | 0.5975 |
| Colloquial | 7 | 0.5714 | 0.5833 |

### 5.4 공식 RAGAS

| 지표 | 결과 | 유효 표본 |
|---|---:|---:|
| Faithfulness | 0.8142 | 104/104 |
| Answer relevancy | 0.3364 | 104/104 |

RAGAS answer relevancy는 답변으로부터 질문을 재생성하고 embedding 유사도를 계산한다.
현재 데이터에는 의도적으로 answerable=false인 질문과 근거 부족 응답이 포함되며,
이 응답들이 0점에 가까운 값을 많이 받아 평균이 낮다. 따라서 이 점수만으로 실제
질문 대응 품질을 판단하지 않고 별도 Judge와 함께 본다.

### 5.5 별도 LLM Judge

| 지표 | 결과 | 유효 표본 |
|---|---:|---:|
| Faithfulness | 0.8894 | 104/104 |
| Answer relevancy | 0.9091 | 104/104 |

두 평가 모두 faithfulness는 비교적 높다. Answer relevancy의 큰 차이는 평가 방식이
다르기 때문이며, 별도 Judge는 적절한 답변 거절도 관련성이 있는 것으로 평가한다.

### 5.6 모델 비교용 partial 결과

Gemini 결과는 전체 평가가 아니며 같은 질문 ID에 대해서만 비교해야 한다.

| 동일 표본 | 모델 | Answerability | Fact coverage | Numeric recall | Numeric precision | 평균 생성시간 |
|---|---|---:|---:|---:|---:|---:|
| 23개 | Gemini 3.5 Flash | 0.913 | 0.569 | 0.800 | 0.893 | 6.2초 |
| 23개 | gpt-5-nano | 0.826 | 0.699 | 0.813 | 0.832 | 27.2초 |
| 16개 | Gemini 3.5 Flash-Lite | 0.938 | 0.594 | 0.872 | 0.558 | 3.4초 |
| 16개 | gpt-5-nano | 0.875 | 0.661 | 0.831 | 0.821 | 29.1초 |

Gemini는 훨씬 빠르고 짧은 답변을 만들었으며, `gpt-5-nano`는 필수 사실 coverage와
인용 수가 더 높았다. Flash-Lite는 숫자 precision이 낮아 숫자 답변에 주의가 필요하다.

### 5.7 테스트

```text
47 passed, 9 warnings
```

warning은 기존 LangChain/RAGAS deprecated import에 관한 것으로 테스트 실패는 아니다.

---

## 6. 산출물

| 산출물 | 위치 |
|---|---|
| Retrieval 요약 | `reports/main_advanced/retrieval_summary.json` |
| Retrieval 상세 | `reports/main_advanced/retrieval_details.jsonl` |
| 104개 최종 답변 | `reports/main_advanced/answers.jsonl` |
| 결정론적 평가 | `reports/main_advanced/answer_summary.json` |
| 공식 RAGAS 요약 | `reports/main_advanced/ragas.json` |
| 공식 RAGAS 상세 | `reports/main_advanced/ragas_details.jsonl` |
| LLM Judge 요약 | `reports/main_advanced/llm_judge.json` |
| LLM Judge 상세 | `reports/main_advanced/llm_judge_details.jsonl` |
| Gemini Flash partial | `reports/main_advanced/answers_gemini_3_5_flash_partial.jsonl` |
| Gemini Flash-Lite partial | `reports/main_advanced/answers_gemini_3_5_flash_lite_partial.jsonl` |

---

## 7. 남은 일

### 7.1 P1 품질 개선 권장사항

P1 실행 및 결과 생성은 완료됐지만 다음 개선은 P2 전에 검토할 가치가 있다.

1. **Hard 질문 fact coverage 개선**
   - Hard fact coverage가 0.4667로 가장 낮다.
   - 실패 질문의 `retrieval_details.jsonl`을 기준으로 chunk 경계와 top-k를 점검한다.
   - multi-fact 질문은 top-k 5와 top-k 10의 coverage 차이를 비교한다.

2. **Colloquial 질문 처리 개선**
   - Colloquial answerability accuracy가 0.5714다.
   - query rewrite 결과와 원문 질문을 함께 검색하는 방식의 효과를 별도 실험한다.

3. **숫자 precision 개선**
   - 전체 numeric precision이 0.5893이다.
   - 답변의 숫자마다 citation quote에 동일 숫자가 존재하는지 후처리 검증을 고려한다.

4. **생성 latency 개선**
   - `gpt-5-nano`는 partial 공통 표본에서 평균 27~29초였다.
   - context 길이, rewrite 호출, reasoning/output token 사용량을 profiling한다.
   - Gemini quota/billing 정책이 해결되면 Flash를 latency 대안으로 재검토할 수 있다.

5. **비용 집계 구현**
   - token usage는 저장하지만 현재 `estimated_cost_usd`는 계산하지 않아 0으로 표시된다.
   - 모델별 단가를 설정 파일로 관리하고 실제 추정 비용을 계산해야 한다.

6. **RAGAS 최신 API 마이그레이션**
   - 현재 RAGAS/LangChain wrapper에 deprecation warning이 있다.
   - `ragas.metrics.collections`, `llm_factory`, 최신 embedding provider로 옮긴다.

7. **평가 결과 심층 오류 분석**
   - answerability 오판 22건과 낮은 fact coverage 질문을 검색 실패/생성 누락/Golden
     표현 불일치로 분류한다.
   - RAGAS relevancy 0점 사례가 적절한 거절인지 실제 무관 답변인지 표본 검수한다.

### 7.2 다음 단계 P2

`MAIN_ADVANCED_RAG_PRIORITY_TODO.md` 기준 다음 우선순위는 UI 연결이다.

1. **데모 generation model을 `gemini-3.5-flash-lite`로 전환**
   - P1 평가 결과의 생성 모델(`gpt-5-nano`)은 변경하지 않고 평가 재현성을 보존한다.
   - 데모 runtime 설정만 Gemini Flash-Lite로 분리한다.
   - `GEMINI_API_KEY`, 해당 프로젝트의 model access 및 요청 quota를 데모 전에 확인한다.
   - 1건 smoke test 후 답변 schema, citation, 한글 품질, 첫 응답 latency를 확인한다.
   - 무료 tier 일일 quota가 부족하면 데모용 quota/billing을 사전에 확보하거나
     `gpt-5-nano` fallback을 유지한다.
2. Main answer/evidence를 현재 UI response model로 변환하는 Q&A adapter 구현
3. backend의 `RAGClient.answer()`를 `MainAdvancedRAGService` 호출로 교체
4. `conversation_id → BidMateRAGSession` 메모리 session 구현 및 reset/정리 추가
5. Workspace 카드의 retrieval을 `AdvancedRetriever`로 연결
6. Q&A, citation, 후속 질문, reset, Workspace 카드 UI smoke test

P2에서는 DB/Redis session, multi-worker session 공유, Naive/Advanced 선택 UI는 범위에서
제외한다.

---

## 8. 최종 판단

P1의 기술적 완료 조건은 충족했다.

- **Advanced 검색이 정답 근거를 찾는가?**
  대체로 그렇다. Hit@5는 0.8842지만 multi-fact 전체 근거 coverage는 추가 개선 여지가 있다.

- **생성 답변이 검색 근거와 일치하는가?**
  대체로 그렇다. RAGAS faithfulness 0.8142, 별도 Judge 0.8894다.

- **Citation이 올바른 문서·페이지·chunk를 가리키는가?**
  그렇다. 문서 filter 오류와 검색되지 않은 chunk citation은 각각 0건이다.

- **데모 시간 내 답변이 생성되는가?**
  기능적으로 가능하지만 `gpt-5-nano` 평균 생성시간은 데모 UX 관점에서 느리다.
  P2 연결과 함께 loading UI 또는 더 빠른 generation model 검토가 필요하다.
