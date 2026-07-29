# Main Advanced RAG P1 상태

상세 수행 내용, 의사결정 근거, 결과 해석 및 남은 일은
[`P1_REPORT.md`](./P1_REPORT.md)를 참조한다.

## 완료

- Main Advanced retrieval adapter 및 Golden v3 연결
- 95개 answerable 질문 Dense retrieval 평가
- 질문별 독립 세션과 `--resume`, `--max-workers`를 지원하는 answer batch
- 실제 검색 chunk만 허용하는 citation 검증
- `gpt-5-nano`로 Golden 질문 104개 답변 생성 완료 (104/104, 오류 0)
- 결정론적 Golden 평가 완료
- 공식 RAGAS 평가 완료 (`gpt-4o-mini`, 104/104, null 0)
- 별도 RAGAS-compatible LLM Judge 완료 (`gpt-4o-mini`, 104/104)
- 전체 테스트 47개 통과

## 결과

Retrieval (answerable 95개):

- Hit@1: 0.6105
- Hit@3: 0.7789
- Hit@5: 0.8842
- Fact coverage@5: 0.6877
- 평균 latency: 294.81 ms
- 최대 latency: 2495.65 ms
- 빈 검색: 0
- 타 문서 혼입: 0

Golden deterministic evaluation (104개):

- Answerability accuracy: 0.7885
- Required fact coverage: 0.6079
- Numeric recall: 0.7003
- Numeric precision: 0.5893
- Ground-truth token coverage: 0.3761

공식 RAGAS (104개 유효값):

- Faithfulness: 0.8142
- Answer relevancy: 0.3364

별도 LLM Judge (104개):

- Faithfulness: 0.8894
- Answer relevancy: 0.9091

RAGAS answer relevancy는 답변으로부터 질문을 재생성한 뒤 embedding 유사도를
측정하므로, 답변 불가/근거 부족 응답에 0이 다수 포함되어 별도 Judge보다 낮다.

## 모델 및 보존 결과

- 답변 생성: `gpt-5-nano`
- RAGAS/Judge: `gpt-4o-mini` (RAGAS가 `gpt-5-nano`에서 지원되지 않는 temperature를 사용)
- Gemini 3.5 Flash partial 23개와 Flash-Lite partial 16개는 비교용 별도 파일로 보존

## P2 데모 모델 계획

- P1 평가 결과는 `gpt-5-nano`로 고정해 재현성을 유지한다.
- P2 데모 runtime은 낮은 latency를 위해 `gemini-3.5-flash-lite`로 전환한다.
- 데모 전 API key/model access/quota를 확인하고, 실패 시 `gpt-5-nano`로 fallback한다.

## 재현 명령

```bash
uv run python -m scripts.main_rag.run_answers --resume --max-workers 4
uv run python -m scripts.main_rag.evaluate_answers
RAGAS_DO_NOT_TRACK=true uv run python -m scripts.main_rag.evaluate_ragas --resume --batch-size 16 --max-workers 8 --max-retries 1 --timeout 60
uv run python -m scripts.main_rag.evaluate_llm_judge --resume
```
