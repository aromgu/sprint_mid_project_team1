# 테스트 실행시간 최소화 정책

## 목표

모든 테스트와 평가 작업은 정확성·격리·재현성을 유지하는 범위에서 총 경과시간을
최소화한다. 독립 작업은 병렬 실행하고, 완료 결과는 재사용하며, 유료 API 호출은
필요한 범위만 실행한다.

## 기본 실행 순서

1. 변경 파일과 직접 관련된 테스트만 먼저 실행한다.
2. lint/build처럼 서로 독립적인 검증은 동시에 실행한다.
3. 관련 테스트가 통과한 뒤 전체 회귀 테스트를 한 번 실행한다.
4. 실제 API smoke는 대표 1건으로 시작하고 성공 후에만 전체 batch를 실행한다.
5. 실패한 항목만 retry하고 이미 성공한 항목은 `--resume`으로 건너뛴다.

## 저장소 기본 병렬값

| 작업 | 기본값 | 이유 |
|---|---:|---|
| Golden 답변 생성 | `--max-workers 4` | OpenAI rate limit과 속도의 균형 |
| 공식 RAGAS | `--max-workers 8 --batch-size 16` | 질문별 평가가 독립적 |
| Retrieval 평가 | 현재 직렬 | local Chroma query가 빠르고 측정 latency 왜곡 방지 |
| Playwright | test file 병렬, 기본 worker 사용 | browser context가 테스트별로 격리됨 |
| pytest 49개 | 현재 직렬 | 약 12초로 짧아 process startup이 이득보다 큼 |
| 실제 OpenAI browser E2E | 직렬·opt-in | 비용과 session 흐름 검증 필요 |

worker 수는 고정된 목표가 아니다. 429, timeout, memory pressure가 없고 처리량이
증가할 때만 4→6→8 순으로 올린다. 재시도가 증가하면 즉시 낮춘다.

## 권장 명령

```bash
# 빠른 관련 Python 테스트
uv run pytest -q tests/test_main_advanced_rag.py tests/test_backend_mvp.py

# 전체 Python 회귀는 마지막에 한 번
uv run pytest -q

# mock 기반 UI 회귀
cd frontend && npm run test:ui

# Top-10 Golden 답변: 독립 session 4개 병렬
uv run python -m scripts.main_rag.run_answers \
  --top-k 10 --max-workers 4 --output reports/main_advanced/answers_top10.jsonl

# RAGAS: 8개 worker, 16개 batch, 중단 결과 재사용
RAGAS_DO_NOT_TRACK=true uv run python -m scripts.main_rag.evaluate_ragas \
  --resume --batch-size 16 --max-workers 8 --max-retries 1 --timeout 60
```

## 새 테스트·평가 코드 요건

- 독립 항목 batch는 bounded concurrency를 제공한다.
- 장시간 작업은 각 항목 완료 즉시 결과를 flush한다.
- `--resume` 또는 동등한 checkpoint 기능을 제공한다.
- 입력·model·top-k가 달라지면 기존 결과를 잘못 resume하지 않도록 별도 output이나
  실행 fingerprint를 사용한다.
- live API 테스트는 환경변수로 명시적으로 활성화한다.
- 공유 manifest, Chroma collection, screenshot baseline을 쓰는 mutation은 격리된
  임시 디렉터리 또는 직렬 실행을 사용한다.
- 실행 보고에는 항목 수, worker 수, elapsed time, retry/error 수를 남긴다.

## 병렬화하지 않는 경우

- 동일 conversation/session의 순서를 검증할 때
- 동일 파일이나 Chroma collection을 동시에 변경할 때
- screenshot baseline을 갱신할 때
- provider quota가 낮거나 429 재시도로 처리량이 오히려 감소할 때
- 병렬 실행으로 latency benchmark 자체가 왜곡될 때
