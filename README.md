# sprint_mid_project_team1
mid project with RAG


## Settings
해당 repo를 clone 한 후에, repo directory에서  ```uv sync``` 와 ```uvx prek install```  터미널에 실행.

### 다른 컴퓨터에서 clone 후 바로 실행

raw PDF와 바이너리 검색 인덱스는 Git LFS로 관리한다. 먼저 Git LFS를 설치한 뒤
파일을 내려받아야 한다.

```bash
git clone https://github.com/hyojin33kim/RFP.ai.git
cd RFP.ai
git lfs install
git lfs pull
uv sync
cp .env.example .env
```

`.env`에 실제 API 키를 설정한 뒤 백엔드와 프런트엔드를 실행한다.

```bash
uv run uvicorn backend.main:app --reload
```

별도 터미널:

```bash
cd frontend
npm ci
npm run dev
```

`data/processed`, `data/manifests`, `data/indexes`가 함께 배포되므로 clone 직후
PDF ingestion과 인덱스 재생성 없이 검색·Workspace 기능을 사용할 수 있다.

## RFP PDF ingestion

검증 대상은 `RAG_검증_샘플_파일명_정리_2.csv`의 9개 문서이며, 원본 PDF는
`data/raw` 아래에서 읽는다. 원본 PDF는 수정하지 않는다.

```bash
uv sync
python -m scripts.run_ingestion
```

설정은 `configs/ingestion.yaml`에서 변경할 수 있다.

실행 결과:

- `data/manifests/documents.json`: CSV와 실제 PDF를 연결한 문서 manifest
- `data/processed/pdf_diagnostics.json`: 문서별 추출 품질과 OCR 필요 페이지
- `data/processed/pages.jsonl`: 페이지·블록·표 행을 보존한 파싱 결과
- `data/processed/chunks.jsonl`: 검색 및 Golden set에서 사용할 구조화 청크

테스트:

```bash
python -m pytest
```

## RFP retrieval

실제 검색 입력은 `data/processed/chunks.jsonl`이다. 검색 설정은
`configs/search.yaml`에서 관리한다. 기본 검색기는 Hybrid RRF이며 CLI에서
`--retriever`를 생략하면 `pipeline.retriever` 설정을 사용한다.

최초 인덱스 생성:

```bash
python -m scripts.build_search_indexes
```

BM25만 먼저 생성할 수도 있다.

```bash
python -m scripts.build_search_indexes --bm25-only
```

단일 질의 검색:

```bash
python -m scripts.run_search \
  --query "전자기록관 DB 정비 및 마이그레이션 요구사항" \
  --retriever hybrid \
  --top-k 5
```

`--retriever`는 `bm25`, `dense`, `hybrid`, `reranked` 중 하나다. `--document-id`와
`--content-type`을 반복 지정해 검색 범위를 제한할 수 있다.

JSONL 질의 파일을 일괄 검색하고 결과를 저장할 수 있다.

```bash
python -m scripts.run_search \
  --queries questions.jsonl \
  --retriever hybrid \
  --output reports/search_results.jsonl
```

Golden set이 준비되면 다음 명령으로 검색기를 비교한다.

```bash
python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all
```

Golden set 입력 형식은 `data/eval/GOLDEN_SET_INPUT_SCHEMA.md`에 정의되어 있다.

Cross-encoder 재정렬과 인접 청크 확장:

```bash
python -m scripts.run_search \
  --query "SFR-007 예약 시스템 기능" \
  --config configs/search/hybrid_reranked.yaml \
  --neighbor-window 1 \
  --top-k 5
```

`reranked`는 로컬 `Dongjin-kr/ko-reranker`를 사용한다. CPU에서는 일반
Hybrid 검색보다 느리므로 Golden set에서 효과를 검증한 뒤 기본 활성화 여부를
결정한다. `--neighbor-window 1`은 최종 검색 청크의 앞뒤 청크를 답변용 문맥에
추가하며 검색 순위 자체는 바꾸지 않는다.

구성요소별 프리셋도 제공한다.

```bash
python -m scripts.run_search --query "SFR-007 예약 기능" \
  --config configs/search/bm25.yaml
python -m scripts.run_search --query "SFR-007 예약 기능" \
  --config configs/search/hybrid_reranked.yaml
```

- BM25 tokenizer: `korean_ngram`, `regex`, `whitespace` (`morpheme`는 분석기 어댑터 설치 후 사용)
- Dense model: `dense.model`, `query_prefix`, `passage_prefix`로 교체
- Fusion: `rrf` 또는 `weighted_score`
- Reranker: `reranker.enabled`; 기본값은 `false`
- 인접 문맥: `context_expansion.enabled`, `window`

`configs/search/`에는 BM25, Dense, KoE5, Hybrid RRF, weighted-score,
reranked 프리셋이 있다. KoE5와 reranker 프리셋은 CPU에서 느릴 수 있다.
평가의 `--retriever all`은 해당 설정에서 활성화된 검색기만 비교하므로,
reranker까지 비교할 때는 `--config configs/search/hybrid_reranked.yaml`을 함께 쓴다.

## OpenAI RAG answer and API

답변 생성 기본 모델은 저비용 `gpt-5-nano`이며 Responses API의 구조화 출력을
사용한다. 모델과 reasoning 수준은 `configs/generation.yaml`에서 교체할 수 있다.
API 키는 저장소에 커밋하지 않고 `.env` 또는 `.env.local`의
`OPENAI_API_KEY`로만 전달한다.

```bash
uv run python -m scripts.run_answer \
  --query "SFR-007 예약 시스템에서 예약기간은 누가 설정합니까?"
```

응답에는 답변 가능 여부, 문서·페이지·요구사항·청크 인용, 검색/생성 시간,
입출력 토큰과 설정 단가 기반 예상 비용이 포함된다. 존재하지 않는 출처 라벨은
제거되며 유효한 인용이 없으면 답변을 강제로 유보한다.

FastAPI 실행:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- `GET /health`: 청크 수, 기본 검색기, API 키 설정 여부
- `POST /search`: 검색 결과만 반환
- `POST /answer`: 검색 후 근거 기반 OpenAI 답변 반환
- API 문서: `http://127.0.0.1:8000/docs`

Handover v3 MVP API와 기본 화면을 함께 실행하려면 두 터미널에서 실행한다.

```bash
# backend
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# frontend
cd frontend
npm run dev
```

backend는 기본적으로 시작 시 Dense 모델을 미리 로드한다. 따라서 서버 시작은
약간 느려지지만 첫 검색 요청의 모델 로딩 지연을 줄인다. 끄려면
`RAG_PREWARM_DENSE=false`를 설정한다. 현재 MVP는 OpenAI context를 12,000자,
최대 출력을 700토큰으로 제한해 응답 latency와 비용을 줄인다.

기본 화면은 `frontend/src/`에 있으며, 실제 문서 목록·Overview·위험·요구사항·AI
질문을 현재 RAG API에 연결한다. 상세 구현 순서는
`HANDOVER_V3_MVP_IMPLEMENTATION_PROCEDURE.md`를 따른다.

검색·생성 latency 측정:

```bash
uv run python -m scripts.profile_latency
uv run python -m scripts.profile_latency --live  # OpenAI 호출 1회 포함
```

API 응답에는 `X-Process-Time-ms` 헤더가 포함되고 backend 터미널에도 요청별
처리 시간이 기록된다. 첫 Dense 모델 로딩, 반복 검색, OpenAI 생성 시간을
분리해서 확인할 수 있다.

## OCR augmentation

시스템 Tesseract를 설치할 권한이 없는 환경에서는 사용자 공간 runtime을 준비한다.

```bash
bash scripts/bootstrap_tesseract_user.sh
python -m scripts.run_ocr
python -m scripts.run_ingestion
python -m scripts.build_search_indexes
```

OCR 결과는 `data/processed/ocr_pages.jsonl`에 저장된다. ingestion은 이 파일이
있으면 페이지 메타데이터와 별도 `ocr_*` 검색 청크를 자동으로 추가한다.
시스템 구성도 OCR은 관계 구조를 완전히 보존하지 못하므로
`multimodal_review_required` 표시를 확인해야 한다.

> `scripts/run_rag.py`, `src/loader/load_documents.py`, `src/dataset.py`는 기존
> 스마트워치·사내규정 데모 코드다. 실제 RFP 검색은 `scripts/run_search.py`와
> `src/search/` 모듈을 사용한다.




## Project Structure

```text
rag-project/
├─ README.md
├─ .gitignore
├─ .env.example
├─ src/
│  ├─ loader/
│  │  └─ load_documents.py
│  ├─ preprocessing/
│  │  └─ clean_text.py
│  ├─ chunking/
│  │  └─ split_text.py
│  ├─ embeddings/
│  │  └─ build_embeddings.py
│  ├─ retrieval/
│  │  ├─ retriever.py
│  │  └─ reranker.py
│  ├─ generation/
│  │  └─ generate_answer.py
│  └─ evaluation/
│     └─ eval_rag.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ eval/
├─ scripts/
│  ├─ run_indexing.py
│  ├─ run_rag.py
│  └─ run_eval.py
├─ notebooks/
├─ tests/
├─ docs/
└─ .github/
   ├─ ISSUE_TEMPLATE/
   └─ pull_request_template.md
```
