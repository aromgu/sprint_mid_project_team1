# sprint_mid_project_team1
mid project with RAG

## Settings
작업할 환경에서 ```pip install uv```

해당 repo를 clone 한 후에, repo directory에서  ```uv sync``` 와 ```uvx prek install```  터미널에 실행.


# RFP RAG 

컴퓨터 비전 프로젝트에 이어 진행하는 **자연어처리 · LLM 스프린트 미드 프로젝트**입니다.
RAG(Retrieval-Augmented Generation) 시스템을 구축해, 수십 페이지짜리 기업/정부 제안요청서(RFP)의
핵심 정보를 빠르게 추출·요약·질의응답 할 수 있는 서비스를 만듭니다.

## 프로젝트 배경

이 프로젝트에서는 **B2G 입찰지원 전문 컨설팅 스타트업 '입찰메이트'의 엔지니어링 팀**이라는
가상의 상황을 가정합니다.

- 나라장터 등에는 하루에도 수백 건의 RFP(제안요청서)가 올라오고, 한 건당 수십 페이지가 넘습니다.
- 컨설턴트가 이 많은 문서를 일일이 읽고 고객사에게 맞는 입찰 기회를 찾는 것은 비효율적입니다.
- 그래서 RFP의 **주요 요구 조건 / 대상 기관 / 예산 / 제출 방식** 등 핵심 정보를 빠르게 파악할 수 있는
  사내 RAG 시스템을 구축하는 것이 이 팀의 미션입니다.

실제 RFP 원본 문서 100건과 메타데이터를 바탕으로 여러 자연어처리 기법을 실험하고,
팀이 직접 선정한 지표로 성능을 평가합니다.

## 주요 기능

- **하이브리드 검색(Hybrid Retrieval)**: BM25(키워드 검색) + 벡터 검색(의미 검색)을
  `EnsembleRetriever`(RRF)로 결합하고, Cross-Encoder 리랭커로 재정렬
- **Naive RAG / Advanced RAG 두 가지 파이프라인**: 전처리·청킹 방식을 다르게 실험해 비교
- **한국어 특화 처리**: `kiwipiepy`(형태소 분석/BM25 토크나이징), `kss`(한국어 문장 분리기) 사용
- **표(table) 이중 표현 처리**: RFP 안의 표를 HTML/Markdown 형태로 함께 보존해 검색 품질 확보
- **멀티턴 Q&A**: 이전 대화 맥락을 유지하며 질문을 재작성(query rewrite)하고, 구조화된 JSON으로 답변(근거 인용 포함)
- **골든셋 기반 리트리버 검증**: Hit@k / Recall@k / MRR@k로 검색 자체 성능을 LLM 호출 없이 정량 측정
- **RAGAS 기반 정량 평가**: Faithfulness, Context Precision/Recall, Answer Relevancy 등으로 답변 품질 측정

## 전체 파이프라인

```
원본 RFP (HWP/HWPX/PDF)
    │  src/loader        : 원본 파일 탐색, 형식/크기/SHA-256 확인 
    ▼
전처리 (src/preprocessing)
    │  - clean_text.py       : Naive 경로 텍스트 정제
    │  - prepare_advanced.py : Advanced 경로. 문단/페이지 경계 보존, 표 HTML+Markdown 이중 저장
    ▼
청킹 (src/chunking)
    │  - split_text.py        : Naive, RecursiveCharacterTextSplitter 기반 512/102 토큰 청크
    │  - advanced_chunking.py : Advanced, KSS 문장 경계 기반 의미 단위 512/51 토큰 청크
    ▼
임베딩 · 인덱싱 (src/embeddings)
    │  - build_embeddings.py     : OpenAI 임베딩 → Chroma 벡터 DB (Naive)
    │  - build_advanced_index.py : OpenAI 임베딩(Dense, Chroma) + Kiwi 기반 BM25 인덱스 (Advanced)
    ▼
검색 (src/retrieval)
    │  - retriever.py : BM25 + 벡터 검색을 RRF로 앙상블
    │  - reranker.py  : Qwen3-Reranker-0.6B로 최종 재정렬
    ▼
답변 생성 (src/generation)
    │  - generate_answer.py : OpenAI Responses API(비동기), 멀티턴 세션, 구조화된 JSON 답변(근거 인용 포함)
    ▼
평가 (src/evaluation)
       - eval_retriever.py : 골든셋 기반 Hit@k / Recall@k / MRR@k로 리트리버 자체 성능 검증
       - eval_ragas.py      : RAGAS 지표로 최종 답변 품질 자동 평가
```


## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 언어 / 패키지 관리 | Python 3.11, [uv](https://docs.astral.sh/uv/) |
| LLM / 오케스트레이션 | OpenAI API(Responses API), LangChain(core/community/openai/huggingface/chroma) |
| 벡터 DB | Chroma (`langchain-chroma`, `chromadb`) |
| 키워드 검색 | `rank-bm25` |
| 한국어 NLP | `kiwipiepy`(형태소 분석), `kss`(문장 분리) |
| 리랭킹 | `sentence-transformers` (Qwen3-Reranker-0.6B, CrossEncoder) |
| 문서 파싱 | `pdfplumber`(PDF), `rhwp-python`(HWP/HWPX) |
| 토큰 계산 | `tiktoken` |
| 데이터 처리 | `pandas`, `numpy` |
| 평가 | `ragas`(답변 품질), 자체 스크립트(리트리버 Hit@k/Recall@k/MRR@k) |
| 코드 품질 | `ruff`, `pre-commit`, GitHub Actions Lint 워크플로우 |
| 테스트 | `unittest` 기반 유닛 테스트 (`tests/`) |

## 폴더 구조

```
sprint_mid_project_team1/
├─ README.md
├─ main.py                    # 멀티턴 RAG Q&A 데모 실행 진입점
├─ logging_config.py          # 로깅 설정
├─ pyproject.toml / uv.lock   # 의존성 정의
├─ .env.example               # 환경변수 템플릿
├─ .pre-commit-config.yaml    # pre-commit 훅(ruff 등)
├─ .github/workflows/lint.yaml
├─ scripts/                   # 파이프라인 단계별 실행 CLI
│  ├─ run_advanced_preprocessing.py
│  ├─ run_advanced_chunking.py
│  ├─ run_advanced_indexing.py
│  ├─ run_chunking.py
│  ├─ run_indexing.py
│  ├─ run_rag.py
│  └─ run_eval.py
├─ src/
│  ├─ loader/load_documents.py
│  ├─ preprocessing/{clean_text.py, prepare_advanced.py, table_formats.py}
│  ├─ chunking/{split_text.py, advanced_chunking.py}
│  ├─ embeddings/{build_embeddings.py, build_advanced_index.py}
│  ├─ retrieval/{retriever.py, reranker.py}
│  ├─ generation/generate_answer.py
│  └─ evaluation/{eval_ragas.py, eval_samples.json, eval_retriever.py}
└─ tests/                
```

## 설치 및 환경 설정

```bash
# 1. uv 설치 (패키지 매니저)
pip install uv

# 2. 저장소 clone 후 이동
git clone https://github.com/aromgu/sprint_mid_project_team1.git
cd sprint_mid_project_team1

# 3. 의존성 설치
uv sync

# 4. pre-commit 훅 설치 (커밋 전 자동 린트/포맷)
uvx prek install
```

`.env.example`을 복사해 `.env`를 만들고 값을 채워주세요.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 (임베딩·생성·평가에 모두 사용) |
| `EMBEDDINGS_MODEL` | 임베딩 모델명 (기본값: `text-embedding-3-small`) |
| `CHROMA_COLLECTION_NAME` | 벡터 검색에 사용할 Chroma 컬렉션 이름 (예: `ai11_policy_advanced_v2_1024`) |
| `CHROMA_PERSIST_DIRECTORY` | Chroma 벡터 DB가 저장된 경로 (예: `/home/data/chroma_advanced_v2_1024`) |
| `BM25_INDEX_PATH` | 미리 만들어둔 BM25 인덱스 pickle 파일 경로 (예: `/home/data/bm25_advanced_v2_1024/bm25_index.pkl`) |
| `OPENAI_MODEL` | 답변 생성에 사용할 LLM 모델명 (기본값: `gpt-5-mini`, `main.py`에서 읽음) |

> `CHROMA_COLLECTION_NAME` / `CHROMA_PERSIST_DIRECTORY` / `BM25_INDEX_PATH` 값은 어떤 인덱싱 버전(청크
> 크기 등)을 쓰느냐에 따라 달라집니다. `scripts/run_advanced_indexing.py`로 인덱스를 새로 만들 때 사용한
> 경로·컬렉션 이름과 반드시 맞춰주세요.

## 실행 방법

### 1) 대화형 Q&A 데모

```bash
uv run python main.py
```

`질문 >` 프롬프트에 자연어로 질문을 입력하면 됩니다. `reset`으로 대화를 초기화하고,
`exit` / `quit`으로 종료합니다.

- 답변에 사용할 LLM 모델은 `.env`의 `OPENAI_MODEL`로 바꿀 수 있습니다 (기본값 `gpt-5-mini`).
- 직접 답변 / 요약 / 근거 인용 / 신뢰도 등은 화면에 로그로 함께 출력되며, 같은 내용이
  `logs/app_YYYYMMDD_HHMMSS.log` 파일에도 저장됩니다.


## 평가 (Evaluation)

리트리버 자체 성능과 최종 답변 품질을 각각 별도로 검증합니다.

### 1) 리트리버 검증 — Hit@k / Recall@k / MRR@k

`src/evaluation/eval_retriever.py`는 골든셋으로
하이브리드 리트리버(BM25 + 벡터 + 리랭킹)의 검색 성능만 따로 검증합니다. RAGAS는 "최종 답변"을 채점하는
반면, 이 검증은 "검색이 정답 근거를 실제로 상위권에 가져오는가"만 순수하게 측정합니다.

| 지표 | 의미 |
|---|---|
| **Hit@k** | 상위 k개 검색 결과 안에 정답 청크가 하나라도 있으면 1, 없으면 0 |
| **Recall@k** | 정답 청크 중 몇 %를 상위 k개 안에서 찾아냈는가 (가장 중요하게 보는 지표) |
| **MRR@k** | 첫 정답 청크가 나온 순위의 역수 평균 (1등=1.0, 2등=0.5, 3등=0.33 …) |

- 골든셋의 질문·근거청크ID·방해청크ID를 기준으로 채점하며, 정답 청크 개념이 없는 부정거부(refuse) 케이스는
  검색 평가에서 제외합니다.
- `--stage` 옵션으로 **리랭킹 전(hybrid) / 후(rerank) / BM25 단독**을 비교해, 리랭커가 실제로
  성능을 얼마나 끌어올리는지 확인할 수 있습니다.
- 검색은 k=20(기본값)으로 한 번만 수행한 뒤 결과를 잘라서 k=1/3/5/10/20 지표를 한 번에 계산합니다
  (검색 결과는 `retrieval_results.jsonl`에 캐시되어, k만 바꿔 재계산할 때 재검색이 필요 없습니다).

### 2) 답변 품질 평가 — RAGAS

`src/evaluation/eval_ragas.py`에서 [RAGAS](https://github.com/explodinggpt/ragas) 라이브러리로
아래 지표를 자동 계산합니다.

- **Faithfulness**: 답변이 검색된 근거 문서 내용에 충실한지
- **LLMContextPrecisionWithReference**: 검색된 문서가 얼마나 정답과 관련 있는지
- **LLMContextRecall**: 정답에 필요한 정보를 검색이 얼마나 잘 커버했는지
- **ResponseRelevancy**: 답변이 질문과 얼마나 관련 있는지

평가용 질의-정답 세트는 `src/evaluation/eval_samples.json`에 있습니다.

> `ragas`는 `pyproject.toml` 기본 의존성에는 포함되어 있지 않으므로, 평가를 돌리기 전에
> `uv add ragas` 등으로 별도 설치가 필요할 수 있습니다.

## 코드 품질

- **Lint / Format**: `ruff` (`ruff check .`, `ruff format --check .`)
- **Pre-commit**: 커밋 시 trailing-whitespace, YAML/JSON/TOML 검사, ruff 자동 수정 실행
- **CI**: `.py` 파일 push 시 GitHub Actions에서 `ruff check` + `ruff format --check` 자동 실행

## 팀

- 박창준 : 옵시디언 초안구성/전처리/청킹/벡터저장
- 강지연 : 리트리버검색/검증/PM
- 구아롬 : 클라우드관리/LLM생성/검증/Github관리 
- 김효진 : UI앱개발

## 결과물 제출 기한
- Github Repository 링크 : ~ 2026/07/31 19:00
- 보고서(pdf) : ~ 2026/07/31 19:00 (Github Repository Readme에 보고서 파일을 다운로드 할 수 있도록 첨부)
- 협업일지 : ~ 2026/08/03 23:50 (개인 단위로 작성하되, 내용을 확인할 수 있도록 Readme에 링크 또는 pdf)