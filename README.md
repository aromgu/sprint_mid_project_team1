# Sprint Mid Project Team 1

사내 규정과 제품 정책 문서를 대상으로 검색·재정렬·답변 생성·평가를 수행하는 RAG 프로젝트입니다.

현재 파이프라인은 OpenAI Embedding, Chroma, BM25/Vector Hybrid Retrieval, LLM Reranking, 답변 생성, RAGAS 평가로 구성되어 있습니다.

## Pipeline

```text
문서 로드 및 전처리
        ↓
OpenAI Embedding + Chroma Vector Store
        ↓
Naive Vector / BM25·Vector Hybrid Retrieval
        ↓
Query Rewrite + LLM Reranking
        ↓
답변 생성
        ↓
Hit@K + RAGAS 평가
```

`pipeline.py`는 위 과정을 처음부터 끝까지 실행하고 평가 결과를 `reports/`에 저장합니다.

## Requirements

- Python 3.11 이상
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key

## Setup

저장소를 clone한 후 프로젝트 루트에서 의존성을 설치합니다.

```bash
uv sync --frozen
```

`.env.example`을 복사하고 OpenAI API key를 설정합니다.

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY="your-api-key"
```

`.env`는 Git에서 제외됩니다. API key가 포함된 `.env`를 commit하지 마세요.

선택적으로 pre-commit hook을 설치할 수 있습니다.

```bash
uv run pre-commit install
```

## Run

### 전체 파이프라인

```bash
uv run python pipeline.py
```

전체 파이프라인 실행 후 다음 결과가 생성됩니다.

```text
reports/hit_scoreboard.csv
reports/ragas_evaluation_result.csv
```

결과 CSV까지 터미널에서 연속으로 확인하려면 다음 스크립트를 사용할 수 있습니다.

```bash
bash run_pipe.sh
```

OpenAI Embedding, LLM Reranking, 답변 생성 및 RAGAS 평가 과정에서 API 사용량이 발생합니다.

### 인덱싱

```bash
uv run python scripts/run_indexing.py
```

현재 vector store는 Chroma `EphemeralClient`를 사용하므로 실행 프로세스가 종료되면 유지되지 않습니다.

### 질의응답

```bash
uv run python scripts/run_rag.py
```

터미널에 질문을 입력하면 retrieval, reranking, generation 과정을 거쳐 답변을 출력합니다.

### 평가

```bash
uv run python scripts/run_eval.py
```

내장 평가 데이터로 Hit@K와 RAGAS 평가를 실행하고 결과 CSV를 갱신합니다.

## Project Structure

```text
project-root/
├── configs/
│   ├── config.py                 # 모델, retrieval 및 결과 경로 설정
│   └── prompt.py                 # rewrite, rerank, answer prompt
├── reports/
│   ├── hit_scoreboard.csv        # retrieval 단계별 Hit@K 결과
│   └── ragas_evaluation_result.csv
├── scripts/
│   ├── run_indexing.py           # 문서 embedding 및 vector store 생성
│   ├── run_rag.py                # 대화형 단일 질문 실행
│   └── run_eval.py               # 전체 평가 실행
├── src/
│   ├── loader/                   # 원본 문서 로드
│   ├── preprocessing/            # 텍스트 정제
│   ├── chunking/                 # 문서 분할
│   ├── embeddings/               # OpenAI embedding 및 Chroma 생성
│   ├── retrieval/                # Vector, BM25, Hybrid, LLM reranking
│   ├── generation/               # Query rewrite 및 답변 생성 chain
│   ├── evaluation/               # Hit@K 및 RAGAS 평가
│   ├── agentic/                  # Agentic RAG 실험 코드
│   ├── TMP/                      # 이전 프로토타입 및 참고 코드
│   └── dataset.py                # 내장 Hit@K/RAGAS 평가 데이터
├── .env.example                  # 환경변수 예시
├── .gitignore
├── ExeGuide.md                   # 간단 실행 가이드
├── pipeline.py                   # 전체 파이프라인 진입점
├── pyproject.toml                # 프로젝트 및 의존성 정의
├── run_pipe.sh                   # 파이프라인 실행 및 결과 출력
└── uv.lock                       # 고정된 의존성 버전
```

## Configuration

기본 설정은 `configs/config.py`의 `RAGConfig`에서 관리합니다.

```python
llm_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
naive_k = 3
wide_k = 8
rerank_top_n = 3
```

Prompt는 `configs/prompt.py`에서 관리합니다.

## Evaluation

현재 retrieval 평가는 다음 세 단계를 비교합니다.

- `Naive(Vector)`: vector similarity retrieval
- `Hybrid`: BM25와 vector retrieval 결합
- `Rerank`: 넓게 검색한 후보를 LLM으로 재정렬

RAGAS 평가는 다음 지표를 사용합니다.

- Faithfulness
- Response Relevancy
- LLM Context Precision with Reference
- Context Recall

현재 평가 데이터는 `src/dataset.py`에 있으며, 베이스라인 확정 이후 회귀 평가 구조로 분리할 예정입니다.

## Development Notes

- 실행 명령은 프로젝트 루트 기준으로 작성되어 있습니다.
- `.env`, `.venv`, Python cache 및 FAISS 파일은 Git에서 제외됩니다.
- 파이프라인 구조 개편은 현재 베이스라인 실행 상태를 원격 저장소에 보존한 이후 진행할 예정입니다.
