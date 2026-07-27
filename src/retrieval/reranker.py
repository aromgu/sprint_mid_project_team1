"""
# reranker.py
Qwen3-Reranker-0.6B 기반 리랭커
"""
import torch

from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# 모델/디바이스도 다른 설정값처럼 .env로 바꿀 수 있게 함
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 지금 RFP 데이터셋은 "실제 요구사항(기능/성능/데이터 요구사항 등)"과
# "서약서·서식·청렴서약문 같은 행정 보일러플레이트"가 같은 기관명/사업명을
# 반복해서 포함하고 있어서, 지시문 없이는 리랭커가 이 둘을 잘 구분하지 못하고
# 단순히 기관명·사업명이 많이 겹치는 보일러플레이트 문서에 더 높은 점수를 줄 수 있다.
# -> instruct로 "실제 요구사항 내용"을 우선하도록 명시적으로 알려준다.
RERANK_TASK_INSTRUCTION = (
    "Given a query about a government or public-sector RFP (Request for Proposal) document, "
    "retrieve the passage that describes actual project requirements such as functional, "
    "technical, performance, or data requirements. Do not prefer boilerplate administrative "
    "content such as pledges, non-disclosure oaths, compliance certificates, or form indexes, "
    "even if they repeat the same organization or project name as the query."
)


@lru_cache(maxsize=1)
def get_reranker():
    """CrossEncoder(Qwen3-Reranker-0.6B) 모델을 한 번만 로드해서 재사용한다.

    ※ prompts/default_prompt_name으로 위에서 정의한 RFP 전용 instruct를 심어준다.
      이걸 안 하면 라이브러리 기본값인 "Given a web search query, retrieve relevant
      passages that answer the query" 라는 범용 지시문이 쓰이는데, 이건 "요구사항 문서"와
      "행정 서약서"를 구분하라는 의미가 전혀 없어서 지금 같은 문제가 생기기 쉽다.
    """
    from sentence_transformers import CrossEncoder

    print(f"[reranker] '{RERANKER_MODEL}' 로딩 중... (device={RERANKER_DEVICE})")
    model = CrossEncoder(
        RERANKER_MODEL,
        device=RERANKER_DEVICE,
        prompts={"rfp_requirement": RERANK_TASK_INSTRUCTION},
        default_prompt_name="rfp_requirement",
    )
    print("[reranker] 로딩 완료")
    return model


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 5,
) -> list[Document]:
    if not documents:
        return []

    model = get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda pair: pair[1], reverse=True)

    reranked: list[Document] = []
    for doc, score in scored_docs[:top_n]:

        new_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        reranked.append(new_doc)

    return reranked


def rerank_search_results(
    query: str,
    results: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:

    if not results:
        return []

    model = get_reranker()

    pairs = [(query, r["text"]) for r in results]
    # scores = model.predict(pairs)
    scores = model.predict(pairs, batch_size=32)

    scored = list(zip(results, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked: list[dict[str, Any]] = []
    for r, score in scored[:top_n]:
        r = dict(r)  # 원본 딕셔너리를 건드리지 않도록 복사본 사용
        r["score"] = float(score)  # 리랭커 점수로 score 필드 채움
        reranked.append(r)

    return reranked